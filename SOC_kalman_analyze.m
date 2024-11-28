clc; clear; close all;

%% Seed Setting
%rng(13); 

%% Font Size Settings
axisFontSize = 14;
titleFontSize = 16;
legendFontSize = 12;
labelFontSize = 14;

%% 1. Data Load

% ECM Parameters (from HPPC test)
load('optimized_params_struct_final_2RC.mat'); % Fields: R0, R1, C1, R2, C2, SOC, avgI, m, Crate

% DRT Parameters (gamma and tau values)
load('theta_discrete.mat');
load('gamma_est_all.mat', 'gamma_est_all');  % Modified part: Removed SOC_mid_all
load('R0_est_all.mat');

tau_discrete = exp(theta_discrete); % tau values

% SOC-OCV Lookup Table (from C/20 test)
load('soc_ocv.mat', 'soc_ocv'); % [SOC, OCV]
soc_values = soc_ocv(:, 1);     % SOC values % 1083 x 1
ocv_values = soc_ocv(:, 2);     % Corresponding OCV values [V] % 1083 x 1

% Driving Data (17 trips)
load('udds_data.mat'); % Struct array 'udds_data' containing fields V, I, t, Time_duration, SOC

Q_batt = 2.7742; % [Ah]
SOC_begin_true = 0.9907;
SOC_begin_cc = 0.9907;
current_noise_percent = 0.02;
voltage_noise_percent = 0.01;

[unique_ocv, b] = unique(ocv_values); % unique_soc : 1029x1
unique_soc = soc_values(b);           % unique_ocv : 1029x1  

%% Compute the Derivative of OCV with Respect to SOC
dOCV_dSOC_values = gradient(unique_ocv) ./ gradient(unique_soc);

windowSize = 10; 
dOCV_dSOC_values_smooth = movmean(dOCV_dSOC_values, windowSize);

%% 2. Kalman Filter Settings

% 1 : 1-RC, 2: 2-RC, 3 : DRT

num_RC = length(tau_discrete);

% Initialize P
P1_init = [1e-5 0;
           0   1e-5]; % [SOC ; V1] State Covariance

P2_init = [1e-3 0       0;
           0   1e-3   0;
           0   0       1e-3]; % [SOC; V1; V2] State Covariance

P3_init = zeros(1 + num_RC); % DRT Model State Covariance
P3_init(1,1) = 1e-7;    % Initial covariance for SOC
for i = 2:(1 + num_RC)
    P3_init(i,i) = 1e-7; % Initial covariance for each V_i
end

% Q Process Noise Covariance
Q1 = [1e-4 0;
      0    1e-5];  % [SOC ; V1] Process Noise

Q2 = [1e-4  0        0;
      0     1e-7    0;
      0      0     1e-7]; % [SOC; V1; V2] Process Noise

Q3 = zeros(1 + num_RC);
Q3(1,1) = 1e-10; % Process noise for SOC
for i = 2:(1 + num_RC)
    Q3(i,i) = 1e-10; % Process noise for each V_i
end

% R Measurement Noise Covariance
R1 = 5.25e-6;
R2 = 5.25e-6;
R3 = 5.25e-6;

%% 3. Extract ECM Parameters

num_params = length(optimized_params_struct_final_2RC);
SOC_params = zeros(num_params, 1);
R0_params = zeros(num_params, 1);
R1_params = zeros(num_params, 1);
R2_params = zeros(num_params, 1);
C1_params = zeros(num_params, 1);
C2_params = zeros(num_params, 1);

for i = 1:num_params
    SOC_params(i) = optimized_params_struct_final_2RC(i).SOC;
    R0_params(i) = optimized_params_struct_final_2RC(i).R0;
    R1_params(i) = optimized_params_struct_final_2RC(i).R1;
    R2_params(i) = optimized_params_struct_final_2RC(i).R2;
    C1_params(i) = optimized_params_struct_final_2RC(i).C1;
    C2_params(i) = optimized_params_struct_final_2RC(i).C2;
end

%% 4. Apply Kalman Filter to All Trips

num_trips = length(udds_data);

% Initialize cell arrays to store results
True_SOC_all = cell(num_trips, 1);   
CC_SOC_all = cell(num_trips, 1);
SOC_est_1RC_all = cell(num_trips, 1);
SOC_est_2RC_all = cell(num_trips, 1);
SOC_est_DRT_all = cell(num_trips, 1);

% Additional cell arrays to store predictions, estimates, and Kalman gains
x_pred_1RC_all = cell(num_trips, 1);
x_estimate_1RC_all = cell(num_trips, 1);
K_1RC_all = cell(num_trips, 1);

x_pred_2RC_all = cell(num_trips, 1);
x_estimate_2RC_all = cell(num_trips, 1);
K_2RC_all = cell(num_trips, 1);

x_pred_DRT_all = cell(num_trips, 1);
x_estimate_DRT_all = cell(num_trips, 1);
K_DRT_all = cell(num_trips, 1);

epsilon_percent_span = 0.1; % ±5% epsilon percentages
sigma_percent = 0.001;        % 1% standard deviation

for s = 1 : num_trips-16 % For each trip
    fprintf('Processing Trip %d/%d...\n', s, num_trips-16);

    I = udds_data(s).I;
    V = udds_data(s).V;
    t = udds_data(s).t; % All trips start at 0 seconds
    dt = [t(1); diff(t)];
    dt(1) = dt(2);
    Time_duration = udds_data(s).Time_duration; % All trips are consecutive

    [noisy_I] = Markov(I, epsilon_percent_span, sigma_percent); % Updated function call
    noisy_V = V + voltage_noise_percent * V .* randn(size(V)); % Add Gaussian noise to voltage

    True_SOC = SOC_begin_true + cumtrapz(t,I)/(3600 * Q_batt); % True SOC (no noise)
    CC_SOC = SOC_begin_cc + cumtrapz(t,noisy_I)/(3600 * Q_batt); % CC SOC (with noise)

    True_SOC_all{s} = True_SOC;
    CC_SOC_all{s} = CC_SOC;

    SOC_begin_true = True_SOC(end);
    SOC_begin_cc = CC_SOC(end);

    %% DRT Model

    gamma = gamma_est_all(s,:); % 1x201
    delta_theta = theta_discrete(2) - theta_discrete(1); % 0.0476
    R_i = gamma * delta_theta; % 1x201
    C_i = tau_discrete' ./ R_i; % 1x201

    SOC_estimate = CC_SOC(1);
    V_estimate = zeros(num_RC,1); % Initial values for V_i
    P_estimate = P3_init;
    SOC_est_DRT = zeros(length(t),1);
    V_DRT_est = zeros(length(t), num_RC); % Store each V_i

    % Initialize arrays to store predictions, estimates, and Kalman gains
    x_pred_DRT = zeros(length(t), 1 + num_RC);      % [SOC_pred, V_pred_RC]
    x_estimate_DRT = zeros(length(t), 1 + num_RC);  % [SOC_estimate, V_estimate_RC]
    K_DRT = zeros(1 + num_RC, length(t));           % Kalman Gain

    for k = 1:length(t) % Prediction and correction for each time step

        R0 = interp1(SOC_params, R0_params, SOC_estimate, 'linear', 'extrap');

        % Prediction Step
        if k == 1
            % Initial prediction of V_i
            V_pred = zeros(num_RC,1);
            for i = 1:num_RC
                V_pred(i) = noisy_I(k) * R_i(i) * (1 - exp(-dt(k) / (R_i(i) * C_i(i))));
            end
        else
            % Predict V_i
            V_pred = zeros(num_RC,1);
            for i = 1:num_RC
                V_pred(i) = V_estimate(i) * exp(-dt(k) / (R_i(i) * C_i(i))) + ...
                           noisy_I(k) * R_i(i) * (1 - exp(-dt(k) / (R_i(i) * C_i(i))));
            end
        end

        SOC_pred = SOC_estimate + (dt(k) / (Q_batt * 3600)) * noisy_I(k);
        
        x_pred = [SOC_pred; V_pred];

        % Store the predicted state
        x_pred_DRT(k, :) = x_pred';

        % Predict the error covariance
        A = zeros(1 + num_RC);
        A(1,1) = 1; % SOC
        for i = 1:num_RC
            A(i+1,i+1) = exp(-dt(k) / (R_i(i) * C_i(i)));
        end
        P_pred = A * P_estimate * A' + Q3;

        % Predict OCV and compute dOCV/dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred, 'linear', 'extrap');

        % Measurement Matrix H
        H = zeros(1, 1 + num_RC);
        H(1) = dOCV_dSOC;
        H(2:end) = ones(1, num_RC);

        % Compute the predicted total voltage
        V_pred_total = OCV_pred + sum(V_pred) + R0 * noisy_I(k);

        % Calculate Kalman Gain
        S = H * P_pred * H' + R3; % Measurement noise covariance
        K = P_pred * H' / S;

        % Store Kalman Gain
        K_DRT(:, k) = K;

        % Update Step
        z = noisy_V(k); % Measurement
        x_estimate = x_pred + K * (z - V_pred_total);

        % Store the estimated state
        x_estimate_DRT(k, :) = x_estimate';

        % Update the error covariance
        P_estimate = (eye(1 + num_RC) - K * H) * P_pred;

        % Store SOC and V_i
        SOC_est_DRT(k) = x_estimate(1);
        V_estimate = x_estimate(2:end);

        V_DRT_est(k, :) = V_estimate'; % Store V1, V2, ..., V201

        % Update estimate for next iteration
        SOC_estimate = x_estimate(1);
    end

    SOC_est_DRT_all{s} = SOC_est_DRT; 
    x_pred_DRT_all{s} = x_pred_DRT;
    x_estimate_DRT_all{s} = x_estimate_DRT;
    K_DRT_all{s} = K_DRT;

    %% 1-RC Model

    SOC_est_1RC =  zeros(length(t), 1);
    V1_est_1RC = zeros(length(t), 1);

    % Initialize arrays to store predictions, estimates, and Kalman gains
    x_pred_1RC = zeros(length(t), 2);        % [SOC_pred, V1_pred]
    x_estimate_1RC = zeros(length(t), 2);    % [SOC_estimate, V1_estimate]
    K_1RC = zeros(2, length(t));             % Kalman Gain

    SOC_estimate = CC_SOC(1);
    P_estimate = P1_init;

    for k = 1:length(t)

        % Calculate R0, R1, C1 from SOC_estimate
        R0 = interp1(SOC_params, R0_params, SOC_estimate, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate, 'linear', 'extrap');

        % Prediction Step
        if k == 1
            % Initial prediction of V1
            V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
        else
            % Predict V1
            V1_pred = V_estimate(1) * exp(-dt(k) / (R1 * C1)) + ...
                      noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
        end

        % Predict SOC
        SOC_pred = SOC_estimate + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % Predicted state vector
        x_pred = [SOC_pred; V1_pred];

        % Store the predicted state
        x_pred_1RC(k, :) = x_pred';

        % Predict the error covariance
        A = [1 0;
             0 exp(-dt(k) / (R1 * C1))];
        P_pred = A * P_estimate * A' + Q1;

        % Predict OCV and compute dOCV/dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred, 'linear', 'extrap');

        % Measurement Matrix H
        H = [dOCV_dSOC, 1];

        % Compute the predicted total voltage
        V_pred_total = OCV_pred + V1_pred + R0 * noisy_I(k);

        % Calculate Kalman Gain
        S = H * P_pred * H' + R1; % Measurement noise covariance
        K = P_pred * H' / S;

        % Store Kalman Gain
        K_1RC(:, k) = K;

        % Update Step
        z = noisy_V(k); % Measurement
        x_estimate = x_pred + K * (z - V_pred_total);

        % Store the estimated state
        x_estimate_1RC(k, :) = x_estimate';

        % Update the error covariance
        P_estimate = (eye(2) - K * H) * P_pred;

        % Store SOC and V1
        SOC_est_1RC(k) = x_estimate(1);
        V1_est_1RC(k) = x_estimate(2);

        % Update estimate for next iteration
        SOC_estimate = x_estimate(1);
    end

    SOC_est_1RC_all{s} = SOC_est_1RC;
    x_pred_1RC_all{s} = x_pred_1RC;
    x_estimate_1RC_all{s} = x_estimate_1RC;
    K_1RC_all{s} = K_1RC;

    %% 2-RC Model

    SOC_est_2RC = zeros(length(t),1);
    V1_est_2RC = zeros(length(t),1);
    V2_est_2RC = zeros(length(t),1);

    % Initialize arrays to store predictions, estimates, and Kalman gains
    x_pred_2RC = zeros(length(t), 3);        % [SOC_pred, V1_pred, V2_pred]
    x_estimate_2RC = zeros(length(t), 3);    % [SOC_estimate, V1_estimate, V2_estimate]
    K_2RC = zeros(3, length(t));             % Kalman Gain

    SOC_estimate = CC_SOC(1);
    P_estimate = P2_init;

    for k = 1:length(t)

        % Calculate R0, R1, C1, R2, C2 from SOC_estimate
        R0 = interp1(SOC_params, R0_params, SOC_estimate, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate, 'linear', 'extrap');
        R2 = interp1(SOC_params, R2_params, SOC_estimate, 'linear', 'extrap');
        C2 = interp1(SOC_params, C2_params, SOC_estimate, 'linear', 'extrap');

        % Prediction Step
        if k == 1
            % Initial prediction of V1 and V2
            V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            V2_pred = noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
        else
            % Predict V1 and V2
            V1_pred = V_estimate(1) * exp(-dt(k) / (R1 * C1)) + ...
                      noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            V2_pred = V_estimate(2) * exp(-dt(k) / (R2 * C2)) + ...
                      noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
        end

        % Predict SOC
        SOC_pred = SOC_estimate + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % Predicted state vector
        x_pred = [SOC_pred; V1_pred; V2_pred];

        % Store the predicted state
        x_pred_2RC(k, :) = x_pred';

        % Predict the error covariance
        A = [1 0 0;
             0 exp(-dt(k) / (R1 * C1)) 0;
             0 0 exp(-dt(k) / (R2 * C2))];
        P_pred = A * P_estimate * A' + Q2;

        % Predict OCV and compute dOCV/dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc,dOCV_dSOC_values_smooth, SOC_pred, 'linear', 'extrap');

        % Measurement Matrix H
        H = [dOCV_dSOC, 1, 1];

        % Compute the predicted total voltage
        V_pred_total = OCV_pred + V1_pred + V2_pred + R0 * noisy_I(k);

        % Calculate Kalman Gain
        S = H * P_pred * H' + R2; % Measurement noise covariance
        K = P_pred * H' / S;

        % Store Kalman Gain
        K_2RC(:, k) = K;

        % Update Step
        z = noisy_V(k); % Measurement
        x_estimate = x_pred + K * (z - V_pred_total);

        % Store the estimated state
        x_estimate_2RC(k, :) = x_estimate';

        % Update the error covariance
        P_estimate = (eye(3) - K * H) * P_pred;

        % Store SOC, V1, V2
        SOC_est_2RC(k) = x_estimate(1);
        V1_est_2RC(k) = x_estimate(2);
        V2_est_2RC(k) = x_estimate(3);

        % Update estimate for next iteration
        SOC_estimate = x_estimate(1);
    end

    SOC_est_2RC_all{s} = SOC_est_2RC;  
    x_pred_2RC_all{s} = x_pred_2RC;
    x_estimate_2RC_all{s} = x_estimate_2RC;
    K_2RC_all{s} = K_2RC;

end

%% Plotting Graphs for Example Trip (First Trip)
s = 1; % Select the first trip

% Graph for 1-RC Model
figure('Name', '1-RC Model', 'NumberTitle', 'off');

% SOC Related Subplot
subplot(2,1,1);
plot(udds_data(s).t, True_SOC_all{s}, 'k--', 'LineWidth', 0.5);         % True SOC
hold on;
plot(udds_data(s).t, x_pred_1RC_all{s}(:, 1), 'b-', 'LineWidth', 0.5);  % Predicted SOC
plot(udds_data(s).t, x_estimate_1RC_all{s}(:, 1), 'r-', 'LineWidth', 0.5); % Estimated SOC
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('SOC', 'FontSize', labelFontSize);
legend('True SOC', 'Predicted SOC', 'Estimated SOC', 'FontSize', legendFontSize);
title('SOC Estimation (1-RC Model)', 'FontSize', titleFontSize);
grid on;

% Kalman Gain Subplot
subplot(2,1,2);
plot(udds_data(s).t, K_1RC_all{s}(1, :), 'b-', 'LineWidth', 0.5);  % Kalman Gain for SOC
hold on;
plot(udds_data(s).t, K_1RC_all{s}(2, :), 'r-', 'LineWidth', 0.5);  % Kalman Gain for V1
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('Kalman Gain', 'FontSize', labelFontSize);
legend('Kalman Gain for SOC', 'Kalman Gain for V1', 'FontSize', legendFontSize);
title('Kalman Gain (1-RC Model)', 'FontSize', titleFontSize);
grid on;

% Graph for 2-RC Model
figure('Name', '2-RC Model', 'NumberTitle', 'off');

% SOC Related Subplot
subplot(2,1,1);
plot(udds_data(s).t, True_SOC_all{s}, 'k--', 'LineWidth', 0.5);         % True SOC
hold on;
plot(udds_data(s).t, x_pred_2RC_all{s}(:, 1), 'b-', 'LineWidth',0.5);  % Predicted SOC
plot(udds_data(s).t, x_estimate_2RC_all{s}(:, 1), 'r-', 'LineWidth', 0.5); % Estimated SOC
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('SOC', 'FontSize', labelFontSize);
legend('True SOC', 'Predicted SOC', 'Estimated SOC', 'FontSize', legendFontSize);
title('SOC Estimation (2-RC Model)', 'FontSize', titleFontSize);
grid on;

% Kalman Gain Subplot
subplot(2,1,2);
plot(udds_data(s).t, K_2RC_all{s}(1, :), 'b-', 'LineWidth', 0.5);  % Kalman Gain for SOC
hold on;
plot(udds_data(s).t, K_2RC_all{s}(2, :), 'r-', 'LineWidth', 0.5);  % Kalman Gain for V1
plot(udds_data(s).t, K_2RC_all{s}(3, :), 'g-', 'LineWidth', 0.5);  % Kalman Gain for V2
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('Kalman Gain', 'FontSize', labelFontSize);
legend('Kalman Gain for SOC', 'Kalman Gain for V1', 'Kalman Gain for V2', 'FontSize', legendFontSize);
title('Kalman Gain (2-RC Model)', 'FontSize', titleFontSize);
grid on;

% Graph for DRT Model
figure('Name', 'DRT Model', 'NumberTitle', 'off');

% SOC Related Subplot
subplot(2,1,1);
plot(udds_data(s).t, True_SOC_all{s}, 'k--', 'LineWidth', 1.5);         % True SOC
hold on;
plot(udds_data(s).t, x_pred_DRT_all{s}(:, 1), 'b-', 'LineWidth', 1.5);  % Predicted SOC
plot(udds_data(s).t, x_estimate_DRT_all{s}(:, 1), 'r-', 'LineWidth', 1.5); % Estimated SOC
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('SOC', 'FontSize', labelFontSize);
legend('True SOC', 'Predicted SOC', 'Estimated SOC', 'FontSize', legendFontSize);
title('SOC Estimation (DRT Model)', 'FontSize', titleFontSize);
grid on;

% Kalman Gain Subplot
subplot(2,1,2);
plot(udds_data(s).t, K_DRT_all{s}(1, :), 'b-', 'LineWidth', 1.5);  % Kalman Gain for SOC
xlabel('Time [s]', 'FontSize', labelFontSize);
ylabel('Kalman Gain', 'FontSize', labelFontSize);
legend('Kalman Gain for SOC', 'FontSize', legendFontSize);
title('Kalman Gain (DRT Model)', 'FontSize', titleFontSize);
grid on;

%% Plot Example for the First Trip
figure;
plot(udds_data(1).t, CC_SOC_all{1}, 'b', 'LineWidth', 1.5);
hold on;
plot(udds_data(1).t, True_SOC_all{1}, 'k--', 'LineWidth', 1.5);
plot(udds_data(1).t, SOC_est_1RC_all{1}, 'r', 'LineWidth', 1.5);
plot(udds_data(1).t, SOC_est_2RC_all{1}, 'g', 'LineWidth', 1.5);
plot(udds_data(1).t, SOC_est_DRT_all{1}, 'm-', 'LineWidth', 1.5);

xlabel('Time [s]');
ylabel('SOC');
legend('Coulomb Counting SOC', 'True SOC', 'Estimated SOC (1-RC)', 'Estimated SOC (2-RC)', 'Estimated SOC (DRT)');
title('SOC Estimation');
grid on;


%% Function to Add Markov Noise to the Current
function [noisy_I] = Markov(I, epsilon_percent_span, sigma_percent)

    N = 51; % Number of states in the Markov chain
    epsilon_vector = linspace(-epsilon_percent_span/2, epsilon_percent_span/2, N); % From -5% to +5%
    sigma = sigma_percent; % Standard deviation in percentage (e.g., 0.01 for 1%)

    % Initialize the transition probability matrix
    P = zeros(N);
    for i = 1:N
        % Compute the transition probabilities to all other states
        probabilities = normpdf(epsilon_vector, epsilon_vector(i), sigma);
        P(i, :) = probabilities / sum(probabilities); % Normalize to sum to 1
    end

    % Initialize the Markov chain
    initial_state = randsample(1:N, 1); % Randomly select initial state
    current_state = initial_state;

    % Generate the noisy current
    noisy_I = zeros(size(I));
    for k = 1:length(I)
        epsilon = epsilon_vector(current_state);
        noisy_I(k) = I(k) * (1 + epsilon); % Apply the epsilon percentage

        % Transition to the next state
        current_state = randsample(1:N, 1, true, P(current_state, :));
    end
end
