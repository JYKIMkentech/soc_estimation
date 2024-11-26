clc; clear; close all;

%% Seed setting
%rng(13);

%% Font size settings
axisFontSize = 14;
titleFontSize = 16;
legendFontSize = 12;
labelFontSize = 14;

%% 1. Data load

% ECM parameters (from HPPC test)
load('optimized_params_struct_final_2RC.mat'); % Fields: R0, R1, C1, R2, C2, SOC, avgI, m, Crate

% DRT parameters (gamma and tau values)
load('theta_discrete.mat');
load('gamma_est_all.mat', 'gamma_est_all');  % Modified: Removed SOC_mid_all
load('R0_est_all.mat')

tau_discrete = exp(theta_discrete); % tau values

% SOC-OCV lookup table (from C/20 test)
load('soc_ocv.mat', 'soc_ocv'); % [SOC, OCV]
soc_values = soc_ocv(:, 1);     % SOC values % 1083 x 1
ocv_values = soc_ocv(:, 2);     % Corresponding OCV values [V] % 1083 x 1

% Driving data (17 trips)
load('udds_data.mat'); % Contains structure array 'udds_data' with fields V, I, t, Time_duration, SOC

Q_batt = 2.7742; % [Ah]
SOC_begin_true = 0.9907;
SOC_begin_cc = 0.9907;
current_noise_percent = 0.02;
voltage_noise_percent = 0.01;

[unique_ocv, b] = unique(ocv_values); % unique_ocv : 1029x1
unique_soc = soc_values(b);           % unique_soc : 1029x1  

%% Compute the derivative of OCV with respect to SOC
dOCV_dSOC_values = gradient(unique_ocv) ./ gradient(unique_soc);

% Apply moving average filter to smooth the derivative
windowSize = 10; % Adjust the window size as needed
dOCV_dSOC_values_smooth = movmean(dOCV_dSOC_values, windowSize);

%% 2. Kalman filter setting

% 1 : 1-RC , 2: 2-RC , 3 : DRT

num_RC = length(tau_discrete);

% P
P1_init = [1e-6 0;
             0   1e-15]; % [SOC ; V1] % State covariance
P2_init = [1e-6 0       0;
             0   1e-6    0;
             0   0       1e-13]; % [SOC; V1; V2] % State covariance

P3_init(1,1) = 1e-6;    % Initial covariance for SOC
for i = 2:(1 + num_RC)
    P3_init(i,i) = 1e-6; % Initial covariance for each V_i
end

% Q

Q1 = [1e-17 0;
             0  1e-15];  % [SOC ; V1] % Process covariance

Q2 = [1e-17 0        0;
             0     1e-15    0;
             0      0     1e-15]; % [SOC; V1; V2] % Process covariance

Q3(1,1) = 1e-17; % Process noise for SOC
for i = 2:(1 + num_RC)
    Q3(i,i) = 1e-15; % Process noise for each V_i
end

% R , Measurement covariance

R1 = 5.25e-1;
R2 = 5.25e-1;
R3 = 5.25e-1;

%% 3. Extract ECM parameters

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

%% 4. Apply Kalman filter to all trips

num_trips = length(udds_data);

True_SOC_all = [];   
CC_SOC_all = [];
SOC_est_1RC_all = [];
SOC_est_2RC_all = [];
SOC_est_DRT_all = [];
t_total = [];
I_total = [];
V_total = [];

% Initialize estimation variables for DRT model
SOC_estimate_DRT = SOC_begin_cc;
V_estimate_DRT = zeros(num_RC,1); % Initial V_i values
P_estimate_DRT = P3_init;

% Initialize estimation variables for 1-RC model
SOC_estimate_1RC = SOC_begin_cc;
V1_estimate_1RC = 0;
P_estimate_1RC = P1_init;

% Initialize estimation variables for 2-RC model
SOC_estimate_2RC = SOC_begin_cc;
V1_estimate_2RC = 0;
V2_estimate_2RC = 0;
P_estimate_2RC = P2_init;

t_prev_end = 0; % For cumulative time
t_prev = 0; % For dt calculation

for s = 1 : num_trips-14 % For each trip
    fprintf('Processing Trip %d/%d...\n', s, num_trips-14);
    
    I = udds_data(s).I;
    V = udds_data(s).V;
    t = udds_data(s).t + t_prev_end; % Adjust time to be cumulative
    t_prev_end = t(end); % Update for next trip
    dt = [t(1) - t_prev; diff(t)];
    t_prev = t(end); % Update for next dt calculation

    Time_duration = udds_data(s).Time_duration; % Each trip continues from the previous one

    [noisy_I] = Markov(I,current_noise_percent); % Add Markov noise to current
    noisy_V = V + voltage_noise_percent * V .* randn(size(V)); % Add Gaussian noise to voltage

    True_SOC = SOC_begin_true + cumtrapz(t,I)/(3600 * Q_batt); % True SOC (no noise)
    CC_SOC = SOC_begin_cc + cumtrapz(t,noisy_I)/(3600 * Q_batt); % CC SOC (with noise)

    True_SOC_all = [True_SOC_all; True_SOC];
    CC_SOC_all = [CC_SOC_all; CC_SOC];
    t_total = [t_total; t];
    I_total = [I_total; noisy_I];
    V_total = [V_total; noisy_V];

    SOC_begin_true = True_SOC(end);
    SOC_begin_cc = CC_SOC(end);

    %% DRT

    gamma = gamma_est_all(s,:); % 1x201
    delta_theta = theta_discrete(2) - theta_discrete(1); % 0.0476
    R_i = gamma * delta_theta; % 1x201
    C_i = tau_discrete' ./ R_i; % 1x201

    SOC_est_DRT = zeros(length(t),1);
    V_DRT_est = zeros(length(t), num_RC); % Store each V_i

    for k = 1:length(t) % Prediction and correction from time k-1 to k

        R0 = interp1(SOC_params, R0_params, SOC_estimate_DRT, 'linear', 'extrap');

        % Predict step

        if k == 1 && s == 1
            % Initial prediction of V_i
            V_pred = zeros(num_RC,1);
            for i = 1:num_RC
                V_pred(i) = noisy_I(k) * R_i(i) * (1 - exp(-dt(k) / (R_i(i) * C_i(i))));
            end
        else
            % Predict V_i
            V_pred = zeros(num_RC,1);
            for i = 1:num_RC
                V_pred(i) = V_estimate_DRT(i) * exp(-dt(k) / (R_i(i) * C_i(i))) + noisy_I(k) * R_i(i) * (1 - exp(-dt(k) / (R_i(i) * C_i(i))));
            end
        end

        SOC_pred = SOC_estimate_DRT + (dt(k) / (Q_batt * 3600)) * noisy_I(k);
        
        x_pred = [SOC_pred; V_pred];

        % Predict the error covariance
        A = zeros(1 + num_RC);
        A(1,1) = 1; % SOC
        for i = 1:num_RC
            A(i+1,i+1) = exp(-dt(k) / (R_i(i) * C_i(i)));
        end
        P_pred = A * P_estimate_DRT * A' + Q3;

        % Compute OCV_pred and dOCV_dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');

        % Calculate dOCV_dSOC (using precomputed values)
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values, SOC_pred, 'linear', 'extrap');

        % Measurement matrix H
        H = zeros(1, 1 + num_RC);
        H(1) = dOCV_dSOC;
        H(2:end) = ones(1, num_RC);

        % Compute the predicted voltage
        V_pred_total = OCV_pred + sum(V_pred) + R0 * noisy_I(k);

        % Compute the Kalman gain
        S = H * P_pred * H' + R3; % Measurement noise covariance
        K = P_pred * H' / S;

        % Update the estimate
        z = noisy_V(k); % Measurement
        x_estimate = x_pred + K * (z - V_pred_total);

        % Update the error covariance
        P_estimate_DRT = (eye(1 + num_RC) - K * H) * P_pred;

        % Store the estimates
        SOC_est_DRT(k) = x_estimate(1);
        V_estimate_DRT = x_estimate(2:end);

        V_DRT_est(k, :) = V_estimate_DRT'; % Save V1,V2,V3,...V201

        % Update the estimates for next iteration
        SOC_estimate_DRT = x_estimate(1);
    end

    SOC_est_DRT_all = [SOC_est_DRT_all; SOC_est_DRT]; 

    %% 1-RC

    SOC_est_1RC =  zeros(length(t), 1);
    V1_est_1RC_vec = zeros(length(t), 1);

    for k = 1:length(t)

        % Compute R0, R1, C1 at SOC_estimate
        R0 = interp1(SOC_params, R0_params, SOC_estimate_1RC, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate_1RC, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate_1RC, 'linear', 'extrap');

        % Predict step

        if k == 1 && s == 1
            % Initial prediction of V1
            V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
        else
            % Predict V1
            V1_pred = V1_estimate_1RC * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
        end

        % Predict SOC
        SOC_pred = SOC_estimate_1RC + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % Form the predicted state vector
        x_pred = [SOC_pred; V1_pred];

        % Predict the error covariance
        A = [1 0;
             0 exp(-dt(k) / (R1 * C1))];
        P_pred = A * P_estimate_1RC * A' + Q1;

        % Compute OCV_pred and dOCV_dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values, SOC_pred, 'linear', 'extrap');

        % Measurement matrix H
        H = [dOCV_dSOC, 1];

        % Compute the predicted voltage
        V_pred = OCV_pred + V1_pred + R0 * noisy_I(k);

        % Compute the Kalman gain
        S = H * P_pred * H' + R1; % Measurement noise covariance
        K = P_pred * H' / S;

        % Update the estimate
        z = noisy_V(k); % Measurement
        x_estimate = x_pred + K * (z - V_pred);

        % Update the error covariance
        P_estimate_1RC = (eye(2) - K * H) * P_pred;

        % Store the estimates
        SOC_est_1RC(k) = x_estimate(1);
        V1_est_1RC_vec(k) = x_estimate(2);

        % Update the estimates for next iteration
        SOC_estimate_1RC = x_estimate(1);
        V1_estimate_1RC = x_estimate(2);
    end

    SOC_est_1RC_all = [SOC_est_1RC_all; SOC_est_1RC];

    %% 2-RC

    SOC_est_2RC = zeros(length(t),1);
    V1_est_2RC_vec = zeros(length(t),1);
    V2_est_2RC_vec = zeros(length(t),1);

    for k = 1:length(t)

        % Compute R0, R1, C1, R2, C2 at SOC_estimate
        R0 = interp1(SOC_params, R0_params, SOC_estimate_2RC, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate_2RC, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate_2RC, 'linear', 'extrap');
        R2 = interp1(SOC_params, R2_params, SOC_estimate_2RC, 'linear', 'extrap');
        C2 = interp1(SOC_params, C2_params, SOC_estimate_2RC, 'linear', 'extrap');

        % Predict step

        if k == 1 && s == 1
            % Initial prediction of V1 and V2
            V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            V2_pred = noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
        else
            % Predict V1 and V2
            V1_pred = V1_estimate_2RC * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            V2_pred = V2_estimate_2RC * exp(-dt(k) / (R2 * C2)) + noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
        end

        % Predict SOC
        SOC_pred = SOC_estimate_2RC + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % Form the predicted state vector
        x_pred = [SOC_pred; V1_pred; V2_pred];

        % Predict the error covariance
        A = [1 0 0;
             0 exp(-dt(k) / (R1 * C1)) 0;
             0 0 exp(-dt(k) / (R2 * C2))];
        P_pred = A * P_estimate_2RC * A' + Q2;

        % Compute OCV_pred and dOCV_dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc,dOCV_dSOC_values, SOC_pred, 'linear', 'extrap');

        % Measurement matrix H
        H = [dOCV_dSOC, 1, 1];

        % Compute the predicted voltage
        V_pred = OCV_pred + V1_pred + V2_pred + R0 * noisy_I(k);

        % Compute the Kalman gain
        S = H * P_pred * H' + R2; % Measurement noise covariance
        K = P_pred * H' / S;

        % Update the estimate
        z = noisy_V(k); % Measurement
        x_estimate = x_pred + K * (z - V_pred);

        % Update the error covariance
        P_estimate_2RC = (eye(3) - K * H) * P_pred;

        % Store the estimates
        SOC_est_2RC(k) = x_estimate(1);
        V1_est_2RC_vec(k) = x_estimate(2);
        V2_est_2RC_vec(k) = x_estimate(3);

        % Update the estimates for next iteration
        SOC_estimate_2RC = x_estimate(1);
        V1_estimate_2RC = x_estimate(2);
        V2_estimate_2RC = x_estimate(3);
    end

    SOC_est_2RC_all = [SOC_est_2RC_all; SOC_est_2RC];  

end

%% Plot the results over the entire time
figure;
plot(t_total, CC_SOC_all, 'b', 'LineWidth', 1.5);
hold on;
plot(t_total, True_SOC_all, 'k--', 'LineWidth', 1.5);
plot(t_total, SOC_est_1RC_all, 'r', 'LineWidth', 1.5);
plot(t_total, SOC_est_2RC_all, 'g', 'LineWidth', 1.5);
plot(t_total, SOC_est_DRT_all, 'm-', 'LineWidth', 1.5);

xlabel('Time [s]');
ylabel('SOC');
legend('Coulomb Counting SOC', 'True SOC', 'Estimated SOC (1-RC)', 'Estimated SOC (2-RC)', 'Estimated SOC (DRT)');
title('SOC Estimation Over Entire Time');
grid on;

%% Function for adding Markov noise
function [noisy_I] = Markov(I, noise_percent)

    noise_number = 50;
    initial_state = randsample(1:noise_number, 1); % Random initial state
    mean_noise = mean(I) * noise_percent;
    min_noise = min(I) * noise_percent; % min(I) = -4.8 A --> -0.0048 A
    max_noise = max(I) * noise_percent; % max(I) = 3.1 A --> 0.0031 A
    span = max_noise - min_noise; % span = 0.0079 A
    sigma = span / noise_number; % sigma = 1.6e-4
    noise_vector = linspace(mean_noise - span/2, mean_noise + span/2, noise_number); % Range (-0.0434 , .... , 0.0353)
    P = zeros(noise_number);

    for i = 1:noise_number
        probabilities = normpdf(noise_vector, noise_vector(i), sigma); % P(i,i) is highest
        P(i, :) = probabilities / sum(probabilities); % Sum to 1
    end

    noisy_I = zeros(size(I));
    states = zeros(size(I));
    current_state = initial_state;
    
    for m = 1:length(I)
        noisy_I(m) = I(m) + noise_vector(current_state); % Add noise current corresponding to random state
        states(m) = current_state;
        current_state = randsample(1:noise_number, 1 , true, P(current_state, :)); % Sample with replacement
    end

end
