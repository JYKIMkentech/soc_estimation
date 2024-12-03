clc; clear; close all;

%% seed setting
rng(13);

%% Font size settings
axisFontSize = 14;
titleFontSize = 16;
legendFontSize = 12;
labelFontSize = 14;

%% Define Color Matrix
c_mat = lines(9);  

%% 1. Data load

% ECM parameters
load('optimized_params_struct_final_2RC.mat'); % Fields: R0, R1, C1, R2, C2, SOC, avgI, m, Crate

% DRT parameters 
load('theta_discrete.mat');
load('gamma_est_all.mat', 'gamma_est_all'); 
load('R0_est_all.mat')

tau_discrete = exp(theta_discrete); 

% SOC-OCV lookup table (from C/20 test)
load('soc_ocv.mat', 'soc_ocv'); % [SOC, OCV]
soc_values = soc_ocv(:, 1);     % SOC values
ocv_values = soc_ocv(:, 2);     % Corresponding OCV values [V]

load('udds_data.mat'); % Struct array 'udds_data' containing fields V, I, t, Time_duration, SOC

Q_batt = 2.7742 ; % [Ah]
SOC_begin_true = 0.9907;
SOC_begin_cc = 0.9907;
epsilon_percent_span = 0.2;
voltage_noise_percent = 0.01;

[unique_ocv, b] = unique(ocv_values);
unique_soc = soc_values(b);

% Compute the derivative of OCV with respect to SOC
dOCV_dSOC_values = gradient(unique_ocv) ./ gradient(unique_soc);
windowSize = 10; 
dOCV_dSOC_values_smooth = movmean(dOCV_dSOC_values, windowSize);

%% 2. Kalman filter settings
Pcov1_init = [4e-14 0;
              0   16e-4]; % [SOC ; V1] % State covariance

% Q
Qcov1 = [4e-14 0;
         0  16e-4];  % [SOC ; V1] % Process covariance

% R , Measurement covariance
Rcov1 = 5.25e-6;

%% 2. Kalman filter settings for 2-RC model

Pcov2_init = [1e-16 0 0;
              0 16e-8 0;
              0 0 16e-8]; % [SOC ; V1; V2] % State covariance

% Q
Qcov2 = [1e-13 0 0;
         0 16e-8 0;
         0 0 16e-8];  % [SOC ; V1; V2] % Process covariance

% R , Measurement covariance
Rcov2 = 5.25e-6;

%% 3. Extract ECM parameters
SOC_params = vertcat(optimized_params_struct_final_2RC.SOC);
R0_params = vertcat(optimized_params_struct_final_2RC.R0);
R1_params = vertcat(optimized_params_struct_final_2RC.R1);
C1_params = vertcat(optimized_params_struct_final_2RC.C1);
R2_params = vertcat(optimized_params_struct_final_2RC.R2);
C2_params = vertcat(optimized_params_struct_final_2RC.C2);

%% 4. Kalman filter

num_trips = length(udds_data);

% Initialize SOC,V1,P for 1RC model
initial_SOC_true = SOC_begin_true;
initial_SOC_cc = SOC_begin_cc;
initial_P_estimate_1RC = Pcov1_init;
initial_SOC_estimate_1RC = initial_SOC_cc;
initial_V1_estimate_1RC = 0;

% Initialize data over all trips for 1RC
t_all = [];
True_SOC_all = [];
CC_SOC_all = [];
x_pred_1RC_all_trips = [];
x_estimate_1RC_all_trips = [];
KG_1RC_all_trips = [];
residual_1RC_all_trips = [];
I_all = [];
noisy_I_all = [];

% Initialize SOC,V1,V2,P for 2RC model
initial_P_estimate_2RC = Pcov2_init;
initial_SOC_estimate_2RC = initial_SOC_cc;
initial_V1_estimate_2RC = 0;
initial_V2_estimate_2RC = 0;

% Initialize data over all trips for 2RC
x_pred_2RC_all_trips = [];
x_estimate_2RC_all_trips = [];
KG_2RC_all_trips = [];
residual_2RC_all_trips = [];

% Initialize previous trip's end time
previous_trip_end_time = 0;

for s = 1:num_trips-1 % 트립 수에 대하여
    fprintf('Processing Trip %d/%d...\n', s, num_trips);

    % Use Time_duration as time vector
    t = udds_data(s).Time_duration;
    I = udds_data(s).I;
    V = udds_data(s).V;

    % Calculate dt
    if s == 1
        dt = [t(1); diff(t)];      % 첫 번째 트립: t(1)을 앞에 추가
        dt(1) = dt(2);             % dt(1)을 dt(2)로 설정
    else
        dt = [t(1) - previous_trip_end_time; diff(t)]; % 나머지 트립: 이전 트립 끝 시간과 현재 트립 첫 시간의 차이
    end

    
    fprintf('Trip %d, dt(1): %f\n', s, dt(1));

    [noisy_I] = Markov(I, epsilon_percent_span); % 전류: 마르코프 노이즈
    noisy_V = V + voltage_noise_percent * V .* randn(size(V)); % 전압: 가우시안 노이즈

    % Compute True SOC and CC SOC
    True_SOC = initial_SOC_true + cumtrapz(t - t(1), I)/(3600 * Q_batt);
    CC_SOC = initial_SOC_cc + cumtrapz(t - t(1), noisy_I)/(3600 * Q_batt);

    % 1-RC 

    SOC_est_1RC = zeros(length(t), 1);
    V1_est_1RC = zeros(length(t), 1);   

    SOC_estimate_1RC = initial_SOC_estimate_1RC;
    P_estimate_1RC = initial_P_estimate_1RC;
    V1_estimate_1RC = initial_V1_estimate_1RC;

    % Kalman filter variables for 1RC
    x_pred_1RC_all = zeros(length(t), 2);     % Predicted states
    KG_1RC_all = zeros(length(t), 2);         % Kalman gains
    residual_1RC_all = zeros(length(t), 1);   % Residuals
    x_estimate_1RC_all = zeros(length(t), 2); % Updated states

    % 2-RC

    SOC_est_2RC = zeros(length(t), 1);
    V1_est_2RC = zeros(length(t), 1);
    V2_est_2RC = zeros(length(t), 1);

    SOC_estimate_2RC = initial_SOC_estimate_2RC;
    V1_estimate_2RC = initial_V1_estimate_2RC;
    V2_estimate_2RC = initial_V2_estimate_2RC;
    P_estimate_2RC = initial_P_estimate_2RC;

    % Kalman filter variables for 2RC
    x_pred_2RC_all = zeros(length(t), 3);     % Predicted states
    KG_2RC_all = zeros(length(t), 3);         % Kalman gains
    residual_2RC_all = zeros(length(t), 1);   % Residuals
    x_estimate_2RC_all = zeros(length(t), 3); % Updated states

    for k = 1:length(t) % 각 시간 스텝에 대해

        % Interpolate R0, R1, C1 based on SOC for 1RC
        R0 = interp1(SOC_params, R0_params, SOC_estimate_1RC, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate_1RC, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate_1RC, 'linear', 'extrap');

        %% Predict step for 1RC
        if k == 1
            if s == 1
                V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            else
                V1_pred = V1_estimate_1RC * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            end
        else 
            V1_pred = V1_estimate_1RC * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
        end

        % Predict SOC for 1RC
        SOC_pred_1RC = SOC_estimate_1RC + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % Form the predicted state vector for 1RC
        x_pred = [SOC_pred_1RC; V1_pred];

        % Store the predicted state vector for 1RC
        x_pred_1RC_all(k, :) = x_pred';

        % Predict the error covariance for 1RC
        A = [1 0;
             0 exp(-dt(k) / (R1 * C1))];
        P_pred_1RC = A * P_estimate_1RC * A' + Qcov1;

        % Compute OCV_pred and dOCV_dSOC for 1RC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred_1RC, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred_1RC, 'linear', 'extrap');

        % Measurement matrix H for 1RC
        H = [dOCV_dSOC, 1];

        % Compute the predicted voltage for 1RC
        V_pred_total = OCV_pred + V1_pred + R0 * noisy_I(k);

        % Compute the Kalman gain for 1RC
        S_k = H * P_pred_1RC * H' + Rcov1;
        KG = (P_pred_1RC * H') / S_k;

        % Store the Kalman gain for 1RC
        KG_1RC_all(k, :) = KG';

        %% Update step for 1RC

        z = noisy_V(k);
        residual = z - V_pred_total;

        % Store the residual for 1RC
        residual_1RC_all(k) = residual;

        % Update the estimate for 1RC
        x_estimate = x_pred + KG * residual;

        SOC_estimate_1RC = x_estimate(1);
        V1_estimate_1RC = x_estimate(2);

        % Store the updated state vector for 1RC
        x_estimate_1RC_all(k, :) = x_estimate';

        % Update the error covariance for 1RC
        P_estimate_1RC = (eye(2) - KG * H) * P_pred_1RC;

        SOC_est_1RC(k) = x_estimate(1);
        V1_est_1RC(k) = x_estimate(2);


        %% 2RC

        % Interpolate R0, R1, C1, R2, C2 based on SOC for 2RC
        R0 = interp1(SOC_params, R0_params, SOC_estimate_2RC, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate_2RC, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate_2RC, 'linear', 'extrap');
        R2 = interp1(SOC_params, R2_params, SOC_estimate_2RC, 'linear', 'extrap');
        C2 = interp1(SOC_params, C2_params, SOC_estimate_2RC, 'linear', 'extrap');

        %% Predict Step for 2RC
        if k == 1
            if s == 1
                V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
                V2_pred = noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
            else
                V1_pred = V1_estimate_2RC;
                V2_pred = V2_estimate_2RC;
            end
        else
            V1_pred = V1_estimate_2RC * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            V2_pred = V2_estimate_2RC * exp(-dt(k) / (R2 * C2)) + noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
        end

        % Predict SOC for 2RC
        SOC_pred_2RC = SOC_estimate_2RC + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % Form the predicted state vector for 2RC
        x_pred = [SOC_pred_2RC; V1_pred; V2_pred];

        % Store the predicted state vector for 2RC
        x_pred_2RC_all(k, :) = x_pred';

        % Predict the error covariance for 2RC
        A = [1 0 0;
             0 exp(-dt(k) / (R1 * C1)) 0;
             0 0 exp(-dt(k) / (R2 * C2))];
        P_pred_2RC = A * P_estimate_2RC * A' + Qcov2;

        % Compute OCV_pred and dOCV_dSOC for 2RC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred_2RC, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred_2RC, 'linear', 'extrap');

        % Measurement matrix H for 2RC
        H = [dOCV_dSOC, 1, 1];

        % Compute the predicted voltage for 2RC
        V_pred_total = OCV_pred + V1_pred + V2_pred + R0 * noisy_I(k);

        % Compute the Kalman gain for 2RC
        S_k = H * P_pred_2RC * H' + Rcov2;
        KG = (P_pred_2RC * H') / S_k;

        % Store the Kalman gain for 2RC
        KG_2RC_all(k, :) = KG';

        %% Update step for 2RC

        z = noisy_V(k);
        residual = z - V_pred_total;

        % Store the residual for 2RC
        residual_2RC_all(k) = residual;

        % Update the estimate for 2RC
        x_estimate = x_pred + KG * residual;

        SOC_estimate_2RC = x_estimate(1);
        V1_estimate_2RC = x_estimate(2);
        V2_estimate_2RC = x_estimate(3);

        % Store the updated state vector for 2RC
        x_estimate_2RC_all(k, :) = x_estimate';

        % Update the error covariance for 2RC
        P_estimate_2RC = (eye(3) - KG * H) * P_pred_2RC;

        SOC_est_2RC(k) = x_estimate(1);
        V1_est_2RC(k) = x_estimate(2);
        V2_est_2RC(k) = x_estimate(3);

    end

    % Update initial values for the next trip for 1RC
    initial_SOC_true = True_SOC(end);
    initial_SOC_cc = CC_SOC(end);
    initial_SOC_estimate_1RC = SOC_estimate_1RC;
    initial_P_estimate_1RC = P_estimate_1RC;
    initial_V1_estimate_1RC = V1_estimate_1RC;

    % Update initial values for the next trip for 2RC
    initial_SOC_estimate_2RC = SOC_estimate_2RC;
    initial_V1_estimate_2RC = V1_estimate_2RC;
    initial_V2_estimate_2RC = V2_estimate_2RC;
    initial_P_estimate_2RC = P_estimate_2RC;

    % Update previous trip's end time
    previous_trip_end_time = t(end);

    % Concatenate data for 1RC
    t_all = [t_all; t];
    True_SOC_all = [True_SOC_all; True_SOC];
    CC_SOC_all = [CC_SOC_all; CC_SOC];
    x_pred_1RC_all_trips = [x_pred_1RC_all_trips; x_pred_1RC_all];
    x_estimate_1RC_all_trips = [x_estimate_1RC_all_trips; x_estimate_1RC_all];
    KG_1RC_all_trips = [KG_1RC_all_trips; KG_1RC_all];
    residual_1RC_all_trips = [residual_1RC_all_trips; residual_1RC_all];
    I_all = [I_all; I];
    noisy_I_all = [noisy_I_all; noisy_I];

    % Concatenate data for 2RC
    x_pred_2RC_all_trips = [x_pred_2RC_all_trips; x_pred_2RC_all];
    x_estimate_2RC_all_trips = [x_estimate_2RC_all_trips; x_estimate_2RC_all];
    KG_2RC_all_trips = [KG_2RC_all_trips; KG_2RC_all];
    residual_2RC_all_trips = [residual_2RC_all_trips; residual_2RC_all];

end




figure(1);

% 1. Predicted and Estimated SOC and V1, True SOC, and CC SOC for 1RC model
subplot(3,2,1);
yyaxis left
hold on;

% Plot True SOC
plot(t_all, True_SOC_all, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
% Plot CC SOC
plot(t_all, CC_SOC_all, 'b-', 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
% Plot Estimated SOC for 1RC
plot(t_all, x_estimate_1RC_all_trips(:,1), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (1RC)');
ylabel('SOC');
% yyaxis right
% % Plot Estimated V1 for 1RC
% plot(t_all, x_estimate_1RC_all_trips(:,2), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Estimated V1 (1RC)');
%ylabel('Voltage [V]');
xlabel('Time [s]');
title('Predicted and Estimated SOC and V1 over Time (1RC)');
legend('show', 'Location', 'best');
hold off;

% 2. Predicted and Estimated SOC, V1, and V2, True SOC, and CC SOC for 2RC model
subplot(3,2,2);
yyaxis left
hold on;

% Plot True SOC
plot(t_all, True_SOC_all, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
% Plot CC SOC
plot(t_all, CC_SOC_all, 'b-', 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
% Plot Estimated SOC for 2RC
plot(t_all, x_estimate_2RC_all_trips(:,1), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (2RC)');
ylabel('SOC');
% yyaxis right
% % Plot Estimated V1 and V2 for 2RC
% plot(t_all, x_estimate_2RC_all_trips(:,2), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Estimated V1 (2RC)');
% plot(t_all, x_estimate_2RC_all_trips(:,3), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Estimated V2 (2RC)');
%ylabel('Voltage [V]');
xlabel('Time [s]');
title('Predicted and Estimated SOC, V1, and V2 over Time (2RC)');
legend('show', 'Location', 'best');
hold off;

% 3. Kalman Gains over Time for 1RC
subplot(3,2,3);
yyaxis left;
plot(t_all, KG_1RC_all_trips(:,1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Kalman Gain SOC (1RC)');
ylabel('Kalman Gain for SOC');
yyaxis right;
plot(t_all, KG_1RC_all_trips(:,2), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Kalman Gain V1 (1RC)');
ylabel('Kalman Gain for V1');
xlabel('Time [s]');
title('Kalman Gains over Time (1RC)');
legend('show', 'Location', 'best');

% 4. Kalman Gains over Time for 2RC
subplot(3,2,4);
yyaxis left;
plot(t_all, KG_2RC_all_trips(:,1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Kalman Gain SOC (2RC)');
ylabel('Kalman Gain for SOC');
yyaxis right;
plot(t_all, KG_2RC_all_trips(:,2), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Kalman Gain V1 (2RC)');
plot(t_all, KG_2RC_all_trips(:,3), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Kalman Gain V2 (2RC)');
ylabel('Kalman Gain for V1 and V2');
xlabel('Time [s]');
title('Kalman Gains over Time (2RC)');
legend('show', 'Location', 'best');

% 5. Residual over Time for 1RC
subplot(3,2,5);
plot(t_all, residual_1RC_all_trips, 'k-', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Residual');
title('Residual over Time (1RC)');

% 6. Residual over Time for 2RC
subplot(3,2,6);
plot(t_all, residual_2RC_all_trips, 'k-', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Residual');
title('Residual over Time (2RC)');


% 2. SOC 비교 플롯
figure(2)
plot(t_all, True_SOC_all, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
hold on
plot(t_all, CC_SOC_all, 'b-', 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
plot(t_all, x_estimate_1RC_all_trips(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (1RC)');
plot(t_all, x_estimate_2RC_all_trips(:,1), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (2RC)');
xlabel('Time [s]')
ylabel('SOC')
legend('True SOC', 'CC SOC', 'Estimated SOC (1RC)', 'Estimated SOC (2RC)')
title('SOC vs Time (s)')

% 3. 전류 노이즈 플롯
figure(3)
plot(t_all, I_all - noisy_I_all)
xlabel('Time [s]');
ylabel('Noise');
title('I - noisy\_I');

% 4. dOCV/dSOC 플롯
figure(4);
plot(unique_soc, dOCV_dSOC_values_smooth, 'LineWidth', 1.5);
xlabel('SOC');
ylabel('dOCV/dSOC');
title('dOCV/dSOC');
grid on;

% 5. SOC 오차 플롯
figure(5)
plot(t_all, x_estimate_1RC_all_trips(:,1) - True_SOC_all, 'b-', 'LineWidth', 1.5, 'DisplayName', '1RC-KF SOC error')
hold on
plot(t_all, x_estimate_2RC_all_trips(:,1) - True_SOC_all, 'g-', 'LineWidth', 1.5, 'DisplayName', '2RC-KF SOC error')
plot(t_all, CC_SOC_all - True_SOC_all, 'k-', 'LineWidth', 1.5, 'DisplayName', 'CC SOC error')
xlabel('Time [s]');
ylabel('SOC Error');
legend('1RC-KF SOC error', '2RC-KF SOC error', 'CC SOC error')
title('SOC Error Comparison');

%% Function for Adding Markov Noise
function [noisy_I] = Markov(I, epsilon_percent_span)

    % Define noise parameters
    sigma_percent = 0.001;      % Standard deviation in percentage (adjust as needed)

    N = 51; % Number of states
    epsilon_vector = linspace(-epsilon_percent_span/2, epsilon_percent_span/2, N); % From -noise_percent/2 to +noise_percent/2
    sigma = sigma_percent; % Standard deviation

    % Initialize transition probability matrix P
    P = zeros(N);
    for i = 1:N
        probabilities = normpdf(epsilon_vector, epsilon_vector(i), sigma);
        P(i, :) = probabilities / sum(probabilities); % Normalize to sum to 1
    end

    % Initialize state tracking
    initial_state = 48; 
    current_state = initial_state;

    % Initialize output variables
    noisy_I = zeros(size(I));
    states = zeros(size(I)); % Vector to store states
    epsilon = zeros(size(I));

    % Generate noisy current and track states
    for k = 1:length(I)
        epsilon(k) = epsilon_vector(current_state);
        noisy_I(k) = I(k) + abs(I(k)) * epsilon(k); % Apply the epsilon percentage

        states(k) = current_state; % Store the current state

        % Transition to the next state based on probabilities
        current_state = randsample(1:N, 1, true, P(current_state, :));
    end

end
