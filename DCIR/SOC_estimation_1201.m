clc;clear;close all;

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
soc_values = soc_ocv(:, 1);     % SOC values % 1083 x 1
ocv_values = soc_ocv(:, 2);     % Corresponding OCV values [V] % 1083 x 1

load('udds_data.mat'); % Struct array 'udds_data' containing fields V, I, t, Time_duration, SOC

Q_batt = 2.7742 ; % 2.7742; % [Ah]
SOC_begin_true = 0.9907; %0.983;
SOC_begin_cc = 0.9907;
epsilon_percent_span = 0.1;
voltage_noise_percent = 0.01;

[unique_ocv, b] = unique(ocv_values); % unique_ocv : 1029x1
unique_soc = soc_values(b);           % unique_soc : 1029x1  

% Compute the derivative of OCV with respect to SOC (H 행렬 위해)
dOCV_dSOC_values = gradient(unique_ocv) ./ gradient(unique_soc);

windowSize = 10; 
dOCV_dSOC_values_smooth = movmean(dOCV_dSOC_values, windowSize);

%% 2. Kalman filter settings
% 1 : 1RC 
% 2 : 2RC
% 3 : DRT 

num_RC = length(tau_discrete);

P1_init = [1e-4 0;
            0   9.58e-13]; % [SOC ; V1] % State covariance
P2_init = [1e-4 0        0;
            0   1e-6 0;
            0   0       1e-6]; % [SOC; V1; V2] % State covariance

P3_init = zeros(1 + num_RC); % Initialize P3_init
P3_init(1,1) = 1e-13;    % Initial covariance for SOC
for i = 2:(1 + num_RC)
    P3_init(i,i) = 1e-13; % Initial covariance for each V_i
end

% Q

Q1 = [1e-4 0;
      0  9.58e-13];  % [SOC ; V1] % Process covariance

Q2 = [1e-4 0        0;
             0     1e-6    0;
             0      0     1e-6]; % [SOC; V1; V2] % Process covariance

Q3 = zeros(1 + num_RC); % Initialize Q3
Q3(1,1) = 1e-13; %5e-10; % Process noise for SOC
for i = 2:(1 + num_RC)
    Q3(i,i) = 1e-13 ;% 1e-9; % Process noise for each V_i
end

% R , Measurement covariance

R1 = 25e-4;
R2 = 25e-4;
R3 = 25e-4;

%% 3. Extract ECM parameters

% FOR --> vercat으로 수정 (11/30)
SOC_params = vertcat(optimized_params_struct_final_2RC.SOC);
R0_params = vertcat(optimized_params_struct_final_2RC.R0);
R1_params = vertcat(optimized_params_struct_final_2RC.R1);
R2_params = vertcat(optimized_params_struct_final_2RC.R2);
C1_params = vertcat(optimized_params_struct_final_2RC.C1);
C2_params = vertcat(optimized_params_struct_final_2RC.C2);

%% 4. Kalman filter

num_trips = length(udds_data);

True_soc_all = cell(num_trips,1);
CC_SOC_all = cell(num_trips, 1);
SOC_est_1RC_all = cell(num_trips, 1);
SOC_est_2RC_all = cell(num_trips, 1);
SOC_est_DRT_all = cell(num_trips, 1);

time_offset = 0;

for s = 1: num_trips-16 % trips 수에 대하여 
    fprintf('Processing Trip %d/%d...\n', s, num_trips-16);

    I = udds_data(s).I;
    V = udds_data(s).V; 
    t = udds_data(s).t + time_offset;
    dt = [t(1); diff(t)];
    dt(1) = dt(2);

    [noisy_I] = Markov(I,epsilon_percent_span); % 전류 : markov noise
    noisy_V = V + voltage_noise_percent * V .* randn(size(V)); % 전압 : gaussian noise

    True_SOC = SOC_begin_true + cumtrapz(t - time_offset,I)/(3600 * Q_batt);
    CC_SOC = SOC_begin_cc + cumtrapz(t - time_offset,noisy_I)/(3600 * Q_batt);

    % 1-RC 

    SOC_est_1RC = zeros(length(t), 1);
    V1_est_1RC = zeros(length(t), 1);   

    SOC_estimate_1RC = CC_SOC(1); % 초기 SOC 지정 
    V1_estimate_1RC = 0;      % k = 1 일때, k-1번째 V1 = 0 
    P_estimate_1RC = P1_init;     % 초기 공분산 지정 
    
    % kalman gain effect 알아보기 (11/30 upadate)
    x_pred_1RC_all = zeros(length(t), 2);     % 예측 값 (칼만 게인 전)
    K_1RC_all = zeros(length(t), 2);          % 칼만 게인 값
    residual_1RC_all = zeros(length(t), 1);   % 잔차 값 % 칼만 게인 x 잔차 값 더해줌
    x_estimate_1RC_all = zeros(length(t), 2); % 보정 값 (칼만 게인 후) 
    
    for k = 1:length(t) % 각 trip의 시간에 대해여

        % R0,R1,C ( SOC , 0.5C)
        R0 = interp1(SOC_params, R0_params, SOC_estimate_1RC, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate_1RC, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate_1RC, 'linear', 'extrap');

        %% Predict step
        if k == 1 % 첫번째 시간
            if s == 1
                V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            else
                V1_pred = V1_estimate_1RC;
            end
        else 
            V1_pred = V1_estimate_1RC * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));  % Predict RC 전압
        end

        % Predict soc 
        SOC_pred_1RC = SOC_estimate_1RC + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % Form the predicted state vector
        x_pred = [SOC_pred_1RC; V1_pred];

        % Store the predicted state vector
        x_pred_1RC_all(k, :) = x_pred';

        % Predict the error covariance
        A = [1 0;
             0 exp(-dt(k) / (R1 * C1))];
        P_pred_1RC = A * P_estimate_1RC * A' + Q1;

        % Compute OCV_pred and dOCV_dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred_1RC, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred_1RC, 'linear', 'extrap');

        % Measurement matrix H
        H = [dOCV_dSOC, 1];

        % Compute the predicted voltage
        V_pred_total = OCV_pred + V1_pred + R0 * noisy_I(k);

        % Compute the Kalman gain
        S_k = H * P_pred_1RC * H' + R1; % Measurement noise covariance
        K = (P_pred_1RC * H') / S_k;

        % Store the Kalman gain
        K_1RC_all(k, :) = K';

        %% Update step

        z = noisy_V(k); % Measurement
        residual = z - V_pred_total;

        % Store the residual
        residual_1RC_all(k) = residual;

        % Update the estimate
        x_estimate = x_pred + K * residual;

        SOC_estimate_1RC = x_estimate(1);
        V1_estimate_1RC = x_estimate(2);

        % Store the updated state vector
        x_estimate_1RC_all(k, :) = x_estimate';

        % Update the error covariance
        P_estimate_1RC = (eye(2) - K * H) * P_pred_1RC;

        SOC_est_1RC(k) = x_estimate(1);
        V1_est_1RC(k) = x_estimate(2);


    end


    % 2-RC


    time_offset = t(end);


end

figure;

% 1. Predicted and Estimated SOC and V1, True SOC, and CC SOC
subplot(3,1,1);
yyaxis left
hold on;

% Plot True SOC
plot(t, True_SOC, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
% Plot CC SOC
plot(t, CC_SOC, 'b-', 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
% Plot Predicted SOC 
plot(t, x_pred_1RC_all(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Predicted SOC');
% Plot Estimated SOC 
plot(t, x_estimate_1RC_all(:,1), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC');
ylabel('SOC');
yyaxis right
% Plot Predicted V1 (solid line)
plot(t, x_pred_1RC_all(:,2), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Predicted V1');
% Plot Estimated V1 (dashed line)
plot(t, x_estimate_1RC_all(:,2), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Estimated V1');
ylabel('V1 [V]');
xlabel('Time [s]');
title('Predicted and Estimated SOC and V1 over Time');
legend('show', 'Location', 'best');
hold off;

% 2. Kalman Gains over Time
subplot(3,1,2);
yyaxis left;
plot(t, K_1RC_all(:,1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Kalman Gain SOC');
ylabel('Kalman Gain for SOC');
yyaxis right;
plot(t, K_1RC_all(:,2), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Kalman Gain V1');
ylabel('Kalman Gain for V1');
xlabel('Time [s]');
title('Kalman Gains over Time');
legend('show', 'Location', 'best');

% 3. Residual over Time
subplot(3,1,3);
plot(t, residual_1RC_all, 'k-', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Residual');
title('Residual over Time');











%% Function for adding Markov noise
function [noisy_I] = Markov(I, epsilon_percent_span)

    % Define noise parameters
    sigma_percent = 0.001;      % Standard deviation in percentage (adjust as needed)

    N = 51; % Number of states
    epsilon_vector = linspace(-epsilon_percent_span/2, epsilon_percent_span/2, N); % From -noise_percent to +noise_percent
    sigma = sigma_percent; % Standard deviation

    % Initialize transition probability matrix P
    P = zeros(N);
    for i = 1:N
        probabilities = normpdf(epsilon_vector, epsilon_vector(i), sigma);
        P(i, :) = probabilities / sum(probabilities); % Normalize to sum to 1
    end

    % Initialize state tracking
    initial_state = 3; 
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
































