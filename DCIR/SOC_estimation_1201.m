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

%num_RC = length(tau_discrete);

Pcov1_init = [1e-6 0;
            0   9.58e-13]; % [SOC ; V1] % State covariance



% Pcov2_init = [1e-4 0        0;
%           0   1e-6 0;
%           0   0       1e-6]; % [SOC; V1; V2] % State covariance

%Pcov3_init = zeros(1 + num_RC); % Initialize P3_init
%Pcov3_init(1,1) = 1e-13;    % Initial covariance for SOC
%for i = 2:(1 + num_RC)
%    Pcov3_init(i,i) = 1e-13; % Initial covariance for each V_i
%end

% Q

Qcov1 = [1e-6 0;
      0  9.58e-13];  % [SOC ; V1] % Process covariance



%Qcov2 = [1e-4 0        0;
%             0     1e-6    0;
%             0      0     1e-6]; % [SOC; V1; V2] % Process covariance

%Qcov3 = zeros(1 + num_RC); % Initialize Q3
%Qcov3(1,1) = 1e-13; %5e-10; % Process noise for SOC
%for i = 2:(1 + num_RC)
%    Qcov3(i,i) = 1e-13 ;% 1e-9; % Process noise for each V_i
%end

% R , Measurement covariance

Rcov1 = 25e-7;

%Rcov2 = 25e-4;
%Rcov3 = 25e-4;

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

True_soc_alltrip = cell(num_trips,1);
CC_SOC_alltrip = cell(num_trips, 1);
SOC_est_1RC_alltrip = cell(num_trips, 1);
SOC_est_2RC_alltrip = cell(num_trips, 1);
SOC_est_DRT_alltrip = cell(num_trips, 1);

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
    P_estimate_1RC = Pcov1_init;     % 초기 공분산 지정 
    
    % kalman gain effect 알아보기 (11/30 upadate)
    x_pred_1RC_all = zeros(length(t), 2);     % 예측 값 (칼만 게인 전)
    KG_1RC_all = zeros(length(t), 2);          % 칼만 게인 값
    residual_1RC_all = zeros(length(t), 1);   % 잔차 값 % 칼만 게인 x 잔차 값 더해줌
    x_estimate_1RC_all = zeros(length(t), 2); % 보정 값 (칼만 게인 후) 
    
    for k = 1:length(t) % 각 trip의 시간에 대해여

        % R0,R1,C ( SOC , 0.5C)
        R0 = interp1(SOC_params, R0_params, SOC_estimate_1RC, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate_1RC, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate_1RC, 'linear', 'extrap');

        %% Predict step
        if k == 1 % 첫번째 시간
            if s == 1 % 첫번째 trip에 대해서만 , S 번 째 트립부터는 S-1 번째 Trip에서의 V1값 가져오기
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
        dt(k)
        R1 * C1
        A = [1 0;
             0 exp(-dt(k) / (R1 * C1))];
        
        L = A * P_estimate_1RC * A

        P_pred_1RC = A * P_estimate_1RC * A' + Qcov1;
        
        % Compute OCV_pred and dOCV_dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred_1RC, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred_1RC, 'linear', 'extrap');

        % Measurement matrix H
        H = [dOCV_dSOC, 1];
       

        % Compute the predicted voltage
        R0 * noisy_I(k)
        V_pred_total = OCV_pred + V1_pred + R0 * noisy_I(k);

        % Compute the Kalman gain
        H * P_pred_1RC * H'
        S_k = H * P_pred_1RC * H' + Rcov1; % Measurement noise covariance
        P_pred_1RC * H'
        KG = (P_pred_1RC * H') / S_k;
        

        % Store the Kalman gain
        KG_1RC_all(k, :) = KG';

        %% Update step

        z = noisy_V(k); % Measurement
        residual = z - V_pred_total;

        % Store the residual
        residual_1RC_all(k) = residual;

        % Update the estimate
        KG * residual
        x_estimate = x_pred + KG * residual;

        SOC_estimate_1RC = x_estimate(1);
        V1_estimate_1RC = x_estimate(2);

        % Store the updated state vector
        x_estimate_1RC_all(k, :) = x_estimate';

        % Update the error covariance
        P_estimate_1RC = (eye(2) - KG * H) * P_pred_1RC;

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
plot(t, KG_1RC_all(:,1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Kalman Gain SOC');
ylabel('Kalman Gain for SOC');
yyaxis right;
plot(t, KG_1RC_all(:,2), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Kalman Gain V1');
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

figure(2)

plot(t, True_SOC, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
hold on
plot(t, CC_SOC, 'b-', 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
plot(t, x_estimate_1RC_all(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC');
xlabel('time')
ylabel('SOC')
legend('True soc', 'CC SOC', 'Estimated SOC')
title('SOC vs time(sec)')

figure(3)

plot(t,I-noisy_I)
xlabel('Time');
ylabel('noise');
title('I-noisy_I');

figure(4);
plot(unique_soc, dOCV_dSOC_values_smooth, 'LineWidth', 1.5);
xlabel('SOC');
ylabel('dOCV/dSOC');
title('docv/dsoc');
grid on;

figure(5)

plot(t,x_estimate_1RC_all(:,1) - True_SOC, 'b-', 'LineWidth', 1.5, 'DisplayName', 'SOC error')
hold on
plot(t,CC_SOC - True_SOC, 'k-', 'LineWidth', 1.5, 'DisplayName', 'SOC error')
xlabel('time');
ylabel('SOC error');
legend('1RC-KF SOC error', 'CC SOC error')
title('1RC-KF SOC error');


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



























