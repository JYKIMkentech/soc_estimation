clc; clear; close all;

%% 시드 설정
rng(13);

%% Font size settings
axisFontSize = 14;
titleFontSize = 16;
legendFontSize = 12;
labelFontSize = 14;

%% 1. Data load

% ECM 파라미터 (HPPC 테스트로부터)
load('optimized_params_struct_final_2RC.mat'); % 필드: R0, R1, C1, R2, C2, SOC, avgI, m, Crate

% DRT 파라미터 (gamma 및 tau 값)
load('theta_discrete.mat');
load('gamma_est_all.mat', 'gamma_est_all');  % 수정된 부분: SOC_mid_all 제거
load('R0_est_all.mat')

tau_discrete = exp(theta_discrete); % tau 값

% SOC-OCV 룩업 테이블 (C/20 테스트로부터)
load('soc_ocv.mat', 'soc_ocv'); % [SOC, OCV]
soc_values = soc_ocv(:, 1);     % SOC 값 % 1083 x 1
ocv_values = soc_ocv(:, 2);     % 해당하는 OCV 값 [V] % 1083 x 1

% 주행 데이터 (17개의 트립)
load('udds_data.mat'); % 구조체 배열 'udds_data'로 V, I, t, Time_duration, SOC 필드 포함

Q_batt = 2.7742; % [Ah]
SOC_begin_true = 0.9907;
SOC_begin_cc = 0.9907;
epsilon_percent_span = 0.4;
voltage_noise_percent = 0.01;

[unique_ocv, b] = unique(ocv_values); % unique_soc : 1029x1
unique_soc = soc_values(b);           % unique_ocv : 1029x1  

%% Compute the derivative of OCV with respect to SOC
dOCV_dSOC_values = gradient(unique_ocv) ./ gradient(unique_soc);

windowSize = 10; 
dOCV_dSOC_values_smooth = movmean(dOCV_dSOC_values, windowSize);

%% 2. Kalman filter setting

% 1 : 1-RC , 2: 2-RC , 3 : DRT

num_RC = length(tau_discrete);

% P
P1_init = [1e-5 0;
            0   1e-3]; % [SOC ; V1] % State covariance
P2_init = [1e-10 0        0;
            0   1e-6    0;
            0   0       1e-6]; % [SOC; V1; V2] % State covariance

P3_init(1,1) = 1e-10;    % SOC의 초기 공분산
for i = 2:(1 + num_RC)
    P3_init(i,i) = 1e-10; % 각 V_i의 초기 공분산
end


% Q

Q1 = [1e-5 0;
      0  1e-5];  % [SOC ; V1] % Process covariance

Q2 = [1e-8 0        0;
             0     1e-5    0;
             0      0     1e-5]; % [SOC; V1; V2] % Process covariance

Q3(1,1) = 1e-9; % SOC의 프로세스 노이즈
for i = 2:(1 + num_RC)
    Q3(i,i) = 1e-9; % 각 V_i의 프로세스 노이즈
end

% R , Measurement covariance

R1 = 25e-3;
R2 = 25e-3;
R3 = 25e-3;

%% 3. ECM parameter 추출

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

%% 4. 모든 trips에 칼만 필터 적용

num_trips = length(udds_data);

True_SOC_all = cell(num_trips, 1);   
CC_SOC_all = cell(num_trips, 1);
SOC_est_1RC_all = cell(num_trips, 1);
SOC_est_2RC_all = cell(num_trips, 1);
SOC_est_DRT_all = cell(num_trips, 1);

for s = 1 : num_trips-16 % 각 Trip에 대해
    fprintf('Processing Trip %d/%d...\n', s, num_trips-16);

    I = udds_data(s).I;
    V = udds_data(s).V;
    t = udds_data(s).t; % 모든 trip의 시작이 0초
    dt = [t(1); diff(t)];
    dt(1) = dt(2);
    Time_duration = udds_data(s).Time_duration; % 모든 trip이 시작이 이어져있음 

    [noisy_I] = Markov(I,epsilon_percent_span); % 전류에 Markov noise 추가 
    noisy_V = V + voltage_noise_percent * V .* randn(size(V)); % 전압에 Gaussian noise 추가 

    True_SOC = SOC_begin_true + cumtrapz(t,I)/(3600 * Q_batt); % True SOC (noisy 존재 x)
    CC_SOC = SOC_begin_cc + cumtrapz(t,noisy_I)/(3600 * Q_batt); % CC SOC (noisy 존재)

    True_SOC_all{s} = True_SOC;
    CC_SOC_all{s} = CC_SOC;

    SOC_begin_true = True_SOC(end);
    SOC_begin_cc = CC_SOC(end);

    %% DRT

    gamma = gamma_est_all(s,:); % 1x201
    delta_theta = theta_discrete(2) - theta_discrete(1); % 0.0476
    R_i = gamma * delta_theta; % 1x201
    C_i = tau_discrete' ./ R_i; % 1x201

    SOC_estimate = CC_SOC(1);
    V_estimate = zeros(num_RC,1); % V_i들의 초기값
    P_estimate = P3_init;
    SOC_est_DRT = zeros(length(t),1);
    V_DRT_est = zeros(length(t), num_RC); % 각 V_i 저장

    for k = 1:length(t) % k-1 --> k번째 시간 Prediction and correction

        R0 = interp1(SOC_params, R0_params, SOC_estimate, 'linear', 'extrap');

        % predict step

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
                V_pred(i) = V_estimate(i) * exp(-dt(k) / (R_i(i) * C_i(i))) + noisy_I(k) * R_i(i) * (1 - exp(-dt(k) / (R_i(i) * C_i(i))));
            end
        end

        SOC_pred = SOC_estimate + (dt(k) / (Q_batt * 3600)) * noisy_I(k);
        
        x_pred = [SOC_pred; V_pred];


        % Predict the error covariance
        A = zeros(1 + num_RC);
        A(1,1) = 1; % SOC
        for i = 1:num_RC
            A(i+1,i+1) = exp(-dt(k) / (R_i(i) * C_i(i)));
        end
        P_pred = A * P_estimate * A' + Q3;

        % Compute OCV_pred and dOCV_dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');

        % dOCV_dSOC 계산 (미리 계산된 값 사용)
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred, 'linear', 'extrap');

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
        P_estimate = (eye(1 + num_RC) - K * H) * P_pred;

        % Store the estimates
        SOC_est_DRT(k) = x_estimate(1);
        V_estimate = x_estimate(2:end);

        V_DRT_est(k, :) = V_estimate'; % V1,V2,V3,...V201까지 저장

        % Update the estimates for next iteration
        SOC_estimate = x_estimate(1);
    end

        SOC_est_DRT_all{s} = SOC_est_DRT; 

    


    %% 1-RC

    SOC_est_1RC =  zeros(length(t), 1);
    V1_est_1RC = zeros(length(t), 1);

    SOC_estimate = CC_SOC(1);
    P_estimate = P1_init;

    for k = 1:length(t)

        % Compute R0, R1, C1 at SOC_estimate
        R0 = interp1(SOC_params, R0_params, SOC_estimate, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate, 'linear', 'extrap');

        % Predict step

        if k == 1
            % Initial prediction of V1
            V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
        else
            % Predict V1
            V1_pred = V1_estimate * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
        end

        % Predict SOC
        SOC_pred = SOC_estimate + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % Form the predicted state vector
        x_pred = [SOC_pred; V1_pred];

        % Predict the error covariance
        A = [1 0;
             0 exp(-dt(k) / (R1 * C1))];
        P_pred = A * P_estimate * A' + Q1;

        % Compute OCV_pred and dOCV_dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred, 'linear', 'extrap');

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
        P_estimate = (eye(2) - K * H) * P_pred;

        % Store the estimates
        SOC_est_1RC(k) = x_estimate(1);
        V1_est_1RC(k) = x_estimate(2);

        % Update the estimates for next iteration
        SOC_estimate = x_estimate(1);
        V1_estimate = x_estimate(2);
    end

    SOC_est_1RC_all{s} = SOC_est_1RC;

    %% 2-RC

    SOC_est_2RC = zeros(length(t),1);
    V1_est_2RC = zeros(length(t),1);
    V2_est_2RC = zeros(length(t),1);

    SOC_estimate = CC_SOC(1);
    P_estimate = P2_init;

    for k = 1:length(t)

        % Compute R0, R1, C1, R2, C2 at SOC_estimate
        R0 = interp1(SOC_params, R0_params, SOC_estimate, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate, 'linear', 'extrap');
        R2 = interp1(SOC_params, R2_params, SOC_estimate, 'linear', 'extrap');
        C2 = interp1(SOC_params, C2_params, SOC_estimate, 'linear', 'extrap');

        % Predict step

        if k == 1
            % Initial prediction of V1 and V2
            V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            V2_pred = noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
        else
            % Predict V1 and V2
            V1_pred = V1_estimate * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            V2_pred = V2_estimate * exp(-dt(k) / (R2 * C2)) + noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
        end

        % Predict SOC
        SOC_pred = SOC_estimate + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % Form the predicted state vector
        x_pred = [SOC_pred; V1_pred; V2_pred];

        % Predict the error covariance
        A = [1 0 0;
             0 exp(-dt(k) / (R1 * C1)) 0;
             0 0 exp(-dt(k) / (R2 * C2))];
        P_pred = A * P_estimate * A' + Q2;

        % Compute OCV_pred and dOCV_dSOC
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc,dOCV_dSOC_values_smooth, SOC_pred, 'linear', 'extrap');

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
        P_estimate = (eye(3) - K * H) * P_pred;

        % Store the estimates
        SOC_est_2RC(k) = x_estimate(1);
        V1_est_2RC(k) = x_estimate(2);
        V2_est_2RC(k) = x_estimate(3);

        % Update the estimates for next iteration
        SOC_estimate = x_estimate(1);
        V1_estimate = x_estimate(2);
        V2_estimate = x_estimate(3);
    end

    SOC_est_2RC_all{s} = SOC_est_2RC;  

end



%% Plot example for the first trip
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

figure;
plot(t,I-noisy_I)




%% Function for adding Markov noise
function [noisy_I] = Markov(I, epsilon_percent_span)

    % Define noise parameters
    sigma_percent = 0.001;      % Standard deviation in percentage (adjust as needed)

    N = 51; % Number of states
    epsilon_vector = linspace(-epsilon_percent_span/2, epsilon_percent_span/2, N); % From -noise_percent to +noise_percent
    sigma = sigma_percent; % Standard deviation in percentage

    % Initialize transition probability matrix P
    P = zeros(N);
    for i = 1:N
        probabilities = normpdf(epsilon_vector, epsilon_vector(i), sigma);
        P(i, :) = probabilities / sum(probabilities); % Normalize to sum to 1
    end

    % Initialize state tracking
    initial_state = 3; %randsample(1:N, 1); % Randomly select initial state
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
























