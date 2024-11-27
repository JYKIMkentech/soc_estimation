clc; clear; close all;

%% 시드 설정
% rng(13); % 랜덤 시드 고정 (필요시 주석 해제)

%% 폰트 크기 설정
axisFontSize = 14;
titleFontSize = 16;
legendFontSize = 12;
labelFontSize = 14;

%% 1. 데이터 로드

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
current_noise_percent = 0.02;
voltage_noise_percent = 0.01;

[unique_ocv, b] = unique(ocv_values); % unique_soc : 1029x1
unique_soc = soc_values(b);           % unique_ocv : 1029x1  

%% OCV에 대한 SOC의 도함수 계산
dOCV_dSOC_values = gradient(unique_ocv) ./ gradient(unique_soc);

windowSize = 10; 
dOCV_dSOC_values_smooth = movmean(dOCV_dSOC_values, windowSize);

%% 2. 칼만 필터 설정

% 1 : 1-RC, 2: 2-RC, 3 : DRT

num_RC = length(tau_discrete);

% P 초기값 설정
P1_init = [1e-18 0;
           0   1e-17]; % [SOC ; V1] 상태 공분산

P2_init = [1e-15 0       0;
           0   1e-15    0;
           0   0       1e-15]; % [SOC; V1; V2] 상태 공분산

P3_init = zeros(1 + num_RC); % DRT 모델 상태 공분산
P3_init(1,1) = 1e-3;    % SOC의 초기 공분산
for i = 2:(1 + num_RC)
    P3_init(i,i) = 1e-1; % 각 V_i의 초기 공분산
end

% Q 프로세스 노이즈 공분산
Q1 = [1e-10 0;
      0  1e-15];  % [SOC ; V1] 프로세스 노이즈

Q2 = [1e-10 0        0;
      0     1e-15    0;
      0      0     1e-15]; % [SOC; V1; V2] 프로세스 노이즈

Q3 = zeros(1 + num_RC);
Q3(1,1) = 1e-7; % SOC의 프로세스 노이즈
for i = 2:(1 + num_RC)
    Q3(i,i) = 1e-5; % 각 V_i의 프로세스 노이즈
end

% R 측정 노이즈 공분산
R1 = 5.25e-3;
R2 = 5.25e-3;
R3 = 5.25e-2;

%% 3. ECM 파라미터 추출

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

%% 4. 모든 트립에 칼만 필터 적용

num_trips = length(udds_data);

% 결과를 저장할 셀 배열 초기화
True_SOC_all = cell(num_trips, 1);   
CC_SOC_all = cell(num_trips, 1);
SOC_est_1RC_all = cell(num_trips, 1);
SOC_est_2RC_all = cell(num_trips, 1);
SOC_est_DRT_all = cell(num_trips, 1);

% 예측 및 추정을 저장할 셀 배열 추가
x_pred_1RC_all = cell(num_trips, 1);
x_estimate_1RC_all = cell(num_trips, 1);
K_1RC_all = cell(num_trips, 1);

x_pred_2RC_all = cell(num_trips, 1);
x_estimate_2RC_all = cell(num_trips, 1);
K_2RC_all = cell(num_trips, 1);

x_pred_DRT_all = cell(num_trips, 1);
x_estimate_DRT_all = cell(num_trips, 1);
K_DRT_all = cell(num_trips, 1);

for s = 1 : num_trips-16 % 각 트립에 대해
    fprintf('Processing Trip %d/%d...\n', s, num_trips-16);

    I = udds_data(s).I;
    V = udds_data(s).V;
    t = udds_data(s).t; % 모든 트립의 시작이 0초
    dt = [t(1); diff(t)];
    dt(1) = dt(2);
    Time_duration = udds_data(s).Time_duration; % 모든 트립이 이어져있음 

    [noisy_I] = Markov(I, current_noise_percent); % 전류에 마코프 노이즈 추가 
    noisy_V = V + voltage_noise_percent * V .* randn(size(V)); % 전압에 가우시안 노이즈 추가 

    True_SOC = SOC_begin_true + cumtrapz(t,I)/(3600 * Q_batt); % True SOC (노이즈 없음)
    CC_SOC = SOC_begin_cc + cumtrapz(t,noisy_I)/(3600 * Q_batt); % CC SOC (노이즈 있음)

    True_SOC_all{s} = True_SOC;
    CC_SOC_all{s} = CC_SOC;

    SOC_begin_true = True_SOC(end);
    SOC_begin_cc = CC_SOC(end);

    %% DRT 모델

    gamma = gamma_est_all(s,:); % 1x201
    delta_theta = theta_discrete(2) - theta_discrete(1); % 0.0476
    R_i = gamma * delta_theta; % 1x201
    C_i = tau_discrete' ./ R_i; % 1x201

    SOC_estimate = CC_SOC(1);
    V_estimate = zeros(num_RC,1); % V_i들의 초기값
    P_estimate = P3_init;
    SOC_est_DRT = zeros(length(t),1);
    V_DRT_est = zeros(length(t), num_RC); % 각 V_i 저장

    % 예측 및 추정, 칼만 게인 저장할 배열 초기화
    x_pred_DRT = zeros(length(t), 1 + num_RC);      % [SOC_pred, V_pred_RC]
    x_estimate_DRT = zeros(length(t), 1 + num_RC);  % [SOC_estimate, V_estimate_RC]
    K_DRT = zeros(1 + num_RC, length(t));           % 칼만 게인

    for k = 1:length(t) % k-1 --> k번째 시간 Prediction and correction

        R0 = interp1(SOC_params, R0_params, SOC_estimate, 'linear', 'extrap');

        % 예측 단계
        if k == 1
            % 초기 V_i 예측
            V_pred = zeros(num_RC,1);
            for i = 1:num_RC
                V_pred(i) = noisy_I(k) * R_i(i) * (1 - exp(-dt(k) / (R_i(i) * C_i(i))));
            end
        else
            % V_i 예측
            V_pred = zeros(num_RC,1);
            for i = 1:num_RC
                V_pred(i) = V_estimate(i) * exp(-dt(k) / (R_i(i) * C_i(i))) + noisy_I(k) * R_i(i) * (1 - exp(-dt(k) / (R_i(i) * C_i(i))));
            end
        end

        SOC_pred = SOC_estimate + (dt(k) / (Q_batt * 3600)) * noisy_I(k);
        
        x_pred = [SOC_pred; V_pred];

        % 예측된 상태 저장
        x_pred_DRT(k, :) = x_pred';

        % 예측된 오차 공분산
        A = zeros(1 + num_RC);
        A(1,1) = 1; % SOC
        for i = 1:num_RC
            A(i+1,i+1) = exp(-dt(k) / (R_i(i) * C_i(i)));
        end
        P_pred = A * P_estimate * A' + Q3;

        % OCV 예측 및 dOCV_dSOC 계산
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred, 'linear', 'extrap');

        % 측정 행렬 H
        H = zeros(1, 1 + num_RC);
        H(1) = dOCV_dSOC;
        H(2:end) = ones(1, num_RC);

        % 전체 전압 예측
        V_pred_total = OCV_pred + sum(V_pred) + R0 * noisy_I(k);

        % 칼만 게인 계산
        S = H * P_pred * H' + R3; % 측정 노이즈 공분산
        K = P_pred * H' / S;

        % 칼만 게인 저장
        K_DRT(:, k) = K;

        % 추정 단계
        z = noisy_V(k); % 측정값
        x_estimate = x_pred + K * (z - V_pred_total);

        % 추정된 상태 저장
        x_estimate_DRT(k, :) = x_estimate';

        % 오차 공분산 업데이트
        P_estimate = (eye(1 + num_RC) - K * H) * P_pred;

        % SOC 및 V_i 저장
        SOC_est_DRT(k) = x_estimate(1);
        V_estimate = x_estimate(2:end);

        V_DRT_est(k, :) = V_estimate'; % V1,V2,V3,...V201까지 저장

        % 다음 반복을 위한 추정값 업데이트
        SOC_estimate = x_estimate(1);
    end

    SOC_est_DRT_all{s} = SOC_est_DRT; 
    x_pred_DRT_all{s} = x_pred_DRT;
    x_estimate_DRT_all{s} = x_estimate_DRT;
    K_DRT_all{s} = K_DRT;

    %% 1-RC 모델

    SOC_est_1RC =  zeros(length(t), 1);
    V1_est_1RC = zeros(length(t), 1);

    % 예측 및 추정, 칼만 게인 저장할 배열 초기화
    x_pred_1RC = zeros(length(t), 2);        % [SOC_pred, V1_pred]
    x_estimate_1RC = zeros(length(t), 2);    % [SOC_estimate, V1_estimate]
    K_1RC = zeros(2, length(t));             % 칼만 게인

    SOC_estimate = CC_SOC(1);
    P_estimate = P1_init;

    for k = 1:length(t)

        % SOC_estimate에서 R0, R1, C1 계산
        R0 = interp1(SOC_params, R0_params, SOC_estimate, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate, 'linear', 'extrap');

        % 예측 단계
        if k == 1
            % 초기 V1 예측
            V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
        else
            % V1 예측
            V1_pred = V_estimate(1) * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
        end

        % SOC 예측
        SOC_pred = SOC_estimate + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % 예측된 상태 벡터
        x_pred = [SOC_pred; V1_pred];

        % 예측된 상태 저장
        x_pred_1RC(k, :) = x_pred';

        % 오차 공분산 예측
        A = [1 0;
             0 exp(-dt(k) / (R1 * C1))];
        P_pred = A * P_estimate * A' + Q1;

        % OCV 예측 및 dOCV_dSOC 계산
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred, 'linear', 'extrap');

        % 측정 행렬 H
        H = [dOCV_dSOC, 1];

        % 전체 전압 예측
        V_pred_total = OCV_pred + V1_pred + R0 * noisy_I(k);

        % 칼만 게인 계산
        S = H * P_pred * H' + R1; % 측정 노이즈 공분산
        K = P_pred * H' / S;

        % 칼만 게인 저장
        K_1RC(:, k) = K;

        % 추정 단계
        z = noisy_V(k); % 측정값
        x_estimate = x_pred + K * (z - V_pred_total);

        % 추정된 상태 저장
        x_estimate_1RC(k, :) = x_estimate';

        % 오차 공분산 업데이트
        P_estimate = (eye(2) - K * H) * P_pred;

        % SOC 및 V1 저장
        SOC_est_1RC(k) = x_estimate(1);
        V1_est_1RC(k) = x_estimate(2);

        % 다음 반복을 위한 추정값 업데이트
        SOC_estimate = x_estimate(1);
    end

    SOC_est_1RC_all{s} = SOC_est_1RC;
    x_pred_1RC_all{s} = x_pred_1RC;
    x_estimate_1RC_all{s} = x_estimate_1RC;
    K_1RC_all{s} = K_1RC;

    %% 2-RC 모델

    SOC_est_2RC = zeros(length(t),1);
    V1_est_2RC = zeros(length(t),1);
    V2_est_2RC = zeros(length(t),1);

    % 예측 및 추정, 칼만 게인 저장할 배열 초기화
    x_pred_2RC = zeros(length(t), 3);        % [SOC_pred, V1_pred, V2_pred]
    x_estimate_2RC = zeros(length(t), 3);    % [SOC_estimate, V1_estimate, V2_estimate]
    K_2RC = zeros(3, length(t));             % 칼만 게인

    SOC_estimate = CC_SOC(1);
    P_estimate = P2_init;

    for k = 1:length(t)

        % SOC_estimate에서 R0, R1, C1, R2, C2 계산
        R0 = interp1(SOC_params, R0_params, SOC_estimate, 'linear', 'extrap');
        R1 = interp1(SOC_params, R1_params, SOC_estimate, 'linear', 'extrap');
        C1 = interp1(SOC_params, C1_params, SOC_estimate, 'linear', 'extrap');
        R2 = interp1(SOC_params, R2_params, SOC_estimate, 'linear', 'extrap');
        C2 = interp1(SOC_params, C2_params, SOC_estimate, 'linear', 'extrap');

        % 예측 단계
        if k == 1
            % 초기 V1, V2 예측
            V1_pred = noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            V2_pred = noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
        else
            % V1, V2 예측
            V1_pred = V_estimate(1) * exp(-dt(k) / (R1 * C1)) + noisy_I(k) * R1 * (1 - exp(-dt(k) / (R1 * C1)));
            V2_pred = V_estimate(2) * exp(-dt(k) / (R2 * C2)) + noisy_I(k) * R2 * (1 - exp(-dt(k) / (R2 * C2)));
        end

        % SOC 예측
        SOC_pred = SOC_estimate + (dt(k) / (Q_batt * 3600)) * noisy_I(k);

        % 예측된 상태 벡터
        x_pred = [SOC_pred; V1_pred; V2_pred];

        % 예측된 상태 저장
        x_pred_2RC(k, :) = x_pred';

        % 오차 공분산 예측
        A = [1 0 0;
             0 exp(-dt(k) / (R1 * C1)) 0;
             0 0 exp(-dt(k) / (R2 * C2))];
        P_pred = A * P_estimate * A' + Q2;

        % OCV 예측 및 dOCV_dSOC 계산
        OCV_pred = interp1(unique_soc, unique_ocv, SOC_pred, 'linear', 'extrap');
        dOCV_dSOC = interp1(unique_soc,dOCV_dSOC_values_smooth, SOC_pred, 'linear', 'extrap');

        % 측정 행렬 H
        H = [dOCV_dSOC, 1, 1];

        % 전체 전압 예측
        V_pred_total = OCV_pred + V1_pred + V2_pred + R0 * noisy_I(k);

        % 칼만 게인 계산
        S = H * P_pred * H' + R2; % 측정 노이즈 공분산
        K = P_pred * H' / S;

        % 칼만 게인 저장
        K_2RC(:, k) = K;

        % 추정 단계
        z = noisy_V(k); % 측정값
        x_estimate = x_pred + K * (z - V_pred_total);

        % 추정된 상태 저장
        x_estimate_2RC(k, :) = x_estimate';

        % 오차 공분산 업데이트
        P_estimate = (eye(3) - K * H) * P_pred;

        % SOC, V1, V2 저장
        SOC_est_2RC(k) = x_estimate(1);
        V1_est_2RC(k) = x_estimate(2);
        V2_est_2RC(k) = x_estimate(3);

        % 다음 반복을 위한 추정값 업데이트
        SOC_estimate = x_estimate(1);
    end

    SOC_est_2RC_all{s} = SOC_est_2RC;  
    x_pred_2RC_all{s} = x_pred_2RC;
    x_estimate_2RC_all{s} = x_estimate_2RC;
    K_2RC_all{s} = K_2RC;

end

%% 예시 트립에 대한 그래프 그리기 (첫 번째 트립)
s = 1; % 첫 번째 트립 선택

% 1-RC 모델에 대한 그래프
figure('Name', '1-RC 모델', 'NumberTitle', 'off');

% SOC 관련 서브플롯
subplot(2,1,1);
plot(udds_data(s).t, True_SOC_all{s}, 'k--', 'LineWidth', 1.5);         % 실제 SOC
hold on;
plot(udds_data(s).t, x_pred_1RC_all{s}(:, 1), 'b-', 'LineWidth', 1.5);  % 예측 SOC
plot(udds_data(s).t, x_estimate_1RC_all{s}(:, 1), 'r-', 'LineWidth', 1.5); % 추정 SOC
xlabel('시간 [초]', 'FontSize', labelFontSize);
ylabel('SOC', 'FontSize', labelFontSize);
legend('실제 SOC', '예측 SOC', '추정 SOC', 'FontSize', legendFontSize);
title('SOC 추정 (1-RC 모델)', 'FontSize', titleFontSize);
grid on;

% 칼만 게인 서브플롯
subplot(2,1,2);
plot(udds_data(s).t, K_1RC_all{s}(1, :), 'b-', 'LineWidth', 1.5);  % SOC에 대한 칼만 게인
hold on;
plot(udds_data(s).t, K_1RC_all{s}(2, :), 'r-', 'LineWidth', 1.5);  % V1에 대한 칼만 게인
xlabel('시간 [초]', 'FontSize', labelFontSize);
ylabel('칼만 게인', 'FontSize', labelFontSize);
legend('SOC에 대한 칼만 게인', 'V1에 대한 칼만 게인', 'FontSize', legendFontSize);
title('칼만 게인 (1-RC 모델)', 'FontSize', titleFontSize);
grid on;

% 2-RC 모델에 대한 그래프
figure('Name', '2-RC 모델', 'NumberTitle', 'off');

% SOC 관련 서브플롯
subplot(2,1,1);
plot(udds_data(s).t, True_SOC_all{s}, 'k--', 'LineWidth', 1.5);         % 실제 SOC
hold on;
plot(udds_data(s).t, x_pred_2RC_all{s}(:, 1), 'b-', 'LineWidth', 1.5);  % 예측 SOC
plot(udds_data(s).t, x_estimate_2RC_all{s}(:, 1), 'r-', 'LineWidth', 1.5); % 추정 SOC
xlabel('시간 [초]', 'FontSize', labelFontSize);
ylabel('SOC', 'FontSize', labelFontSize);
legend('실제 SOC', '예측 SOC', '추정 SOC', 'FontSize', legendFontSize);
title('SOC 추정 (2-RC 모델)', 'FontSize', titleFontSize);
grid on;

% 칼만 게인 서브플롯
subplot(2,1,2);
plot(udds_data(s).t, K_2RC_all{s}(1, :), 'b-', 'LineWidth', 1.5);  % SOC에 대한 칼만 게인
hold on;
plot(udds_data(s).t, K_2RC_all{s}(2, :), 'r-', 'LineWidth', 1.5);  % V1에 대한 칼만 게인
plot(udds_data(s).t, K_2RC_all{s}(3, :), 'g-', 'LineWidth', 1.5);  % V2에 대한 칼만 게인
xlabel('시간 [초]', 'FontSize', labelFontSize);
ylabel('칼만 게인', 'FontSize', labelFontSize);
legend('SOC에 대한 칼만 게인', 'V1에 대한 칼만 게인', 'V2에 대한 칼만 게인', 'FontSize', legendFontSize);
title('칼만 게인 (2-RC 모델)', 'FontSize', titleFontSize);
grid on;

% DRT 모델에 대한 그래프
figure('Name', 'DRT 모델', 'NumberTitle', 'off');

% SOC 관련 서브플롯
subplot(2,1,1);
plot(udds_data(s).t, True_SOC_all{s}, 'k--', 'LineWidth', 1.5);         % 실제 SOC
hold on;
plot(udds_data(s).t, x_pred_DRT_all{s}(:, 1), 'b-', 'LineWidth', 1.5);  % 예측 SOC
plot(udds_data(s).t, x_estimate_DRT_all{s}(:, 1), 'r-', 'LineWidth', 1.5); % 추정 SOC
xlabel('시간 [초]', 'FontSize', labelFontSize);
ylabel('SOC', 'FontSize', labelFontSize);
legend('실제 SOC', '예측 SOC', '추정 SOC', 'FontSize', legendFontSize);
title('SOC 추정 (DRT 모델)', 'FontSize', titleFontSize);
grid on;

% 칼만 게인 서브플롯
subplot(2,1,2);
plot(udds_data(s).t, K_DRT_all{s}(1, :), 'b-', 'LineWidth', 1.5);  % SOC에 대한 칼만 게인
xlabel('시간 [초]', 'FontSize', labelFontSize);
ylabel('칼만 게인', 'FontSize', labelFontSize);
legend('SOC에 대한 칼만 게인', 'FontSize', legendFontSize);
title('칼만 게인 (DRT 모델)', 'FontSize', titleFontSize);
grid on;

%% 칼만 필터에 마코프 노이즈 추가하는 함수
function [noisy_I] = Markov(I, noise_percent)

    noise_number = 50;
    initial_state = randsample(1:noise_number, 1); % 현재 상태 랜덤 선택
    mean_noise = mean(I) * noise_percent;
    min_noise = min(I) * noise_percent; % 최소 노이즈
    max_noise = max(I) * noise_percent; % 최대 노이즈
    span = max_noise - min_noise; % 노이즈 범위
    sigma = span / noise_number; % 표준편차
    noise_vector = linspace(mean_noise - span/2, mean_noise + span/2, noise_number); % 노이즈 벡터
    P = zeros(noise_number);

    for i = 1:noise_number
        probabilities = normpdf(noise_vector, noise_vector(i), sigma); % 확률 계산
        P(i, :) = probabilities / sum(probabilities); % 정규화하여 확률 행렬 생성
    end

    noisy_I = zeros(size(I));
    states = zeros(size(I));
    current_state = initial_state;
    
    for m = 1:length(I)
        noisy_I(m) = I(m) + noise_vector(current_state); % 현재 상태의 노이즈 추가
        states(m) = current_state;
        current_state = randsample(1:noise_number, 1 , true, P(current_state, :)); % 다음 상태로 전이
    end

end
