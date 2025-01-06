clc; clear; close all;

%% 1) 폰트 사이즈 설정
axisFontSize   = 14;
titleFontSize  = 16;
legendFontSize = 12;
labelFontSize  = 14;

%% 2) 원하는 색상 팔레트(총 10가지)
% 1:  [0.00000  0.45098  0.76078]
% 2:  [0.93725  0.75294  0.00000]
% 3:  [0.80392  0.32549  0.29803]
% 4:  [0.12549  0.52157  0.30588]
% 5:  [0.57255  0.36863  0.62353]
% 6:  [0.88235  0.52941  0.15294]
% 7:  [0.30196  0.73333  0.83529]
% 8:  [0.93333  0.29803  0.59216]
% 9:  [0.49412  0.38039  0.28235]
% 10: [0.45490  0.46275  0.47059]

%% 3) 적용할 컬러 정의(예시)
% - 필요에 따라 다른 색상 번호를 골라 자유롭게 바꿔 사용하세요.
color_true = [0, 0, 0];                          % True SOC: 검정
color_cc   = [0.0000, 0.45098, 0.76078];         % #1
color_1rc  = [0.12549  0.52157  0.30588];        % #3
color_2rc  = [0.80392, 0.32549, 0.29803];        % #2
color_drt  = [0.93725, 0.75294, 0.00000];        % #5

%% 4) Seed Setting
rng(208);

%% 5) 데이터 불러오기
% ECM parameters
load('optimized_params_struct_final_2RC.mat');  % Fields: R0, R1, C1, R2, C2, SOC, avgI, m, Crate

% DRT parameters (gamma and tau values)
load('theta_discrete.mat');
load('gamma_est_all.mat', 'gamma_est_all');  % Note: Removed SOC_mid_all
load('R0_est_all.mat');
tau_discrete = exp(theta_discrete);          % tau values

% SOC-OCV lookup table (from C/20 test)
load('soc_ocv.mat', 'soc_ocv');  % [SOC, OCV]
soc_values = soc_ocv(:, 1);      % SOC values
ocv_values = soc_ocv(:, 2);      % Corresponding OCV values [V]
[unique_ocv, b] = unique(ocv_values);
unique_soc = soc_values(b);

% Compute the derivative of OCV with respect to SOC
dOCV_dSOC_values = gradient(unique_ocv) ./ gradient(unique_soc);
windowSize = 10;
dOCV_dSOC_values_smooth = movmean(dOCV_dSOC_values, windowSize);

% UDDS data
load('udds_data.mat');           % Struct array 'udds_data' containing fields V, I, t, Time_duration, SOC

%% 배터리 기본 설정
Q_batt = 2.7742;         % [Ah]
SOC_begin_true = 0.9907;
SOC_begin_cc   = 0.9907;
epsilon_percent_span = 4;         % ex) 0.02 --> 4
voltage_noise_percent = 0.01;     % 전압에 가우시안 잡음 비율

%% 6) Kalman 필터에서 사용할 초기 Covariance 및 파라미터 설정
Voltage_cov = logspace(-4,-20,17);
soc_cov = 0.2e-12;
V_cov   = 1e-8; % Voltage_cov(1);

% Number of RC elements for DRT model
num_RC = length(tau_discrete);

% 초기 P행렬
Pcov1_init = [soc_cov/50, 0;
              0,         V_cov];
Pcov2_init = [soc_cov/5,   0,       0;
              0,          V_cov/4,  0;
              0,          0,        V_cov/4]; % [SOC; V1; V2]

Pcov3_init = zeros(1 + num_RC);
Pcov3_init(1,1) = 1 * soc_cov;
for i = 2:(1 + num_RC)
    Pcov3_init(i,i) = V_cov / (201^2);
end

% 프로세스 잡음 Q
Qcov1 = [soc_cov/50, 0;
         0,          V_cov];
Qcov2 = [soc_cov/5,    0,          0;
         0,           V_cov/4,    0;
         0,           0,          V_cov/4];
Qcov3 = zeros(1 + num_RC);
Qcov3(1,1) = 1 * soc_cov;
for i = 2:(1 + num_RC)
    Qcov3(i,i) = V_cov / (201^2);
end

% 측정 잡음 R
Rcov1 = 5.25e-6;
Rcov2 = 5.25e-6;
Rcov3 = 5.25e-6;

%% 7) ECM 파라미터 추출
SOC_params = vertcat(optimized_params_struct_final_2RC.SOC);
R0_params  = vertcat(optimized_params_struct_final_2RC.R0);
R1_params  = vertcat(optimized_params_struct_final_2RC.R1);
C1_params  = vertcat(optimized_params_struct_final_2RC.C1);
R2_params  = vertcat(optimized_params_struct_final_2RC.R2);
C2_params  = vertcat(optimized_params_struct_final_2RC.C2);

%% 8) Kalman Filter 준비
num_trips = length(udds_data);

% (1) 1RC 초기화
initial_SOC_true         = SOC_begin_true;
initial_SOC_cc           = SOC_begin_cc;
initial_P_estimate_1RC   = Pcov1_init;
initial_SOC_estimate_1RC = initial_SOC_cc;
initial_V1_estimate_1RC  = 0;

% 결과 저장용
t_all = [];
True_SOC_all = [];
CC_SOC_all   = [];
x_pred_1RC_all_trips      = [];
x_estimate_1RC_all_trips  = [];
KG_1RC_all_trips          = [];
residual_1RC_all_trips    = [];
I_all     = [];
noisy_I_all = [];
states_all  = [];

% (2) 2RC 초기화
initial_P_estimate_2RC   = Pcov2_init;
initial_SOC_estimate_2RC = initial_SOC_cc;
initial_V1_estimate_2RC  = 0;
initial_V2_estimate_2RC  = 0;

x_pred_2RC_all_trips     = [];
x_estimate_2RC_all_trips = [];
KG_2RC_all_trips         = [];
residual_2RC_all_trips   = [];

% (3) DRT 초기화
initial_P_estimate_DRT   = Pcov3_init;
initial_SOC_estimate_DRT = initial_SOC_cc;
initial_V_estimate_DRT   = zeros(num_RC, 1);

x_pred_DRT_all_trips      = [];
x_estimate_DRT_all_trips  = [];
KG_DRT_all_trips          = [];
residual_DRT_all_trips    = [];

previous_trip_end_time = 0;
initial_markov_state  = 50;   % Markov 잡음 초기 상태 (예시)

%% 9) 메인 루프 (각 Trip마다 반복)
for s = 1:num_trips-4
    fprintf('Processing Trip %d/%d...\n', s, num_trips);

    t = udds_data(s).Time_duration;
    I = udds_data(s).I;
    V = udds_data(s).V;

    if s == 1
        dt = [t(1); diff(t)];
        dt(1) = dt(2);   % 초반 구간 차이 보정
    else
        dt = [t(1) - previous_trip_end_time; diff(t)];
    end

    fprintf('Trip %d, dt(1): %f\n', s, dt(1));

    % --> (가) Current에 Markov 잡음 추가
    [noisy_I, states, final_markov_state, P] = Markov(I, epsilon_percent_span, initial_markov_state);
    initial_markov_state = final_markov_state;

    % --> (나) Voltage에 백색 잡음 추가
    noisy_V = V + voltage_noise_percent * V .* randn(size(V));

    % --> (다) True_SOC, CC_SOC 계산
    True_SOC = initial_SOC_true + cumtrapz(t - t(1), I)       / (3600 * Q_batt);
    CC_SOC   = initial_SOC_cc   + cumtrapz(t - t(1), noisy_I) / (3600 * Q_batt);

    % ============== 1RC ==============
    SOC_est_1RC = zeros(length(t), 1);
    V1_est_1RC  = zeros(length(t), 1);

    SOC_estimate_1RC = initial_SOC_estimate_1RC;
    P_estimate_1RC   = initial_P_estimate_1RC;
    V1_estimate_1RC  = initial_V1_estimate_1RC;

    x_pred_1RC_all     = zeros(length(t), 2);
    KG_1RC_all         = zeros(length(t), 2);
    residual_1RC_all   = zeros(length(t), 1);
    x_estimate_1RC_all = zeros(length(t), 2);

    % ============== 2RC ==============
    SOC_est_2RC = zeros(length(t), 1);
    V1_est_2RC  = zeros(length(t), 1);
    V2_est_2RC  = zeros(length(t), 1);

    SOC_estimate_2RC = initial_SOC_estimate_2RC;
    V1_estimate_2RC  = initial_V1_estimate_2RC;
    V2_estimate_2RC  = initial_V2_estimate_2RC;
    P_estimate_2RC   = initial_P_estimate_2RC;

    x_pred_2RC_all     = zeros(length(t), 3);
    x_estimate_2RC_all = zeros(length(t), 3);
    KG_2RC_all         = zeros(length(t), 3);
    residual_2RC_all   = zeros(length(t), 1);

    % ============== DRT ==============
    gamma      = gamma_est_all(s,:);
    delta_theta = theta_discrete(2) - theta_discrete(1);
    R_i = gamma * delta_theta;              % 각 이산화된 freq bin에 대한 R
    C_i = tau_discrete' ./ R_i;

    SOC_est_DRT = zeros(length(t), 1);
    V_est_DRT   = zeros(length(t), num_RC);

    SOC_estimate_DRT = initial_SOC_estimate_DRT;
    V_estimate_DRT   = initial_V_estimate_DRT;
    P_estimate_DRT   = initial_P_estimate_DRT;

    x_pred_DRT_all      = zeros(length(t), 1 + num_RC);
    x_estimate_DRT_all  = zeros(length(t), 1 + num_RC);
    KG_DRT_all          = zeros(length(t), 1 + num_RC);
    residual_DRT_all    = zeros(length(t), 1);

    %% 타임스텝 루프
    for k = 1:length(t)
        %% (1) 1RC 예측 단계
        R0_1RC = interp1(SOC_params, R0_params, SOC_estimate_1RC, 'linear', 'extrap');
        R1_1RC = interp1(SOC_params, R1_params, SOC_estimate_1RC, 'linear', 'extrap');
        C1_1RC = interp1(SOC_params, C1_params, SOC_estimate_1RC, 'linear', 'extrap');

        if k == 1
            if s == 1
                V1_pred = noisy_I(k) * R1_1RC * (1 - exp(-dt(k)/(R1_1RC*C1_1RC)));
            else
                % 두 번째 트립부터는 이전 값 이어서
                V1_pred = V1_estimate_1RC * exp(-dt(k)/(R1_1RC*C1_1RC)) ...
                          + noisy_I(k)*R1_1RC*(1 - exp(-dt(k)/(R1_1RC*C1_1RC)));
            end
        else
            V1_pred = V1_estimate_1RC * exp(-dt(k)/(R1_1RC*C1_1RC)) ...
                      + noisy_I(k)*R1_1RC*(1 - exp(-dt(k)/(R1_1RC*C1_1RC)));
        end

        SOC_pred_1RC = SOC_estimate_1RC + (dt(k)/(Q_batt*3600))*noisy_I(k);

        x_pred = [SOC_pred_1RC; V1_pred];
        x_pred_1RC_all(k, :) = x_pred';

        % 예측 공분산
        A = [1, 0;
             0, exp(-dt(k)/(R1_1RC*C1_1RC))];
        P_pred_1RC = A * P_estimate_1RC * A' + Qcov1;

        % OCV 및 미분
        OCV_pred   = interp1(unique_soc, unique_ocv, SOC_pred_1RC, 'linear', 'extrap');
        dOCV_dSOC  = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred_1RC, 'linear', 'extrap');
        H          = [dOCV_dSOC, 1];
        V_pred_total = OCV_pred + V1_pred + R0_1RC*noisy_I(k);

        % 보정 단계
        S_k      = H * P_pred_1RC * H' + Rcov1;
        KG_1RC   = (P_pred_1RC * H') / S_k;
        KG_1RC_all(k, :) = KG_1RC';

        z        = noisy_V(k);
        residual = z - V_pred_total;
        residual_1RC_all(k) = residual;

        x_estimate = x_pred + KG_1RC * residual;

        SOC_estimate_1RC = x_estimate(1);
        V1_estimate_1RC  = x_estimate(2);
        x_estimate_1RC_all(k, :) = x_estimate';

        P_estimate_1RC = (eye(2) - KG_1RC*H) * P_pred_1RC;

        SOC_est_1RC(k) = x_estimate(1);
        V1_est_1RC(k)  = x_estimate(2);

        %% (2) 2RC 예측 단계
        R0_2RC = interp1(SOC_params, R0_params, SOC_estimate_2RC, 'linear', 'extrap');
        R1_2RC = interp1(SOC_params, R1_params, SOC_estimate_2RC, 'linear', 'extrap');
        C1_2RC = interp1(SOC_params, C1_params, SOC_estimate_2RC, 'linear', 'extrap');
        R2_2RC = interp1(SOC_params, R2_params, SOC_estimate_2RC, 'linear', 'extrap');
        C2_2RC = interp1(SOC_params, C2_params, SOC_estimate_2RC, 'linear', 'extrap');

        if k == 1
            if s == 1
                V1_pred = noisy_I(k)*R1_2RC*(1 - exp(-dt(k)/(R1_2RC*C1_2RC)));
                V2_pred = noisy_I(k)*R2_2RC*(1 - exp(-dt(k)/(R2_2RC*C2_2RC)));
            else
                V1_pred = V1_estimate_2RC;
                V2_pred = V2_estimate_2RC;
            end
        else
            V1_pred = V1_estimate_2RC * exp(-dt(k)/(R1_2RC*C1_2RC)) ...
                      + noisy_I(k)*R1_2RC*(1 - exp(-dt(k)/(R1_2RC*C1_2RC)));
            V2_pred = V2_estimate_2RC * exp(-dt(k)/(R2_2RC*C2_2RC)) ...
                      + noisy_I(k)*R2_2RC*(1 - exp(-dt(k)/(R2_2RC*C2_2RC)));
        end

        SOC_pred_2RC = SOC_estimate_2RC + (dt(k)/(Q_batt*3600))*noisy_I(k);

        x_pred = [SOC_pred_2RC; V1_pred; V2_pred];
        x_pred_2RC_all(k, :) = x_pred';

        A = [1, 0, 0;
             0, exp(-dt(k)/(R1_2RC*C1_2RC)), 0;
             0, 0, exp(-dt(k)/(R2_2RC*C2_2RC))];
        P_pred_2RC = A * P_estimate_2RC * A' + Qcov2;

        OCV_pred   = interp1(unique_soc, unique_ocv, SOC_pred_2RC, 'linear', 'extrap');
        dOCV_dSOC  = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred_2RC, 'linear', 'extrap');
        H          = [dOCV_dSOC, 1, 1];
        V_pred_total = OCV_pred + V1_pred + V2_pred + R0_2RC*noisy_I(k);

        S_k    = H * P_pred_2RC * H' + Rcov2;
        KG_2RC = (P_pred_2RC * H') / S_k;
        KG_2RC_all(k, :) = KG_2RC';

        z        = noisy_V(k);
        residual = z - V_pred_total;
        residual_2RC_all(k) = residual;

        x_estimate = x_pred + KG_2RC*residual;
        SOC_estimate_2RC = x_estimate(1);
        V1_estimate_2RC  = x_estimate(2);
        V2_estimate_2RC  = x_estimate(3);

        x_estimate_2RC_all(k, :) = x_estimate';
        P_estimate_2RC = (eye(3) - KG_2RC*H) * P_pred_2RC;

        SOC_est_2RC(k) = x_estimate(1);
        V1_est_2RC(k)  = x_estimate(2);
        V2_est_2RC(k)  = x_estimate(3);

        %% (3) DRT 예측 단계
        SOC_pred_DRT = SOC_estimate_DRT + (dt(k)/(Q_batt*3600))*noisy_I(k);

        if k == 1
            if s == 1
                V_prev_DRT = zeros(num_RC,1);
            else
                V_prev_DRT = V_estimate_DRT;
            end
        else
            V_prev_DRT = V_estimate_DRT;
        end

        V_pred_DRT = zeros(num_RC, 1);
        for i = 1:num_RC
            V_pred_DRT(i) = V_prev_DRT(i)*exp(-dt(k)/(R_i(i)*C_i(i))) ...
                            + noisy_I(k)*R_i(i)*(1 - exp(-dt(k)/(R_i(i)*C_i(i))));
        end

        x_pred = [SOC_pred_DRT; V_pred_DRT];
        x_pred_DRT_all(k, :) = x_pred';

        A_DRT = diag([1; exp(-dt(k) ./ (R_i .* C_i))']);
        P_pred_DRT = A_DRT * P_estimate_DRT * A_DRT' + Qcov3;

        OCV_pred   = interp1(unique_soc, unique_ocv, SOC_pred_DRT, 'linear', 'extrap');
        dOCV_dSOC  = interp1(unique_soc, dOCV_dSOC_values_smooth, SOC_pred_DRT, 'linear', 'extrap');
        H_DRT      = [dOCV_dSOC, ones(1,num_RC)];

        % R0_est_all(s,1)은 DRT 추정된 R0값
        V_pred_total_DRT = OCV_pred + sum(V_pred_DRT) + R0_est_all(s,1)*noisy_I(k);

        S_k_DRT = H_DRT * P_pred_DRT * H_DRT' + Rcov3;
        KG_DRT  = (P_pred_DRT * H_DRT') / S_k_DRT;
        KG_DRT_all(k, :) = KG_DRT';

        z            = noisy_V(k);
        residual_DRT = z - V_pred_total_DRT;
        residual_DRT_all(k) = residual_DRT;

        x_estimate_DRT = x_pred + KG_DRT * residual_DRT;
        SOC_estimate_DRT = x_estimate_DRT(1);
        V_estimate_DRT   = x_estimate_DRT(2:end);

        x_estimate_DRT_all(k, :) = x_estimate_DRT';
        P_estimate_DRT = (eye(1+num_RC) - KG_DRT*H_DRT) * P_pred_DRT;

        SOC_est_DRT(k) = SOC_estimate_DRT;
        V_est_DRT(k,:) = V_estimate_DRT';
    end

    % Trip 종료 후 초기값 업데이트
    initial_SOC_true = True_SOC(end);
    initial_SOC_cc   = CC_SOC(end);

    initial_SOC_estimate_1RC = SOC_estimate_1RC;
    initial_P_estimate_1RC   = P_estimate_1RC;
    initial_V1_estimate_1RC  = V1_estimate_1RC;

    initial_SOC_estimate_2RC = SOC_estimate_2RC;
    initial_V1_estimate_2RC  = V1_estimate_2RC;
    initial_V2_estimate_2RC  = V2_estimate_2RC;
    initial_P_estimate_2RC   = P_estimate_2RC;

    initial_SOC_estimate_DRT = SOC_estimate_DRT;
    initial_V_estimate_DRT   = V_estimate_DRT;
    initial_P_estimate_DRT   = P_estimate_DRT;

    previous_trip_end_time = t(end);

    % 결과 누적
    t_all = [t_all; t];
    True_SOC_all  = [True_SOC_all; True_SOC];
    CC_SOC_all    = [CC_SOC_all;   CC_SOC];
    x_pred_1RC_all_trips      = [x_pred_1RC_all_trips;     x_pred_1RC_all];
    x_estimate_1RC_all_trips  = [x_estimate_1RC_all_trips; x_estimate_1RC_all];
    KG_1RC_all_trips          = [KG_1RC_all_trips;         KG_1RC_all];
    residual_1RC_all_trips    = [residual_1RC_all_trips;   residual_1RC_all];
    I_all     = [I_all; I];
    noisy_I_all = [noisy_I_all; noisy_I];
    states_all  = [states_all; states];

    x_pred_2RC_all_trips     = [x_pred_2RC_all_trips;     x_pred_2RC_all];
    x_estimate_2RC_all_trips = [x_estimate_2RC_all_trips; x_estimate_2RC_all];
    KG_2RC_all_trips         = [KG_2RC_all_trips;         KG_2RC_all];
    residual_2RC_all_trips   = [residual_2RC_all_trips;   residual_2RC_all];

    x_pred_DRT_all_trips      = [x_pred_DRT_all_trips;     x_pred_DRT_all];
    x_estimate_DRT_all_trips  = [x_estimate_DRT_all_trips; x_estimate_DRT_all];
    KG_DRT_all_trips          = [KG_DRT_all_trips;         KG_DRT_all];
    residual_DRT_all_trips    = [residual_DRT_all_trips;   residual_DRT_all];

end

%% 10) 결과 Plotting 및 RMSE 계산

% RMSE 계산
rmse_True_1RC = sqrt(mean((x_estimate_1RC_all_trips(:,1) - True_SOC_all).^2));
rmse_True_2RC = sqrt(mean((x_estimate_2RC_all_trips(:,1) - True_SOC_all).^2));
rmse_True_DRT = sqrt(mean((x_estimate_DRT_all_trips(:,1) - True_SOC_all).^2));
rmse_True_CC  = sqrt(mean((CC_SOC_all - True_SOC_all).^2));

fprintf("\nRMSE of SOC Estimation:\n");
fprintf("CC RMSE: %.6f\n",  rmse_True_CC);
fprintf("1RC RMSE: %.6f\n", rmse_True_1RC);
fprintf("2RC RMSE: %.6f\n", rmse_True_2RC);
fprintf("DRT RMSE: %.6f\n", rmse_True_DRT);

%% (A) 1RC Model Results
figure('Name', '1RC Model Results');
subplot(4,1,1);
hold on;
plot(t_all, True_SOC_all,          '-', 'LineWidth', 1.5, 'Color', color_true, 'DisplayName', 'True SOC');
plot(t_all, CC_SOC_all,            '-', 'LineWidth', 1.5, 'Color', color_cc,   'DisplayName', 'CC SOC');
plot(t_all, x_estimate_1RC_all_trips(:,1), 'LineWidth', 1.5, 'Color', color_1rc, 'DisplayName', 'Estimated SOC (1RC)');
ylabel('SOC'); xlabel('Time [s]');
title('SOC over Time (1RC)');
legend('show', 'Location', 'best');
hold off;

subplot(4,1,2);
plot(t_all, x_estimate_1RC_all_trips(:,2), '--', 'LineWidth', 1.5, 'Color', color_1rc, 'DisplayName', 'Estimated V1 (1RC)');
ylabel('Voltage [V]'); xlabel('Time [s]');
title('V1 over Time (1RC)');
legend('show', 'Location', 'best');

subplot(4,1,3);
yyaxis left;
plot(t_all, KG_1RC_all_trips(:,1), '-', 'LineWidth', 1.5, 'Color', color_cc, 'DisplayName', 'Kalman Gain SOC (1RC)');
ylabel('Kalman Gain for SOC');
yyaxis right;
plot(t_all, residual_1RC_all_trips, '-', 'LineWidth', 1.5, 'Color', color_drt, 'DisplayName', 'Residual (1RC)');
ylabel('Residual');
xlabel('Time [s]');
title('Kalman Gain and Residual over Time (1RC)');
legend('show', 'Location', 'best');

subplot(4,1,4);
plot(t_all, residual_1RC_all_trips, '-', 'LineWidth', 1.5, 'Color', color_drt, 'DisplayName', 'Residual (1RC)');
xlabel('Time [s]'); ylabel('Residual');
title('Residual over Time (1RC)');
legend('show', 'Location', 'best');

%% (B) 2RC Model Results
figure('Name', '2RC Model Results');
subplot(4,1,1); hold on;
plot(t_all, True_SOC_all, '-', 'LineWidth', 1.5, 'Color', color_true, 'DisplayName', 'True SOC');
plot(t_all, CC_SOC_all,   '-', 'LineWidth', 1.5, 'Color', color_cc,   'DisplayName', 'CC SOC');
plot(t_all, x_estimate_2RC_all_trips(:,1), 'LineWidth', 1.5, 'Color', color_2rc, 'DisplayName', 'Estimated SOC (2RC)');
ylabel('SOC'); xlabel('Time [s]');
title('SOC over Time (2RC)');
legend('show', 'Location', 'best');
hold off;

subplot(4,1,2);
plot(t_all, x_estimate_2RC_all_trips(:,2), '--', 'LineWidth', 1.5, 'Color', color_2rc, 'DisplayName', 'Estimated V1 (2RC)');
ylabel('Voltage [V]'); xlabel('Time [s]');
title('V1 over Time (2RC)');
legend('show', 'Location', 'best');

subplot(4,1,3);
plot(t_all, x_estimate_2RC_all_trips(:,3), '--', 'LineWidth', 1.5, 'Color', color_drt, 'DisplayName', 'Estimated V2 (2RC)');
ylabel('Voltage [V]'); xlabel('Time [s]');
title('V2 over Time (2RC)');
legend('show', 'Location', 'best');

subplot(4,1,4);
yyaxis left;
plot(t_all, KG_2RC_all_trips(:,1), '-', 'LineWidth', 1.5, 'Color', color_cc, 'DisplayName', 'Kalman Gain SOC (2RC)');
ylabel('Kalman Gain for SOC');
yyaxis right;
plot(t_all, residual_2RC_all_trips, '-', 'LineWidth', 1.5, 'Color', color_drt, 'DisplayName', 'Residual (2RC)');
ylabel('Residual'); xlabel('Time [s]');
title('Kalman Gain and Residual over Time (2RC)');
legend('show', 'Location', 'best');

%% (C) DRT Model Results
figure('Name', 'DRT Model Results');
subplot(4,1,1); hold on;
plot(t_all, True_SOC_all, '-', 'LineWidth', 1.5, 'Color', color_true, 'DisplayName', 'True SOC');
plot(t_all, CC_SOC_all,   '-', 'LineWidth', 1.5, 'Color', color_cc,   'DisplayName', 'CC SOC');
plot(t_all, x_estimate_DRT_all_trips(:,1), '-', 'LineWidth', 1.5, 'Color', color_drt, 'DisplayName', 'Estimated SOC (DRT)');
ylabel('SOC'); xlabel('Time [s]');
title('SOC over Time (DRT)');
legend('show', 'Location', 'best');
hold off;

subplot(4,1,2);
plot(t_all, sum(x_estimate_DRT_all_trips(:,2:end),2), '--', 'LineWidth', 1.5, 'Color', color_drt, 'DisplayName', 'Estimated V (DRT)');
ylabel('Voltage [V]'); xlabel('Time [s]');
title('Total V over Time (DRT)');
legend('show', 'Location', 'best');

subplot(4,1,3);
yyaxis left;
plot(t_all, KG_DRT_all_trips(:,1), '-', 'LineWidth', 1.5, 'Color', color_cc, 'DisplayName', 'Kalman Gain SOC (DRT)');
ylabel('Kalman Gain for SOC');
yyaxis right;
plot(t_all, residual_DRT_all_trips, '-', 'LineWidth', 1.5, 'Color', color_1rc, 'DisplayName', 'Residual (DRT)');
ylabel('Residual'); xlabel('Time [s]');
title('Kalman Gain and Residual over Time (DRT)');
legend('show', 'Location', 'best');

subplot(4,1,4);
plot(t_all, residual_DRT_all_trips, '-', 'LineWidth', 1.5, 'Color', color_1rc, 'DisplayName', 'Residual (DRT)');
xlabel('Time [s]'); ylabel('Residual');
title('Residual over Time (DRT)');
legend('show', 'Location', 'best');

%% (D) SOC Comparison Across Models
figure('Name', 'SOC Comparison Across Models');
plot(t_all, True_SOC_all, '-', 'LineWidth', 3, 'Color', color_true, 'DisplayName', 'True SOC');
hold on;
plot(t_all, CC_SOC_all,   '-', 'LineWidth', 3, 'Color', color_cc,   'DisplayName', 'CC SOC');
plot(t_all, x_estimate_1RC_all_trips(:,1), '-', 'LineWidth', 3, 'Color', color_1rc, 'DisplayName', 'Estimated SOC (1RC)');
plot(t_all, x_estimate_2RC_all_trips(:,1), '-', 'LineWidth', 3, 'Color', color_2rc, 'DisplayName', 'Estimated SOC (2RC)');
plot(t_all, x_estimate_DRT_all_trips(:,1), '-', 'LineWidth', 3, 'Color', color_drt, 'DisplayName', 'Estimated SOC (DRT)');
xlabel('Time [s]'); ylabel('SOC');
title('SOC Comparison Across Models');
legend('show', 'Location', 'best');
hold off;

%% (E) SOC Error Comparison
figure('Name', 'SOC Error Comparison');
plot(t_all, x_estimate_1RC_all_trips(:,1) - True_SOC_all, '-', 'LineWidth', 3, 'Color', color_1rc, 'DisplayName', '1RC-KF SOC Error');
hold on;
plot(t_all, x_estimate_2RC_all_trips(:,1) - True_SOC_all, '-', 'LineWidth', 3, 'Color', color_2rc, 'DisplayName', '2RC-KF SOC Error');
plot(t_all, x_estimate_DRT_all_trips(:,1) - True_SOC_all, '-', 'LineWidth', 3, 'Color', color_drt, 'DisplayName', 'DRT-KF SOC Error');
plot(t_all, CC_SOC_all - True_SOC_all,       '-', 'LineWidth', 3, 'Color', color_cc, 'DisplayName', 'CC SOC Error');
xlabel('Time [s]'); ylabel('SOC Error');
title('SOC Error Comparison');
legend('show', 'Location', 'best');
hold off;

%% (F) Current Noise (I - noisy_I)
figure('Name', 'Current Noise');
plot(t_all, I_all - noisy_I_all, '-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Noise');
title('Current Noise (I - noisy_I)');

%% (G) dOCV/dSOC 곡선
figure('Name', 'dOCV/dSOC');
plot(unique_soc, dOCV_dSOC_values_smooth, '-', 'LineWidth', 1.5);
xlabel('SOC'); ylabel('dOCV/dSOC');
title('Derivative of OCV with respect to SOC');
grid on;

%% (H) Markov State Evolution
figure('Name', 'Markov States Evolution');
plot(t_all, states_all, '-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('State Index');
title('Markov State');
grid on;

%%%%%========================================================
%%%%%   (부록) Markov 노이즈 생성 함수
%%%%%========================================================
function [noisy_I, states, final_state, P] = Markov(I, epsilon_percent_span, initial_state)
    % sigma_percent를 조정해 잡음 전이 폭/분산 등을 설정
    sigma_percent = 0.1;  % 0.1 정도로 예시
    N = 101;
    epsilon_vector = linspace(-epsilon_percent_span/2, epsilon_percent_span/2, N);
    sigma = sigma_percent;

    % 전이 확률행렬 P 구성 (정규분포 기반)
    P = zeros(N);
    for i = 1:N
        probabilities = normpdf(epsilon_vector, epsilon_vector(i), sigma);
        P(i, :) = probabilities / sum(probabilities);
    end

    current_state = initial_state;
    noisy_I = zeros(size(I));
    states  = zeros(size(I));

    for k = 1:length(I)
        eps_k = epsilon_vector(current_state);
        noisy_I(k) = I(k) + abs(I(k)) * eps_k;  % ex) I에 상대적 퍼센트 잡음
        states(k)  = current_state;
        % 다음 상태로 랜덤 이동
        current_state = randsample(1:N, 1, true, P(current_state, :));
    end

    final_state = states(end);
end

