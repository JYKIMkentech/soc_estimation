clc; clear; close all;

%% Seed Setting
%rng(198);

%% Font Size Settings
axisFontSize = 14;
titleFontSize = 16;
legendFontSize = 12;
labelFontSize = 14;

%% Define Color Matrix
c_mat = lines(9);

%% 1. Data Load
% ECM parameters
load('optimized_params_struct_final_2RC.mat'); % Fields: R0, R1, C1, R2, C2, SOC, avgI, m, Crate

% DRT parameters (gamma and tau values)
load('theta_discrete.mat');
load('gamma_est_all.mat', 'gamma_est_all');  % gamma_est_all: [num_trips x num_RC]
load('R0_est_all.mat');

tau_discrete = exp(theta_discrete); % tau values, assume tau_discrete is num_RC x 1 or 1 x num_RC
tau_discrete = tau_discrete(:);     % make sure tau_discrete is a column vector

% SOC-OCV lookup table (from C/20 test)
load('soc_ocv.mat', 'soc_ocv'); % [SOC, OCV]
soc_values = soc_ocv(:, 1);     % SOC values
ocv_values = soc_ocv(:, 2);     % Corresponding OCV values [V]

load('udds_data.mat'); % Struct array 'udds_data' containing fields V, I, t, Time_duration, SOC

Q_batt = 2.7742 ; % [Ah]
SOC_begin_true = 0.9907;
SOC_begin_cc = 0.9907;
epsilon_percent_span = 0.25;
voltage_noise_percent = 0.01;

[unique_ocv, b] = unique(ocv_values);
unique_soc = soc_values(b);

% Compute the derivative of OCV with respect to SOC
dOCV_dSOC_values = gradient(unique_ocv) ./ gradient(unique_soc);
windowSize = 10; 
dOCV_dSOC_values_smooth = movmean(dOCV_dSOC_values, windowSize);

%% 2. Kalman Filter Settings

soc_cov = 1e-11 ;
V_cov = 1e-10;
num_RC = length(tau_discrete);

% P 초기값들
Pcov1_init = [ soc_cov 0;  
               0       V_cov ]; 
Pcov2_init = [ soc_cov    0         0;
               0       V_cov/4     0;
               0         0       V_cov/4]; 
Pcov3_init = zeros(1 + num_RC);
Pcov3_init(1,1) = soc_cov;    
for i = 2:(1 + num_RC)
    Pcov3_init(i,i) = V_cov/201^2; 
end

% Q
Qcov1 = [ soc_cov 0;
          0      V_cov ];  
Qcov2 = [ soc_cov    0        0;
          0      V_cov/4     0;
          0         0     V_cov/4 ]; 
Qcov3 = zeros(1 + num_RC);
Qcov3(1,1) =  soc_cov; 
for i = 2:(1 + num_RC)
    Qcov3(i,i) = V_cov/201^2;
end

% R
Rcov1 = 5.25e-6;
Rcov2 = 5.25e-6;
Rcov3 = 5.25e-6;

%% 3. Extract ECM Parameters
SOC_params = vertcat(optimized_params_struct_final_2RC.SOC);
R0_params = vertcat(optimized_params_struct_final_2RC.R0);
R1_params = vertcat(optimized_params_struct_final_2RC.R1);
C1_params = vertcat(optimized_params_struct_final_2RC.C1);
R2_params = vertcat(optimized_params_struct_final_2RC.R2);
C2_params = vertcat(optimized_params_struct_final_2RC.C2);

%% State & Measurement Functions for EKF
% 1-RC Model
stateFcn1RC = @(x, u) [x(1) + (u(2)/(Q_batt*3600))*u(1);
                       x(2)*exp(-u(2)/(interp1(SOC_params,R1_params,x(1),'linear','extrap')*interp1(SOC_params,C1_params,x(1),'linear','extrap'))) + ...
                       u(1)*interp1(SOC_params,R1_params,x(1),'linear','extrap')*(1 - exp(-u(2)/(interp1(SOC_params,R1_params,x(1),'linear','extrap')*interp1(SOC_params,C1_params,x(1),'linear','extrap'))))];

measFcn1RC = @(x, u) interp1(unique_soc, unique_ocv, x(1),'linear','extrap') + ...
                     x(2) + ...
                     interp1(SOC_params,R0_params,x(1),'linear','extrap')*u(1);

% 2-RC Model
stateFcn2RC = @(x, u) [x(1) + (u(2)/(Q_batt*3600))*u(1);
                       x(2)*exp(-u(2)/(interp1(SOC_params,R1_params,x(1),'linear','extrap')*interp1(SOC_params,C1_params,x(1),'linear','extrap'))) + ...
                       u(1)*interp1(SOC_params,R1_params,x(1),'linear','extrap')*(1 - exp(-u(2)/(interp1(SOC_params,R1_params,x(1),'linear','extrap')*interp1(SOC_params,C1_params,x(1),'linear','extrap'))));
                       x(3)*exp(-u(2)/(interp1(SOC_params,R2_params,x(1),'linear','extrap')*interp1(SOC_params,C2_params,x(1),'linear','extrap'))) + ...
                       u(1)*interp1(SOC_params,R2_params,x(1),'linear','extrap')*(1 - exp(-u(2)/(interp1(SOC_params,R2_params,x(1),'linear','extrap')*interp1(SOC_params,C2_params,x(1),'linear','extrap'))))];

measFcn2RC = @(x, u) interp1(unique_soc, unique_ocv, x(1),'linear','extrap') + ...
                     x(2) + x(3) + ...
                     interp1(SOC_params,R0_params,x(1),'linear','extrap')*u(1);

% Initialize EKF objects
ekf1RC = extendedKalmanFilter(stateFcn1RC, measFcn1RC, [SOC_begin_cc; 0]);
ekf1RC.ProcessNoise = Qcov1;
ekf1RC.MeasurementNoise = Rcov1;
ekf1RC.StateCovariance = Pcov1_init;

ekf2RC = extendedKalmanFilter(stateFcn2RC, measFcn2RC, [SOC_begin_cc; 0; 0]);
ekf2RC.ProcessNoise = Qcov2;
ekf2RC.MeasurementNoise = Rcov2;
ekf2RC.StateCovariance = Pcov2_init;

%% Initialize Variables for DRT Model
initial_SOC_true = SOC_begin_true;
initial_SOC_cc = SOC_begin_cc;

num_trips = length(udds_data);

t_all = [];
True_SOC_all = [];
CC_SOC_all = [];
I_all = [];
noisy_I_all = [];
states_all = [];

x_estimate_1RC_all_trips = [];
residual_1RC_all_trips = [];
x_estimate_2RC_all_trips = [];
residual_2RC_all_trips = [];
x_estimate_DRT_all_trips = [];
residual_DRT_all_trips = [];

previous_trip_end_time = 0;
initial_markov_state = 3;

%% Loop Through Trips
for s = 1:num_trips-16 
    fprintf('Processing Trip %d/%d...\n', s, num_trips);
    
    t = udds_data(s).Time_duration;
    I = udds_data(s).I;
    V = udds_data(s).V;

    if s == 1
        dt = [t(1); diff(t)];
        dt(1) = dt(2);
    else
        dt = [t(1) - previous_trip_end_time; diff(t)];
    end

    fprintf('Trip %d, dt(1): %f\n', s, dt(1));

    [noisy_I, states, final_markov_state, P_markov] = Markov(I, epsilon_percent_span, initial_markov_state); 
    initial_markov_state = final_markov_state;

    noisy_V = V + voltage_noise_percent * V .* randn(size(V)); 

    True_SOC = initial_SOC_true + cumtrapz(t - t(1), I)/(3600 * Q_batt);
    CC_SOC = initial_SOC_cc + cumtrapz(t - t(1), noisy_I)/(3600 * Q_batt);

    % DRT Parameters for this trip
    gamma = gamma_est_all(s,:); 
    gamma = gamma(:);  % make gamma a column vector
    delta_theta = theta_discrete(2) - theta_discrete(1); 
    R_i = gamma * delta_theta;    % R_i: num_RC x 1 vector
    C_i = tau_discrete ./ R_i;    % C_i: num_RC x 1 vector

    R0_current = R0_est_all(s,1); 
    
    stateFcnDRT = @(x, u) [ x(1) + (u(2)/(Q_batt*3600))*u(1);
                            x(2:end).*exp(-u(2)./(R_i.*C_i)) + ...
                            u(1).*R_i.*(1 - exp(-u(2)./(R_i.*C_i))) ];
    
    measFcnDRT = @(x, u) interp1(unique_soc, unique_ocv, x(1),'linear','extrap') + ...
                         sum(x(2:end)) + ...
                         R0_current * u(1);
    
    initial_V_estimate_DRT = zeros(num_RC, 1);
    initialStateGuessDRT = [initial_SOC_cc; initial_V_estimate_DRT];
    ekfDRT = extendedKalmanFilter(stateFcnDRT, measFcnDRT, initialStateGuessDRT);
    ekfDRT.ProcessNoise = Qcov3;
    ekfDRT.MeasurementNoise = Rcov3;
    ekfDRT.StateCovariance = Pcov3_init;
    
    x_estimate_1RC_all = zeros(length(t), 2); 
    residual_1RC_all = zeros(length(t), 1);

    x_estimate_2RC_all = zeros(length(t), 3); 
    residual_2RC_all = zeros(length(t), 1);

    x_estimate_DRT_all = zeros(length(t), 1 + num_RC);
    residual_DRT_all = zeros(length(t), 1);
    
    %% EKF Estimation Loop for This Trip
    for k = 1:length(t)
        % 1-RC Model EKF
        u_1rc = [noisy_I(k), dt(k)];
        try
            predict(ekf1RC, u_1rc);
            xCorr1RC = correct(ekf1RC, noisy_V(k), u_1rc);
            x_estimate_1RC_all(k,:) = xCorr1RC';
            V_pred_1RC = measFcn1RC(xCorr1RC, u_1rc);
            residual_1RC_all(k) = noisy_V(k) - V_pred_1RC;
        catch ME
            fprintf('1RC EKF Error at trip %d, step %d: %s\n', s, k, ME.message);
            x_estimate_1RC_all(k,:) = [NaN, NaN];
            residual_1RC_all(k) = NaN;
        end

        % 2-RC Model EKF
        u_2rc = [noisy_I(k), dt(k)];
        try
            predict(ekf2RC, u_2rc);
            xCorr2RC = correct(ekf2RC, noisy_V(k), u_2rc);
            x_estimate_2RC_all(k,:) = xCorr2RC';
            V_pred_2RC = measFcn2RC(xCorr2RC, u_2rc);
            residual_2RC_all(k) = noisy_V(k) - V_pred_2RC;
        catch ME
            fprintf('2RC EKF Error at trip %d, step %d: %s\n', s, k, ME.message);
            x_estimate_2RC_all(k,:) = [NaN, NaN, NaN];
            residual_2RC_all(k) = NaN;
        end

        % DRT Model EKF
        u_drt = [noisy_I(k), dt(k)];
        try
            predict(ekfDRT, u_drt);
            xCorrDRT = correct(ekfDRT, noisy_V(k), u_drt);
            x_estimate_DRT_all(k,:) = xCorrDRT';
            V_pred_DRT = measFcnDRT(xCorrDRT, u_drt);
            residual_DRT_all(k) = noisy_V(k) - V_pred_DRT;
        catch ME
            fprintf('DRT EKF Error at trip %d, step %d: %s\n', s, k, ME.message);
            x_estimate_DRT_all(k,:) = [NaN, zeros(1, num_RC)];
            residual_DRT_all(k) = NaN;
        end
    end

    %% Update Initial Conditions for Next Trip
    initial_SOC_true = True_SOC(end);
    initial_SOC_cc = CC_SOC(end);

    % 여기서 reset을 사용하는 대신, 단순히 State를 업데이트합니다.
    % ekf1RC
    ekf1RC.State = [initial_SOC_cc; x_estimate_1RC_all(end,2)];
    ekf1RC.StateCovariance = Pcov1_init;

    % ekf2RC
    ekf2RC.State = [initial_SOC_cc; x_estimate_2RC_all(end,2); x_estimate_2RC_all(end,3)];
    ekf2RC.StateCovariance = Pcov2_init;

    % 상태 저장
    t_all = [t_all; t];
    True_SOC_all = [True_SOC_all; True_SOC];
    CC_SOC_all = [CC_SOC_all; CC_SOC];
    I_all = [I_all; I];
    noisy_I_all = [noisy_I_all; noisy_I];
    states_all = [states_all; states]; 

    x_estimate_1RC_all_trips = [x_estimate_1RC_all_trips; x_estimate_1RC_all];
    residual_1RC_all_trips = [residual_1RC_all_trips; residual_1RC_all];

    x_estimate_2RC_all_trips = [x_estimate_2RC_all_trips; x_estimate_2RC_all];
    residual_2RC_all_trips = [residual_2RC_all_trips; residual_2RC_all];

    x_estimate_DRT_all_trips = [x_estimate_DRT_all_trips; x_estimate_DRT_all];
    residual_DRT_all_trips = [residual_DRT_all_trips; residual_DRT_all];

    previous_trip_end_time = t(end);
end

%% Plotting and RMSE
color_true = [0, 0, 0];                
color_cc   = [0,0.4470,0.7410];        
color_1rc  = [0.8350,0.3333,0.0000];   
color_2rc  = [0.9020,0.6235,0.0000];   
color_drt  = [0.8,0.4745,0.6549];

% 1RC Model Results
figure('Name', '1RC Model Results');
subplot(4,1,1);
hold on;
plot(t_all, True_SOC_all, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
plot(t_all, CC_SOC_all, 'b-', 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
plot(t_all, x_estimate_1RC_all_trips(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (1RC)');
ylabel('SOC');
xlabel('Time [s]');
title('SOC over Time (1RC)');
legend('show', 'Location', 'best');
hold off;

subplot(4,1,2);
plot(t_all, x_estimate_1RC_all_trips(:,2), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimated V1 (1RC)');
ylabel('Voltage [V]');
xlabel('Time [s]');
title('V1 over Time (1RC)');
legend('show', 'Location', 'best');

subplot(4,1,3);
plot(t_all, residual_1RC_all_trips, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Residual (1RC)');
ylabel('Residual');
xlabel('Time [s]');
title('Residual over Time (1RC)');
legend('show', 'Location', 'best');

subplot(4,1,4);
plot(t_all, residual_1RC_all_trips, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Residual (1RC)');
xlabel('Time [s]');
ylabel('Residual');
title('Residual over Time (1RC)');
legend('show', 'Location', 'best');

% 2RC Model Results
figure('Name', '2RC Model Results');
subplot(4,1,1);
hold on;
plot(t_all, True_SOC_all, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
plot(t_all, CC_SOC_all, 'b-', 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
plot(t_all, x_estimate_2RC_all_trips(:,1), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (2RC)');
ylabel('SOC');
xlabel('Time [s]');
title('SOC over Time (2RC)');
legend('show', 'Location', 'best');
hold off;

subplot(4,1,2);
plot(t_all, x_estimate_2RC_all_trips(:,2), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Estimated V1 (2RC)');
ylabel('Voltage [V]');
xlabel('Time [s]');
title('V1 over Time (2RC)');
legend('show', 'Location', 'best');

subplot(4,1,3);
plot(t_all, x_estimate_2RC_all_trips(:,3), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Estimated V2 (2RC)');
ylabel('Voltage [V]');
xlabel('Time [s]');
title('V2 over Time (2RC)');
legend('show', 'Location', 'best');

subplot(4,1,4);
plot(t_all, residual_2RC_all_trips, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Residual (2RC)');
ylabel('Residual');
xlabel('Time [s]');
title('Residual over Time (2RC)');
legend('show', 'Location', 'best');

% DRT Model Results
figure('Name', 'DRT Model Results');
subplot(4,1,1);
hold on;
plot(t_all, True_SOC_all, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True SOC');
plot(t_all, CC_SOC_all, 'b-', 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
plot(t_all, x_estimate_DRT_all_trips(:,1), 'm-', 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (DRT)');
ylabel('SOC');
xlabel('Time [s]');
title('SOC over Time (DRT)');
legend('show', 'Location', 'best');
hold off;

subplot(4,1,2);
plot(t_all, sum(x_estimate_DRT_all_trips(:,2:end), 2), 'c--', 'LineWidth', 1.5, 'DisplayName', 'Estimated V (DRT)');
ylabel('Voltage [V]');
xlabel('Time [s]');
title('Total V over Time (DRT)');
legend('show', 'Location', 'best');

subplot(4,1,3);
plot(t_all, residual_DRT_all_trips, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Residual (DRT)');
ylabel('Residual');
xlabel('Time [s]');
title('Residual over Time (DRT)');
legend('show', 'Location', 'best');

subplot(4,1,4);
plot(t_all, residual_DRT_all_trips, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Residual (DRT)');
xlabel('Time [s]');
ylabel('Residual');
title('Residual over Time (DRT)');
legend('show', 'Location', 'best');

% SOC Comparison Across Models
figure('Name', 'SOC Comparison Across Models');
plot(t_all, True_SOC_all, 'Color', color_true, 'LineWidth', 1.5, 'DisplayName', 'True SOC');
hold on;
plot(t_all, CC_SOC_all, 'Color', color_cc, 'LineWidth', 1.5, 'DisplayName', 'CC SOC');
plot(t_all, x_estimate_1RC_all_trips(:,1), 'Color', color_1rc, 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (1RC)');
plot(t_all, x_estimate_2RC_all_trips(:,1), 'Color', color_2rc, 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (2RC)');
plot(t_all, x_estimate_DRT_all_trips(:,1), 'Color', color_drt, 'LineWidth', 1.5, 'DisplayName', 'Estimated SOC (DRT)');
xlabel('Time [s]');
ylabel('SOC');
title('SOC Comparison Across Models');
legend('show', 'Location', 'best');
hold off;

% SOC Error Comparison
figure('Name', 'SOC Error Comparison');
plot(t_all, x_estimate_1RC_all_trips(:,1) - True_SOC_all, 'Color', color_1rc, 'LineWidth', 1.5, 'DisplayName', '1RC-KF SOC Error');
hold on;
plot(t_all, x_estimate_2RC_all_trips(:,1) - True_SOC_all, 'Color', color_2rc, 'LineWidth', 1.5, 'DisplayName', '2RC-KF SOC Error');
plot(t_all, x_estimate_DRT_all_trips(:,1) - True_SOC_all, 'Color', color_drt, 'LineWidth', 1.5, 'DisplayName', 'DRT-KF SOC Error');
plot(t_all, CC_SOC_all - True_SOC_all, 'Color', color_cc, 'LineWidth', 1.5, 'DisplayName', 'CC SOC Error');
xlabel('Time [s]');
ylabel('SOC Error');
title('SOC Error Comparison');
legend('show', 'Location', 'best');
hold off;

% Current Noise Plot
figure;
plot(t_all, I_all - noisy_I_all, 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Noise');
title('Current Noise (I - noisy\_I)');

% Derivative of OCV Plot
figure;
plot(unique_soc, dOCV_dSOC_values_smooth, 'LineWidth', 1.5);
xlabel('SOC');
ylabel('dOCV/dSOC');
title('Derivative of OCV with respect to SOC');
grid on;

% Markov States Evolution Plot
figure('Name', 'Markov States Evolution');
plot(t_all, states_all, 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('State Index');
title('Markov State ');
grid on;

% RMSE Calculation
rmse_True_1RC = sqrt(mean((x_estimate_1RC_all_trips(:,1) - True_SOC_all).^2, 'omitnan'));
rmse_True_2RC = sqrt(mean((x_estimate_2RC_all_trips(:,1) - True_SOC_all).^2, 'omitnan'));
rmse_True_DRT = sqrt(mean((x_estimate_DRT_all_trips(:,1) - True_SOC_all).^2, 'omitnan'));
rmse_True_CC  = sqrt(mean((CC_SOC_all - True_SOC_all).^2, 'omitnan'));

fprintf("\nRMSE of SOC Estimation:\n");
fprintf("CC RMSE: %.6f\n", rmse_True_CC);
fprintf("1RC RMSE: %.6f\n", rmse_True_1RC);
fprintf("2RC RMSE: %.6f\n", rmse_True_2RC);
fprintf("DRT RMSE: %.6f\n", rmse_True_DRT);

%% Function for Adding Markov Noise
function [noisy_I, states, final_state ,P] = Markov(I, epsilon_percent_span, initial_state)
    sigma_percent = 0.001;      
    N = 101; 
    epsilon_vector = linspace(-epsilon_percent_span/2, epsilon_percent_span/2, N); 
    sigma = sigma_percent;

    P = zeros(N);
    for i = 1:N
        probabilities = normpdf(epsilon_vector, epsilon_vector(i), sigma);
        P(i, :) = probabilities / sum(probabilities); 
    end

    current_state = initial_state;
    noisy_I = zeros(size(I));
    states = zeros(size(I));
    epsilon = zeros(size(I));

    for k = 1:length(I)
        epsilon(k) = epsilon_vector(current_state);
        noisy_I(k) = I(k) + abs(I(k)) * epsilon(k); 
        states(k) = current_state; 
        current_state = randsample(1:N, 1, true, P(current_state, :));
    end
    final_state = states(end);
end
