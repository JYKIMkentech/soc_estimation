clear; clc; close all;

%% 0. 폰트 크기 및 색상 매트릭스 설정
% Font size settings
axisFontSize = 14;      % 축의 숫자 크기
titleFontSize = 16;     % 제목의 폰트 크기
legendFontSize = 12;    % 범례의 폰트 크기
labelFontSize = 14;     % xlabel 및 ylabel의 폰트 크기

% Color matrix 설정
c_mat = lines(9);  % 9개의 고유한 색상 정의

%% 1. 데이터 로드
% UDDS 주행 데이터를 로드합니다.
load('udds_data.mat');  % 'udds_data' 구조체를 로드합니다.

% SOC-OCV 데이터를 로드합니다.
load('soc_ocv.mat', 'soc_ocv');
soc_values = soc_ocv(:, 1);  % SOC 값
ocv_values = soc_ocv(:, 2);  % OCV 값

%% 2. Parameter 설정

n = 201;
dur = 1370; % [sec]
SOC_begin = 0.9907 ; %0.9907; % 초기 SOC 값
Q_batt = 2.7742; % [Ah]

%% 3. 각 trip에 대한 DRT 추정 (Quadprog 사용)

num_trips = length(udds_data);

% 결과 저장을 위한 배열 사전 할당
gamma_est_all = zeros(num_trips, n);  % 모든 트립에 대해
R0_est_all = zeros(num_trips, 1);
V_est_all = cell(num_trips, 1); % 추정된 전압 저장을 위한 셀 배열
SOC_all = cell(num_trips, 1);   
SOC_mid_all = zeros(num_trips,1);

for s = 1:num_trips-1
    fprintf('Processing Trip %d/%d...\n', s, num_trips);
    
    % DRT_estimation_aug input 
    Time_duration = udds_data(s).Time_duration;
    t = udds_data(s).t;    % 시간 벡터 [초]
    I = udds_data(s).I;    % 전류 벡터 [A]
    V = udds_data(s).V;    % 전압 벡터 [V]
    lambda_hat = 3.79e-10;      % 정규화 파라미터
    dt = [t(1); diff(t)];  % 첫 번째 dt는 t(1)으로 설정
    SOC = SOC_begin + cumtrapz(t, I) / (Q_batt * 3600); % SOC 계산
    SOC_all{s} = SOC;  % SOC 저장 (셀 배열 사용)
    SOC_mid_all(s) = mean(SOC);

    [gamma_est, R0_est, V_est , theta_discrete , W, ~, ~] = DRT_estimation_aug(t, I, V, lambda_hat, n, dt, dur, SOC, soc_values, ocv_values);
    
    gamma_est_all(s, :) = gamma_est';
    R0_est_all(s) = R0_est;
    V_est_all{s} = V_est;  % 셀 배열에 저장
    
    % SOC 업데이트 
    SOC_begin = SOC(end);

    %% 각 Trip에 대해 Figure 생성
    figure('Name', ['Trip ', num2str(s)], 'NumberTitle', 'off');
    set(gcf, 'Position', [150, 150, 1200, 800]);  % Figure 크기 조정

    % 첫 번째 subplot: Voltage, Estimated Voltage, Current
    subplot(2,1,1);
    
    % 왼쪽 Y축: Voltage
    yyaxis left
    plot(t, V, 'Color', c_mat(1, :), 'LineWidth', 3, 'DisplayName', 'Measured Voltage');
    hold on;
    plot(t, V_est, '--', 'Color', c_mat(2, :), 'LineWidth', 3, 'DisplayName', 'Estimated Voltage');
    ylabel('Voltage [V]', 'FontSize', labelFontSize, 'Color', c_mat(1, :));
    set(gca, 'YColor', c_mat(1, :));  % 왼쪽 Y축 색상 설정

    % 오른쪽 Y축: Current
    yyaxis right
    plot(t, I, '-', 'Color', c_mat(3, :), 'LineWidth', 3, 'DisplayName', 'Current');
    ylabel('Current [A]', 'FontSize', labelFontSize, 'Color', c_mat(3, :));
    set(gca, 'YColor', c_mat(3, :));  % 오른쪽 Y축 색상 설정

    xlabel('Time [s]', 'FontSize', labelFontSize);
    title(sprintf('Trip %d: Voltage and Current', s), 'FontSize', titleFontSize);
    legend('FontSize', legendFontSize);
    set(gca, 'FontSize', axisFontSize);
    hold off;

    % 두 번째 subplot: DRT (theta vs gamma)
    subplot(2,1,2);
    plot(theta_discrete', gamma_est, '-', 'Color', c_mat(1, :) , 'LineWidth', 3);
    xlabel('\theta = ln(\tau [s])','FontSize', labelFontSize)
    ylabel('\gamma [\Omega]', 'FontSize', labelFontSize);
    title(sprintf('Trip %d: DRT', s), 'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);
    hold on;

    % R0 값을 그림 내부에 표시 
    str_R0 = sprintf('$R_0 = %.1e\\ \\Omega$', R0_est_all(s));
    x_limits = xlim;
    y_limits = ylim;
    text_position_x = x_limits(1) + 0.05 * (x_limits(2) - x_limits(1));
    text_position_y = y_limits(2) - 0.05 * (y_limits(2) - y_limits(1));
    text(text_position_x, text_position_y, str_R0, 'FontSize', labelFontSize, 'Interpreter', 'latex');
    hold off;
end


%% Plot 3D DRT

% SOC_mid_all을 색상 매핑을 위해 정규화
soc_min = min(SOC_mid_all);
soc_max = max(SOC_mid_all);
soc_normalized = (SOC_mid_all - soc_min) / (soc_max - soc_min);

% 사용할 컬러맵 선택
colormap_choice = jet;  % 원하는 다른 컬러맵으로 변경 가능
num_colors = size(colormap_choice, 1);
colors = interp1(linspace(0, 1, num_colors), colormap_choice, soc_normalized);

figure;
hold on;

% 각 트립별로 gamma 추정값을 3D 선으로 플롯
for s = 1:num_trips-1
    x = SOC_mid_all(s) * ones(size(theta_discrete(:)));
    y = theta_discrete(:);
    z = gamma_est_all(s, :)';
    plot3(x, y, z, 'Color', colors(s, :), 'LineWidth', 1.5);
end

xlabel('SOC', 'FontSize', labelFontSize);
ylabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
zlabel('\gamma [\Omega]', 'FontSize', labelFontSize);
title('Gamma Estimates vs. \theta and SOC', 'FontSize', titleFontSize);
grid on;
zlim([0, 1.5]);
set(gca, 'FontSize', axisFontSize);
view(135, 30);  % 시각화 각도 조정
hold off;


colormap(colormap_choice);
c = colorbar;
c.Label.String = 'SOC';
c.Label.FontSize = labelFontSize;
c.Ticks = linspace(0, 1, 5);  
c.TickLabels = arrayfun(@(x) sprintf('%.3f', x), linspace(soc_min, soc_max, 5), 'UniformOutput', false);


%% save

save('gamma_est_all.mat', 'gamma_est_all', 'SOC_mid_all');

save('theta_discrete.mat' , 'theta_discrete' );
save('R0_est_all.mat', 'R0_est_all');
save('udds_data.mat', 'udds_data');














