clear; clc; close all;

%% 0. 폰트 크기 및 색상 설정
% Font size settings
axisFontSize = 14;      % 축 숫자 폰트 크기
titleFontSize = 16;     % 제목 폰트 크기
legendFontSize = 12;    % 범례 폰트 크기
labelFontSize = 14;     % 축 라벨 폰트 크기

% 색상 정의 (RGB)
color_voltage_meas = [0.3, 0.3, 0.3];       % 다크 그레이(측정 전압)
color_voltage_est  = [0.494, 0.184, 0.556]; % 퍼플 계열(추정 전압)
color_current      = [0, 0.45, 0.74];       % 딥 블루(전류)
color_gamma        = [0.4, 0.2, 0.6];       % 보라 계열(gamma plot)

%% 1. 데이터 로드
load('udds_data.mat');   % 'udds_data' 구조체
load('soc_ocv.mat', 'soc_ocv');
soc_values = soc_ocv(:, 1);  % SOC
ocv_values = soc_ocv(:, 2);  % OCV

%% 2. Parameter 설정
n = 201;
dur = 1370;       % [sec]
SOC_begin = 0.9907;
Q_batt = 2.7742;  % [Ah]

%% 3. 각 trip에 대한 DRT 추정
num_trips = length(udds_data);

gamma_est_all = zeros(num_trips, n);  
R0_est_all = zeros(num_trips, 1);
V_est_all = cell(num_trips, 1); 
SOC_all = cell(num_trips, 1);   
SOC_mid_all = zeros(num_trips,1);

for s = 1:num_trips-1
    fprintf('Processing Trip %d/%d...\n', s, num_trips);
    
    Time_duration = udds_data(s).Time_duration;
    t = udds_data(s).t;   
    I = udds_data(s).I;   
    V = udds_data(s).V;   
    lambda_hat = 3.79e-10;      
    dt = [t(1); diff(t)];  
    SOC = SOC_begin + cumtrapz(t, I) / (Q_batt * 3600); 
    SOC_all{s} = SOC;  
    SOC_mid_all(s) = mean(SOC);

    [gamma_est, R0_est, V_est , theta_discrete , W, ~, ~] = DRT_estimation_aug(t, I, V, lambda_hat, n, dt, dur, SOC, soc_values, ocv_values);
    
    gamma_est_all(s, :) = gamma_est';
    R0_est_all(s) = R0_est;
    V_est_all{s} = V_est;  
    
    SOC_begin = SOC(end);

    %% 기존: 각 Trip에 대해 Subplot으로 그림.
    % 요청: 여기서는 기존 코드 그대로 두되, 추가로 Trip1,8,16에 대해 별도의 figure 생성.
    figure('Name', ['Trip ', num2str(s)], 'NumberTitle', 'off');
    set(gcf, 'Position', [150, 150, 1200, 800]);

    % 첫 번째 subplot: Voltage, Estimated Voltage, Current
    subplot(2,1,1);
    yyaxis left
    plot(t, V, 'Color', color_voltage_meas, 'LineWidth', 3, 'DisplayName', 'Measured Voltage');
    hold on;
    plot(t, V_est, '--', 'Color', color_voltage_est, 'LineWidth', 3, 'DisplayName', 'Estimated Voltage');
    ylabel('Voltage [V]', 'FontSize', labelFontSize, 'Color', color_voltage_meas);
    set(gca, 'YColor', color_voltage_meas);

    yyaxis right
    plot(t, I, '-', 'Color', color_current, 'LineWidth', 3, 'DisplayName', 'Current');
    ylabel('Current [A]', 'FontSize', labelFontSize, 'Color', color_current);
    set(gca, 'YColor', color_current);

    xlabel('Time [s]', 'FontSize', labelFontSize);
    title(sprintf('Trip %d: Voltage and Current', s), 'FontSize', titleFontSize);
    legend('FontSize', legendFontSize);
    set(gca, 'FontSize', axisFontSize);
    hold off;

    % 두 번째 subplot: DRT (theta vs gamma)
    subplot(2,1,2);
    plot(theta_discrete', gamma_est, '-', 'Color', color_gamma , 'LineWidth', 3);
    xlabel('\theta = ln(\tau [s])','FontSize', labelFontSize)
    ylabel('\gamma [\Omega]', 'FontSize', labelFontSize);
    title(sprintf('Trip %d: DRT', s), 'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);
    hold on;

    str_R0 = sprintf('$R_0 = %.1e\\ \\Omega$', R0_est_all(s));
    x_limits = xlim;
    y_limits = ylim;
    text_position_x = x_limits(1) + 0.05 * (x_limits(2) - x_limits(1));
    text_position_y = y_limits(2) - 0.05 * (y_limits(2) - y_limits(1));
    text(text_position_x, text_position_y, str_R0, 'FontSize', labelFontSize, 'Interpreter', 'latex');
    hold off;
end

%% 3D DRT Plot
soc_min = min(SOC_mid_all);
soc_max = max(SOC_mid_all);
soc_normalized = (SOC_mid_all - soc_min) / (soc_max - soc_min);

colormap_choice = jet;
num_colors = size(colormap_choice, 1);
colors = interp1(linspace(0, 1, num_colors), colormap_choice, soc_normalized);

figure;
hold on;

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
view(135, 30);
hold off;

colormap(colormap_choice);
c = colorbar;
c.Label.String = 'SOC';
c.Label.FontSize = labelFontSize;
c.Ticks = linspace(0, 1, 5);  
c.TickLabels = arrayfun(@(x) sprintf('%.3f', x), linspace(soc_min, soc_max, 5), 'UniformOutput', false);

%% trip1, trip8, trip16에 대해 전압/전류 및 DRT 그래프 독립적 Figure 생성
special_trips = [1, 8, 16];
for s = special_trips
    if s <= num_trips - 1
        t = udds_data(s).t;
        I = udds_data(s).I;
        V = udds_data(s).V;
        V_est = V_est_all{s};
        gamma_est = gamma_est_all(s, :);
        R0_est = R0_est_all(s);
        
        % Voltage and Current Graph (No subplot)
        figure('Name', ['Trip ', num2str(s), ' Voltage & Current'], 'NumberTitle', 'off');
        yyaxis left
        plot(t, V, 'Color', color_voltage_meas, 'LineWidth', 3, 'DisplayName', 'Measured Voltage');
        hold on;
        plot(t, V_est, '--', 'Color', color_voltage_est, 'LineWidth', 3, 'DisplayName', 'Estimated Voltage');
        ylabel('Voltage [V]', 'FontSize', labelFontSize, 'Color', color_voltage_meas);
        set(gca, 'YColor', color_voltage_meas);

        yyaxis right
        plot(t, I, '-', 'Color', color_current, 'LineWidth', 3, 'DisplayName', 'Current');
        ylabel('Current [A]', 'FontSize', labelFontSize, 'Color', color_current);
        set(gca, 'YColor', color_current);
        xlabel('Time [s]', 'FontSize', labelFontSize);
        title(sprintf('Trip %d: Voltage and Current', s), 'FontSize', titleFontSize);
        legend('FontSize', legendFontSize);
        set(gca, 'FontSize', axisFontSize);
        hold off;

        % DRT Graph (No subplot)
        figure('Name', ['Trip ', num2str(s), ' DRT'], 'NumberTitle', 'off');
        plot(theta_discrete', gamma_est, '-', 'Color', color_gamma , 'LineWidth', 3);
        xlabel('\theta = ln(\tau [s])','FontSize', labelFontSize)
        ylabel('\gamma [\Omega]', 'FontSize', labelFontSize);
        title(sprintf('Trip %d: DRT', s), 'FontSize', titleFontSize);
        set(gca, 'FontSize', axisFontSize);
        hold on;
        str_R0 = sprintf('$R_0 = %.1e\\ \\Omega$', R0_est);
        x_limits = xlim;
        y_limits = ylim;
        text_position_x = x_limits(1) + 0.05 * (x_limits(2) - x_limits(1));
        text_position_y = y_limits(2) - 0.05 * (y_limits(2) - y_limits(1));
        text(text_position_x, text_position_y, str_R0, 'FontSize', labelFontSize, 'Interpreter', 'latex');
        hold off;
    else
        warning(['Trip ' num2str(s) ' does not exist.']);
    end
end

%% save
save('gamma_est_all.mat', 'gamma_est_all', 'SOC_mid_all');
save('theta_discrete.mat' , 'theta_discrete' );
save('R0_est_all.mat', 'R0_est_all');
save('udds_data.mat', 'udds_data');

