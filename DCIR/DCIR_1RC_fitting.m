clc; clear; close all;

% 데이터 로드
%data = load('C:\Users\USER\Desktop\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\5 pulse disch\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');
data = load('G:\공유 드라이브\BSL_Data3\HPPC\03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat');
% 시간, 전압, 전류 데이터 추출

time = data.meas.Time;
voltage = data.meas.Voltage;
current = data.meas.Current;

data1.I = current;
data1.V = voltage;
data1.t = time;

% 전류 상태 구분
data1.type = char(zeros([length(data1.t), 1]));
data1.type(data1.I > 0) = 'C';
data1.type(data1.I == 0) = 'R';
data1.type(data1.I < 0) = 'D';

% step 구분
data1_length = length(data1.t);
data1.step = zeros(data1_length, 1);
m = 1;
data1.step(1) = m;
for j = 2:data1_length
    if data1.type(j) ~= data1.type(j-1)
        m = m + 1;
    end
    data1.step(j) = m;
end

vec_step = unique(data1.step);
num_step = length(vec_step);

data_line = struct('V', zeros(1, 1), 'I', zeros(1, 1), 't', zeros(1, 1), 'indx', zeros(1, 1), 'type', char('R'), ...
    'steptime', zeros(1, 1), 'T', zeros(1, 1), 'SOC', zeros(1, 1));
data = repmat(data_line, num_step, 1);

for i_step = 1:num_step
    range = find(data1.step == vec_step(i_step));
    data(i_step).V = data1.V(range);
    data(i_step).I = data1.I(range);
    data(i_step).t = data1.t(range);
    data(i_step).indx = range;
    data(i_step).type = data1.type(range(1));
    data(i_step).steptime = data1.t(range);
    data(i_step).T = zeros(size(range)); % 온도 데이터가 없으므로 0으로 설정
end

% 초기 SOC 설정 (1로 가정)
initial_SOC = 1;
capacity_Ah = 2.7742; % 배터리 용량 (Ah)

% Discharge step 구하기
step_chg = [];
step_dis = [];

for i = 1:length(data)
    % type 필드가 C인지 확인
    if strcmp(data(i).type, 'C')
        % C가 맞으면 idx 1 추가
        step_chg(end+1) = i;
    % type 필드가 D인지 확인
    elseif strcmp(data(i).type, 'D')
        % 맞으면 idx 1 추가
        step_dis(end+1) = i;
    end
end

%% R0, R1, C 추출 

% 평균 전류 구하기
for i = 1:length(data)
    data(i).avgI = mean(data(i).I);
end

% V 변화량 구하기
for i = 1 : length(data)
    if i == 1
       data(i).deltaV = zeros(size(data(i).V));
    else
       data(i).deltaV = data(i).V - data(i-1).V(end);
    end
end

% Resistance 구하기 
for i = 1 : length(data)
    if data(i).avgI == 0
        data(i).R = zeros(size(data(i).V));
    else 
        data(i).R = (data(i).deltaV / data(i).avgI) .* ones(size(data(i).V));
    end
end

% 시간 초기화
for i = 1 : length(data)
    initialTime = data(i).t(1); % 초기 시간 저장
    data(i).t = data(i).t - initialTime; % 초기 시간을 빼서 시간 초기화
end

for i = 1:length(step_dis)
    % 시간의 길이가 5초 이상인 스텝에 대해서만 r1s 값을 계산
    if length(data(step_dis(i)).t) >= 5
       data(step_dis(i)).R001s = data(step_dis(i)).R(1);
       if length(data(step_dis(i)).R) >= 11
           data(step_dis(i)).R1s = data(step_dis(i)).R(11);
       else
           data(step_dis(i)).R1s = data(step_dis(i)).R(end);
       end
       data(step_dis(i)).R0 = data(step_dis(i)).R001s;
       data(step_dis(i)).R1 = data(step_dis(i)).R1s - data(step_dis(i)).R001s;
    else
       data(step_dis(i)).R001s = NaN;
       data(step_dis(i)).R1s = NaN;
       data(step_dis(i)).R0 = NaN;
       data(step_dis(i)).R1 = NaN;
    end
end

%% 63.2% 값을 이용한 tau 및 C 계산

timeAt632 = zeros(1, length(step_dis));  % Initialize timeAt632 as a matrix

for i = 1:length(step_dis)
    plot(data(step_dis(i)).t, data(step_dis(i)).V);

    % 최소값과 최대값 계산
    minVoltage = min(data(step_dis(i)).V);
    maxVoltage = max(data(step_dis(i)).V);

    % 63.2% 값 계산
    targetVoltage = minVoltage + (1 - 0.632 ) * (maxVoltage - minVoltage);

    % 63.2%에 가장 가까운 값의 인덱스 찾기
    [~, idx] = min(abs(data(step_dis(i)).V - targetVoltage));

    % 해당 시간 찾기
    timeAt632(i) = data(step_dis(i)).t(idx);

    % data(step_dis(i)) 구조체에 timeAt632 필드를 추가하고 값 할당
    data(step_dis(i)).timeAt632 = timeAt632(i);

    % 해당 시간에 선 그리기
    line([timeAt632(i), timeAt632(i)], [minVoltage, maxVoltage], 'Color', 'red', 'LineStyle', '--');

    xlabel('Time');
    ylabel('Voltage (V)', 'fontsize', 12);
    title('Voltage - Time Graph');
end

% C값 구하기
for i = 1:length(step_dis)
    data(step_dis(i)).C = data(step_dis(i)).timeAt632 / (data(step_dis(i)).R1s - data(step_dis(i)).R001s);
end

% SOC 값을 정의된 패턴에 따라 생성
soc_values = [1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05];
steps_per_level = 5;

% SOC 배열 초기화
SOC = zeros(length(step_dis), 1);
current_index = 1;

for i = 1:length(soc_values)
    end_index = min(current_index + steps_per_level - 1, length(step_dis));
    SOC(current_index:end_index) = soc_values(i);
    current_index = end_index + 1;
end

% step_dis 배열을 사용하여 데이터에 SOC 값 할당
for i = 1:length(step_dis)
    data(step_dis(i)).SOC = SOC(i);
end

data(130).SOC = 0.05;

% 구조체 생성
optimized_params_struct_final_1RC = struct('R0', [], 'R1', [], 'C', [], 'SOC', [], 'Crate', [], 'm', []); % 'Crate' 필드 수정
% 초기 추정값 개수 설정
num_start_points = 10; % 원하는 시작점의 개수 설정

for i = 1:length(step_dis)
    deltaV_exp = data(step_dis(i)).deltaV;
    time_exp = data(step_dis(i)).t;
    avgI = data(step_dis(i)).avgI;  % 각 스텝의 평균 전류 가져오기
        
    % m 후보군 정의
    m_candidates = [2 / data(step_dis(i)).timeAt632, 1 / data(step_dis(i)).timeAt632, 1.05, 1.4];
    num_m = length(m_candidates);
    
    best_cost = Inf;
    best_params = [];
    best_m = NaN;
    
    for m_idx = 1:num_m
        m = m_candidates(m_idx);
        
        % 스텝의 시간 길이 확인
        step_duration = time_exp(end) - time_exp(1);

        if step_duration >= 0 
            % 최적화를 위한 초기 추정값 생성
            initial_guesses = repmat([data(step_dis(i)).R1, data(step_dis(i)).C], num_start_points, 1);

            % fmincon을 사용하여 최적화 수행
            options = optimoptions('fmincon', 'Display', 'none', 'MaxIterations', 100);
            problem = createOptimProblem('fmincon', 'objective', @(params) cost_function(params, time_exp, deltaV_exp, avgI, m, data(step_dis(i)).R0), ...
                'x0', initial_guesses, 'lb', [0, 0], 'ub', [], 'options', options);
            ms = MultiStart('Display', 'off');

            [opt_params, cost] = run(ms, problem, num_start_points); % 여러 시작점으로 실행

            % 가장 낮은 비용 함수를 가진 파라미터 선택
            if cost < best_cost
                best_cost = cost;
                best_params = opt_params;
                best_m = m;
            end
        end
    end
    
    if ~isnan(best_m)
        optimized_params_struct_final_1RC(i).R0 = data(step_dis(i)).R0; % R0 고정된 값 사용
        optimized_params_struct_final_1RC(i).R1 = best_params(1);
        optimized_params_struct_final_1RC(i).C = best_params(2);
        optimized_params_struct_final_1RC(i).SOC = mean(data(step_dis(i)).SOC); % 평균 SOC 값을 저장
        optimized_params_struct_final_1RC(i).Crate = avgI/data(step_dis(2)).avgI; % 평균 전류 저장
        optimized_params_struct_final_1RC(i).m = best_m; % 선택된 m 값 저장
    else
        optimized_params_struct_final_1RC(i).R0 = NaN;
        optimized_params_struct_final_1RC(i).R1 = NaN;
        optimized_params_struct_final_1RC(i).C = NaN;
        optimized_params_struct_final_1RC(i).SOC = NaN;
        optimized_params_struct_final_1RC(i).Crate = NaN;
        optimized_params_struct_final_1RC(i).m = NaN;
    end
end

%% Plotting with Subplots (Modified Section)

% Define the number of subplots per figure
plots_per_fig = 9; % 3 rows x 3 columns
num_figures = ceil(length(step_dis) / plots_per_fig);

% Initialize figure counter and subplot index
fig_counter = 1;
subplot_idx = 1;

% Create the first figure with specified resolution
figure(fig_counter);
set(fig_counter, 'Units', 'pixels', 'Position', [100, 100, 1200, 800]); % [left, bottom, width, height]
sgtitle('Comparison of Experimental Data and Model Results');

for i = 1:length(step_dis)
    % 스텝의 시간 길이 확인
    if (data(step_dis(i)).t(end) - data(step_dis(i)).t(1)) >= 0
        % Check if a new figure is needed
        if subplot_idx > plots_per_fig
            fig_counter = fig_counter + 1;
            figure(fig_counter);
            set(fig_counter, 'Units', 'pixels', 'Position', [100, 100, 1200, 800]); % [left, bottom, width, height]
            sgtitle('Comparison of Experimental Data and Model Results');
            subplot_idx = 1;
        end
        
        % Create subplot
        subplot(3, 3, subplot_idx);
        hold on;
        
        % Extract necessary data
        deltaV_exp = data(step_dis(i)).deltaV;
        time_exp = data(step_dis(i)).t;
        avgI = data(step_dis(i)).avgI;
        optimized_R0 = optimized_params_struct_final_1RC(i).R0;
        optimized_R1 = optimized_params_struct_final_1RC(i).R1;
        optimized_C = optimized_params_struct_final_1RC(i).C;
        m = optimized_params_struct_final_1RC(i).m;
        timeAt632 = data(step_dis(i)).timeAt632;
        
        % Generate model voltage
        voltage_model = model_func(time_exp, optimized_R0, optimized_R1, optimized_C, avgI);
        
        % Plot experimental data
        plot(time_exp, deltaV_exp, 'b-', 'LineWidth', 1.5, 'DisplayName', '실험 데이터');
        
        % Plot model data
        plot(time_exp, voltage_model, 'r--', 'LineWidth', 1.5, 'DisplayName', '모델 결과');
        
        % Plot 63.2% time vertical line
        plot([timeAt632, timeAt632], [min(deltaV_exp), max(deltaV_exp)], 'g--', 'LineWidth', 1.5, 'DisplayName', '63.2% 시간');
        
        % Add SOC, C-rate, and m text
        soc_text = sprintf('SOC: %.2f%%', optimized_params_struct_final_1RC(i).SOC * 100);
        crate_text = sprintf('C-rate: %.2f', optimized_params_struct_final_1RC(i).Crate);
        m_text = sprintf('m: %.2f', optimized_params_struct_final_1RC(i).m);
        text(time_exp(1) + 0.05*(time_exp(end)-time_exp(1)), ...
             max(deltaV_exp)*0.9, ...
             {soc_text, crate_text, m_text}, 'FontSize', 8, 'Color', 'k', 'FontWeight', 'bold');
        
        % Labels and title
        xlabel('시간 (sec)', 'FontSize', 8);
        ylabel('전압 (V)', 'FontSize', 8);
        title(sprintf('Discharge Step %d', i), 'FontSize', 10);
        legend('Location', 'best', 'FontSize', 6);
        grid on;
        
        hold off;
        
        % Increment subplot index
        subplot_idx = subplot_idx + 1;
    end
end
%% R0,R1,C Plot

%% R0, R1, C (SOC, C-rate) 3D and Contour Plots

% Extract SOC, C-rate, R0, R1, C from optimized_params_struct
SOC = [optimized_params_struct_final_1RC.SOC]';
Crate = [optimized_params_struct_final_1RC.Crate]';
R0 = [optimized_params_struct_final_1RC.R0]';
R1 = [optimized_params_struct_final_1RC.R1]';
C = [optimized_params_struct_final_1RC.C]';

% Remove entries with NaN values
valid_idx = ~isnan(SOC) & ~isnan(Crate) & ~isnan(R0) & ~isnan(R1) & ~isnan(C);
SOC = SOC(valid_idx);
Crate = Crate(valid_idx);
R0 = R0(valid_idx);
R1 = R1(valid_idx);
C = C(valid_idx);

% Define grid for interpolation
num_grid = 100; % Number of grid points in each dimension
SOC_grid = linspace(min(SOC), max(SOC), num_grid);
Crate_grid = linspace(min(Crate), max(Crate), num_grid);
[SG, CG] = meshgrid(SOC_grid, Crate_grid);

% Interpolate R0, R1, C over the grid
R0_grid = griddata(SOC, Crate, R0, SG, CG, 'cubic');
R1_grid = griddata(SOC, Crate, R1, SG, CG, 'cubic');
C_grid = griddata(SOC, Crate, C, SG, CG, 'cubic');



% Create plots for R0
create_plots(SG, CG, R0_grid, 'R0 ');

% Create plots for R1
create_plots(SG, CG, R1_grid, 'R1 ');

% Create plots for C
create_plots(SG, CG, C_grid, 'C ');

%% Optional: Save the plots
% You can uncomment the following lines to save the figures automatically
% saveas(gcf, 'R0_plot.png');
% saveas(gcf, 'R1_plot.png');
% saveas(gcf, 'C_plot.png');



%% save
save('optimized_params_struct_final_1RC.mat', 'optimized_params_struct_final_1RC');


%% 함수

function cost = cost_function(params, time, deltaV, I, m, R0)
    R1 = params(1);
    C = params(2);
    
    % 모델 함수를 사용하여 예측 전압 계산
    voltage_model = model_func(time, R0, R1, C, I);
    
    % 오차 계산
    error = deltaV - voltage_model;
    
    % 시간에 따라 가중치 함수 적용
    % m 값을 사용하여 가중치 함수를 조절
    time_weights = exp(-m * time); 
    
    % 가중 평균 제곱근 오차(RMS 오차) 계산
    weighted_error = error .* time_weights;
    cost = sqrt(mean(weighted_error.^2));
end

function voltage = model_func(time, R0, R1, C, I)
    voltage = I * (R0 + R1 * (1 - exp(-time / (R1 * C))));
end

% Define a function to create plots
function create_plots(SG, CG, Param_grid, Param_name)
    figure();
    % 3D Surface Plot
   
    surf(SG, CG, Param_grid, 'EdgeColor', 'none');
    xlabel('SOC');
    ylabel('C-rate');
    zlabel(Param_name);
    title([Param_name ' vs SOC and C-rate']);
    colorbar;
    view(45, 30); % Adjust the viewing angle for better visualization
    grid on;

    % Contour Plot
    figure();
    contourf(SG, CG, Param_grid, 20, 'LineColor', 'none');
    xlabel('SOC');
    ylabel('C-rate');
    title([Param_name ' Contour Plot']);
    colorbar;
    grid on;
end
