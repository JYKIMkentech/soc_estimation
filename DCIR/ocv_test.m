clc;
clear;

% C/20 충전/방전 데이터 불러오기
load('C:\Users\USER\Desktop\Panasonic 18650PF Data\Panasonic 18650PF Data\25degC\C20 OCV and 1C discharge tests_start_of_tests\05-08-17_13.26 C20 OCV Test_C20_25dC.mat');
current = meas.Current; % 전류 데이터
voltage = meas.Voltage; % 전압 데이터
time = meas.Time; % 시간 데이터

% 전류 상태 파싱 (C, D, R)
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

% 용량과 SOC 계산
for j = 1:length(data)
    if length(data(j).t) > 1
        data(j).Q = abs(trapz(data(j).t, data(j).I)) / 3600; %[Ah]
        data(j).cumQ = abs(cumtrapz(data(j).t, data(j).I)) / 3600; %[Ah]
    else
        data(j).Q = 0;
        data(j).cumQ = zeros(size(data(j).t));
    end
end

% OCV 충전 단계를 식별
step_ocv_chg = find(strcmp({data.type}, 'C')); % 모든 충전 단계 찾기

% SOC 및 OCV 추출
SOC = [];
OCV = [];
for j = step_ocv_chg'
    if ~isempty(data(j).cumQ)
        soc = data(j).cumQ / data(j).Q;
        SOC = [SOC; soc];
        OCV = [OCV; data(j).V];
    end
end

data(4).soc = soc;

% SOC-OCV 구조체 생성
soc_ocv = [SOC OCV];


% 나중에 사용할 SOC-OCV 구조체 저장
%save('soc_ocv.mat', 'soc_ocv');


plot(time,current);
hold on
plot(time,voltage);
xlabel('time')
ylabel('voltage')
yyaxis right
ylabel('current')

