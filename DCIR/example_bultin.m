clc;clear;close all;

% 배터리 SOC 추정을 위한 EKF 예제 (수정)

% 배터리 모델 파라미터
C = 2.0;              % 배터리 용량 (Ah)
R = 0.05;             % 내부 저항 (Ohm)
dt = 1;               % 시간 간격 (초)
num_steps = 100;      % 시뮬레이션 스텝 수

% OCV(SOC) 함수 정의
OCV = @(SOC) 3.0 + 0.5*SOC + 0.2*SOC.^2;

% 상태 전이 함수
stateTransitionFcn = @(x, u) x - (u.I * dt) / C;

% 측정 함수
measurementFcn = @(x, u) OCV(x) - u.I * R;

% 초기 상태 및 공분산 설정
initialSOC = 1;                         % 초기 SOC
initialCovariance = 0.01;                 % 초기 상태 공분산
measurementNoiseCovariance = 0.02;        % 측정 노이즈 공분산
processNoiseCovariance = 1e-5;            % 프로세스 노이즈 공분산

% EKF 객체 생성
ekf = extendedKalmanFilter(stateTransitionFcn, measurementFcn, initialSOC, ...
    'StateCovariance', initialCovariance, ...
    'ProcessNoise', processNoiseCovariance, ...
    'MeasurementNoise', measurementNoiseCovariance);

% 시뮬레이션용 변수 초기화
trueSOC = initialSOC;
estimatedSOC = zeros(num_steps, 1);
measurements = zeros(num_steps, 1);
trueSOC_history = zeros(num_steps, 1);

rng(0); % 재현성을 위해 난수 초기화

for k = 1:num_steps
    % 전류 프로파일 설정
    if k < 50
        I = 1.0;  % 충전 (1 A)
    else
        I = -1.0; % 방전 (-1 A)
    end
    
    % 실제 SOC 업데이트
    trueSOC = trueSOC - (I * dt) / C;
    trueSOC = max(min(trueSOC, 1.0), 0.0);
    trueSOC_history(k) = trueSOC;
    
    % 실제 전압 측정 (OCV - I*R) + 노이즈 추가
    V_true = OCV(trueSOC) - I * R;
    V_meas = V_true + sqrt(measurementNoiseCovariance) * randn();
    measurements(k) = V_meas;
    
    % 추가 파라미터 구조체
    u.I = I;
    
    % EKF 예측 단계
    predict(ekf, u);
    
    % EKF 보정 단계 (추가 파라미터 u를 바로 전달)
    correct(ekf, V_meas, u);
    
    % 추정된 SOC 저장
    estimatedSOC(k) = ekf.State;
end

% 결과 시각화
time = (1:num_steps) * dt / 3600; % 시간을 시간 단위(시간)로 변환

figure;
plot(time, trueSOC_history, 'g-', 'LineWidth', 2);
hold on;
plot(time, estimatedSOC, 'b--', 'LineWidth', 2);
xlabel('시간 (시간)');
ylabel('SOC');
legend('실제 SOC', '추정된 SOC (EKF)', 'Location', 'Best');
title('배터리 SOC 추정');
grid on;


