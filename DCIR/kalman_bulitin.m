% 비선형 상태 방정식
stateTransitionFcn = @(x, dt) [
    x(1) + dt * x(3) * cos(x(4));  % x 위치
    x(2) + dt * x(3) * sin(x(4));  % y 위치
    x(3);                          % 속도
    x(4)                           % 방향
];

% 비선형 관측 방정식
measurementFcn = @(x) [
    sqrt(x(1)^2 + x(2)^2);  % 거리
    atan2(x(2), x(1))       % 각도
];

% 초기 상태 및 공분산 설정
initialState = [1; 0; 1; pi/4];   % [x, y, 속도, 방향]
initialCovariance = eye(4);       % 초기 상태 공분산

% 프로세스 잡음 공분산 및 측정 잡음 공분산
processNoise = diag([0.1, 0.1, 0.1, 0.1]);  % Q
measurementNoise = diag([0.1, 0.01]);       % R

% EKF 객체 생성
ekf = trackingEKF('StateTransitionFcn', stateTransitionFcn, ...
                  'MeasurementFcn', measurementFcn, ...
                  'State', initialState, ...
                  'StateCovariance', initialCovariance, ...
                  'ProcessNoise', processNoise, ...
                  'MeasurementNoise', measurementNoise);

% 시뮬레이션 설정
dt = 0.1;  % 시간 간격
numSteps = 50;

% 실제 궤적 및 측정값 생성
trueStates = zeros(4, numSteps);
measurements = zeros(2, numSteps);
trueState = initialState;

for k = 1:numSteps
    % 실제 상태 갱신
    trueState = stateTransitionFcn(trueState, dt) + mvnrnd(zeros(4,1), processNoise)';
    trueStates(:, k) = trueState;
    
    % 측정값 생성
    measurement = measurementFcn(trueState) + mvnrnd(zeros(2,1), measurementNoise)';
    measurements(:, k) = measurement;
    
    % EKF 업데이트
    predict(ekf, dt);  % 예측 단계
    correct(ekf, measurement);  % 업데이트 단계
end

% 결과 시각화
figure;
plot(trueStates(1, :), trueStates(2, :), 'g-', 'LineWidth', 2); hold on; % 실제 궤적
scatter(measurements(1, :) .* cos(measurements(2, :)), ...
        measurements(1, :) .* sin(measurements(2, :)), ...
        20, 'r', 'filled'); % 측정값
plot(ekf.State(1), ekf.State(2), 'b-', 'LineWidth', 2); % EKF 추정
legend('True Trajectory', 'Measurements', 'EKF Estimate');
xlabel('X'); ylabel('Y');
title('Extended Kalman Filter Example');
grid on;
