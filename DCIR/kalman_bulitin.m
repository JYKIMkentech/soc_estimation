clear; clc; close all;

%% 시뮬레이션 파라미터
Ts = 1;              % 샘플링 시간 [초]
simTime = 3600;      % 시뮬레이션 시간 [초]
time = (0:Ts:simTime)'; 

% 가상의 인가 전류 (예: 주기적으로 변화하는 부하)
I_load = 2*sin(2*pi*(1/600)*time) + 1; % A, 간단한 예제

%% 배터리 파라미터 (NMC 예시)
Q_cap = 2.3 * 3600; % 2.3Ah -> coulomb 단위
R0 = 0.01;           % [Ohms]
R1 = 0.015; C1 = 3000;   % RC branch 1
R2 = 0.02;  C2 = 5000;   % RC branch 2

% OCV-SOC Table (가상의 NMC OCV 곡선)
SOC_vec = 0:0.01:1;
OCV_vec = 3.0 + 1.2*SOC_vec - 0.1*sin(2*pi*SOC_vec); % 예시로 임의 정의

% 초기조건
SOC0 = 0.8;   % 초기 SOC
Vrc10 = 0;    % RC branch1 초기전압
Vrc20 = 0;    % RC branch2 초기전압

x_true = [SOC0; Vrc10; Vrc20];

%% 상태 공간 모델 유도
% 연속 시간 상태 방정식:
% dSOC/dt = (-I_load / Q_cap)
% dVrc1/dt = (-1/(R1*C1))*Vrc1 + (I_load/C1)
% dVrc2/dt = (-1/(R2*C2))*Vrc2 + (I_load/C2)

% 이산화 (단순 Euler 방법)
A_c = [0, 0, 0;
       0, -1/(R1*C1), 0;
       0, 0, -1/(R2*C2)];
B_c = [-1/Q_cap; 1/C1; 1/C2];

A = eye(3) + A_c*Ts;
B = B_c*Ts;

% 출력 방정식:
% Vt = OCV(SOC) + Vrc1 + Vrc2 + R0*I

% 선형화 지점 (초기 SOC 근처)
SOC_lin = SOC0;
[OCV_lin, dOCV_dSOC_lin] = OCV_lookup(SOC_lin, SOC_vec, OCV_vec);

% 선형화된 출력 방정식:
% y = dOCV_dSOC_lin*SOC + Vrc1 + Vrc2 + (OCV_lin - dOCV_dSOC_lin*SOC_lin) + R0*I

C = [dOCV_dSOC_lin, 1, 1];
D = R0;

% 프로세스 및 측정 잡음 공분산
% 상태 공간 모델의 입력은 [I; w], 여기서 w는 프로세스 잡음
Q_process = 1e-7;        % 프로세스 잡음 공분산 (스칼라)
Q = [0, 0; 0, Q_process]; % 2x2 행렬: 첫 번째 입력은 제어 입력이므로 0, 두 번째 입력은 프로세스 잡음
R = 1e-3;        % 측정 잡음 공분산 (스칼라)

% 상태 공간 모델 정의 (두 개의 입력: 제어 입력 I와 프로세스 잡음)
sys = ss(A, [B zeros(3,1)], C, [D 0], Ts);

% 칼만 필터 설계
[kalmf, L, P, M] = kalman(sys, Q, R);

% 초기 추정값
x_hat = [0.75; 0; 0]; % SOC를 약간 틀리게 가정
P_init = P;

% 시뮬레이션 실행
SOC_est = zeros(length(time),1);
SOC_true = zeros(length(time),1);
V_meas = zeros(length(time),1);
V_est = zeros(length(time),1);

x_est = x_hat;
for k = 1:length(time)
    % 실제 시스템 업데이트
    SOC_k = x_true(1);
    Vrc1_k = x_true(2);
    Vrc2_k = x_true(3);
    I_k = I_load(k);

    % 실제 OCV 계산
    [OCV_k, ~] = OCV_lookup(SOC_k, SOC_vec, OCV_vec);
    Vt_k = OCV_k + Vrc1_k + Vrc2_k + R0*I_k;

    % 실제 다음 상태 업데이트
    x_true = A*x_true + B*I_k;
    % SOC 클리핑
    x_true(1) = max(min(x_true(1),1),0);

    % 측정값: Vt + 측정 잡음 추가 (가우시안 잡음)
    V_meas_k = Vt_k + sqrt(R)*randn;

    % 칼만 필터 업데이트
    % kalmf는 상태 추정기 시스템이므로, 필터에 입력을 제공해야 함
    % 필터의 입력은 [u; y] 형태
    u_kf = [I_k; V_meas_k];

    % 상태 추정기에 입력 제공 및 상태 추정
    y_filt = kalmf.A * x_est + kalmf.B * u_kf(2); % 입력 중 측정값 y
    x_est = kalmf.A * x_est + kalmf.B(1) * I_k + kalmf.B(2) * V_meas_k + L * (V_meas_k - (C * x_est + D * I_k));

    % 결과 저장
    SOC_est(k) = x_est(1);
    SOC_true(k) = SOC_k;
    V_meas(k) = V_meas_k;
    V_est(k) = C * x_est + D * I_k;
end

% 결과 플롯
figure;
subplot(2,1,1);
plot(time, SOC_true,'b','LineWidth',1.5); hold on; grid on;
plot(time, SOC_est,'r--','LineWidth',1.5);
xlabel('Time [s]'); ylabel('SOC');
legend('True SOC','Estimated SOC');
title('SOC Estimation via Kalman Filter (2RC ECM)');

subplot(2,1,2);
plot(time, V_meas,'k','LineWidth',1.0); hold on; grid on;
plot(time, V_est,'m--','LineWidth',1.5);
xlabel('Time [s]'); ylabel('Voltage [V]');
legend('Measured','Estimated');
title('Terminal Voltage Comparison');

%% OCV Lookup 함수
function [ocv, docv_dsoc] = OCV_lookup(soc, soc_vec, ocv_vec)
    % SOC 범위 제한
    soc = max(min(soc,1),0);

    % 선형 보간
    ocv = interp1(soc_vec, ocv_vec, soc, 'linear', 'extrap');

    % 미분 근사 (중앙 차분)
    dsoc = 1e-5;
    ocv_p = interp1(soc_vec, ocv_vec, min(soc+dsoc,1), 'linear', 'extrap');
    ocv_m = interp1(soc_vec, ocv_vec, max(soc-dsoc,0), 'linear', 'extrap');
    docv_dsoc = (ocv_p - ocv_m)/(2*dsoc);
end

