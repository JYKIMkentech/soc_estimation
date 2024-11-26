function [gamma_est, R0_est, V_est, theta_discrete, W_aug, y, OCV] = DRT_estimation_aug(t, ik, V_sd, lambda_hat, n, dt, dur, SOC, soc_values, ocv_values)
    % DRT_estimation_aug estimates the gamma function and voltage using DRT with augmented internal resistance.
    %
    % Inputs:
    %   t           - Time vector
    %   ik          - Current vector
    %   V_sd        - Measured voltage vector
    %   lambda_hat  - Regularization parameter
    %   n           - Number of RC elements
    %   dt          - Sampling time vector
    %   dur         - Duration (tau_max)
    %   SOC         - State of Charge vector
    %   soc_values  - SOC values from SOC-OCV data
    %   ocv_values  - Corresponding OCV values from SOC-OCV data
    %
    % Outputs:
    %   gamma_est       - Estimated gamma vector
    %   R0_est          - Estimated internal resistance
    %   V_est           - Estimated voltage vector
    %   theta_discrete  - Discrete theta values
    %   tau_discrete    - Discrete tau values
    %   W               - Matrix used in estimation


    % Calculate OCV using SOC and soc_ocv data
    OCV = interp1(soc_values, ocv_values, SOC, 'linear', 'extrap');

    % Define theta_discrete and tau_discrete based on dur and n
    tau_min = 0.1;  % Minimum tau value in seconds
    tau_max = dur;   % Maximum tau value in seconds
    theta_min = log(tau_min);
    theta_max = log(tau_max);
    theta_discrete = linspace(theta_min, theta_max, n)';
    delta_theta = theta_discrete(2) - theta_discrete(1);
    tau_discrete = exp(theta_discrete);

    % Set up the W matrix
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                W(k_idx, i) = W(k_idx-1, i) * exp(-dt(k_idx) / tau_discrete(i)) + ...
                              ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            end
        end
    end

    W_aug = [W, ik(:)];  % Append ik as the last column

    % Adjust y (measured voltage)
    y = V_sd - OCV;
    y = y(:);

    % Regularization matrix L (first-order difference)
    L = zeros(n-1, n);
    for i = 1:n-1
        L(i, i) = -1;
        L(i, i+1) = 1;
    end

    L_aug = [L, zeros(n-1, 1)];  % No regularization on R0_est

    % Set up the quadratic programming problem
    H = 2 * (W_aug' * W_aug + lambda_hat * (L_aug' * L_aug));
    f = -2 * W_aug' * y;

    % Inequality constraints: params >= 0
    A_ineq = -eye(n+1);
    b_ineq = zeros(n+1, 1);

    % Solve the quadratic programming problem
    options = optimoptions('quadprog', 'Display', 'off');
    params = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);

    gamma_est = params(1:end-1);
    R0_est = params(end);

    % Compute the estimated voltage
    V_est = OCV + W_aug * params;
end
