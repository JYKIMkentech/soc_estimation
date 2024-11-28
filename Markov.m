clc; clear; close all;

% Load the dataset
load('udds_data.mat');

I = udds_data(1).I;
t = udds_data(1).t;


%% epsilon space
% Define noise parameters
epsilon_percent_span = 0.1; % Span of epsilon in percentage (10%)
sigma_percent = 0.001;      % Standard deviation in percentage (0.1%)

N = 51; % Number of states
epsilon_vector = linspace(-epsilon_percent_span/2, epsilon_percent_span/2, N); % From -5% to +5%
sigma = sigma_percent; % Standard deviation in percentage

% Initialize transition probability matrix P
P = zeros(N);
for i = 1:N
    probabilities = normpdf(epsilon_vector, epsilon_vector(i), sigma);
    P(i, :) = probabilities / sum(probabilities); % Normalize to sum to 1
end

% Initialize state tracking
initial_state = randsample(1:N, 1); % Randomly select initial state
current_state = initial_state;

%% epsilon to current space

noisy_I = zeros(size(I));
states = zeros(size(I)); % Vector to store states
epsilon = zeros(size(I));
% Generate noisy current and track states
for k = 1:length(I)
    epsilon(k) = epsilon_vector(current_state);
    noisy_I(k) = I(k) * (1 + epsilon(k)); % Apply the epsilon percentage
    
    states(k) = current_state; % Store the current state
    
    % Transition to the next state based on probabilities
    current_state = randsample(1:N, 1, true, P(current_state, :));
end

% Plot Original and Noisy Current
figure(1)
plot(t, I, 'b', 'DisplayName', 'Original Current');
hold on;
plot(t, noisy_I, 'r', 'DisplayName', 'Noisy Current');
xlabel('Time');
ylabel('Current');
title('Original vs Noisy Current');
legend;
grid on;
hold off;

% Plot the Difference Between Original and Noisy Current
figure(2)
plot(t, I - noisy_I, 'k');
xlabel('Time');
ylabel('I - Noisy\_I');
title('Difference Between Original and Noisy Current');
grid on;

% Plot the State Transitions
figure(3)
plot(t, states, 'm');
xlabel('Time');
ylabel('State');
title('State Transitions Over Time');
ylim([1 N]); % Set y-axis limits to match state range
yticks(1:5:N); % Adjust y-ticks for better readability
grid on;
