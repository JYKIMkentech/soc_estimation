clc;clear;close all;


load('udds_data.mat');

I = udds_data(1).I;
t = udds_data(1).t;

epsilon_percent_span = 0.1;
sigma_percent = 0.01;

N = 51; % noise_vector 수 
epsilon_vector = linspace(-epsilon_percent_span/2, epsilon_percent_span/2, N); % From -5% to +5%
sigma = sigma_percent; % Standard deviation in percentage % 1%

P = zeros(N);
for i = 1:N
    probabilities = normpdf(epsilon_vector, epsilon_vector(i), sigma);
    P(i, :) = probabilities / sum(probabilities); % Normalize to sum to 1
end

%initalize
initial_state = randsample(1:N, 1); % Randomly select initial state
current_state = initial_state;

    
noisy_I = zeros(size(I));
for k = 1:length(I)
    epsilon = epsilon_vector(current_state);
    noisy_I(k) = I(k) * (1 + epsilon); % Apply the epsilon percentage

    % Transition to the next state
    current_state = randsample(1:N, 1, true, P(current_state, :));
end

figure(1)
plot(t,I,t,noisy_I);
xlabel('time')
ylabel ('current')


figure(2)
plot(t,I-noisy_I);
xlabel('time')
ylabel ( 'I - Noisy_I')