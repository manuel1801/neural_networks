function [u_train_stand, u_test_stand] = standardize_input(u_train,u_test)
%STANDARDIZE_INPUT Summary of this function goes here
%   Detailed explanation goes here

% u_train [n_inputs x N_train]
% u_test [n_inputs x N_test]

mu = mean(u_train,2);
sigma = std(u_train,0,2);
u_train_stand = (u_train - mu)./sigma;
u_test_stand = (u_test - mu)./sigma;
end



