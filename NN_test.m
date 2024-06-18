clc; clear; close all;

rng('default')
addpath('functions\')

% Define the function to approximate (example: a function of two inputs)
% f = @(x, y) sin(x) + cos(y);
f = @(x, y) x.^2 + sin(5 * x) + y.^2;

% Generate training data with two inputs (x and y)
input_size = 2;
output_size = 1;

% X_train = 2 * pi * rand(input_size, N); % Random values between 0 and 2*pi

% Generate training data with two variables
X = 4 * (rand(2, 100) - 0.5);
X_train = X(:, 1:80);
X_test = X(:, 81:end);
Y_train = f(X_train(1, :), X_train(2, :));
Y_test = f(X_test(1, :), X_test(2, :));

% Create a neural network
net = Network();

% Add a fully connected layer with input size 2 and output size 20
net = net.addLayer(FullyConnectedLayer(input_size, 20, 'relu'));

% Add a fully connected layer with input size 10 and output size 10
net = net.addLayer(FullyConnectedLayer(20, 10, 'relu'));

% Add another fully connected layer with output size 1 (single output)
net = net.addLayer(FullyConnectedLayer(10, output_size, []));

N_train = length(X_train);

% Define hyperparameters and options
opts.learning_rate = 0.01;
opts.epochs = 1000;
opts.batch_size = 10;
opts.plot_loss = true;
opts.loss_function = 'mse'; % TODO

% Train the network
net.train(X_train, Y_train, X_test, Y_test, opts);

% Make predictions using the trained network
Y_pred = net.predict(X_test);

% Calculate the prediction error (MSE)
test_error = mean(sum((Y_pred - Y_test).^2, 1));

disp(['Test Error: ', num2str(test_error)]);

% Plot the results
[X1_mesh, X2_mesh] = meshgrid(-2:0.1:2, -2:0.1:2);
Z_ground_truth = f(X1_mesh, X2_mesh);

% Plot the ground truth using surf
figure
surf(X1_mesh, X2_mesh, Z_ground_truth, 'EdgeColor', 'none');
hold on

% Plot the training data
scatter3(X_train(1, :), X_train(2, :), Y_train, 'green', 'Marker', 'x');
hold on

% Plot the test data
scatter3(X_test(1, :), X_test(2, :), Y_pred, 'red', 'Marker', 'x', 'LineWidth', 2);
xlabel('x'); ylabel('y'); zlabel('f(x, y)');
legend({'Ground Truth', 'Training Data', 'Test Data'});
title('Ground Truth, Training Data, and Test Data');

