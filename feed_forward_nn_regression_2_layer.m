clc;  clear; close all;

% rng("default")

addpath('../functions/')

% Define the neural network architecture
n_inputs = 1;       % Number of input features
n_hidden_1 = 8;    % Number of units in the first hidden layer
n_hidden_2 = 6;    % Number of units in the second hidden layer
n_outputs = 1;      % Number of output units

% Initialize weights and biases randomly
W1 = 2*rand(n_hidden_1, n_inputs)-1;
b1 = 2*rand(n_hidden_1, 1)-1;
W2 = 2*rand(n_hidden_2, n_hidden_1)-1;
b2 = 2*rand(n_hidden_2, 1)-1;
W3 = 2*rand(n_outputs, n_hidden_2)-1;
b3 = 2*rand(n_outputs, 1)-1;


% Hyperparameters
epochs = 1000;
learning_rate = 0.05;
N_train = 500;
N_test = 200;
N_batch = 10;
noise_level = 0.05;

% Function to approximate
f = @(x) (1/8)*(63*x.^5 - 70*x.^3 + 15*x);

% Generate training data
U = 1 * (2*rand(n_inputs, N_train+N_test) - 1); % random values between -1 and 1
U_train = U(:, 1:N_train);
U_test = U(:, N_train+1:end);
for i = 1:N_train
    Y_train(:, i) = f(U_train(:, i));
end
for i = 1:N_test
    Y_test(:, i) = f(U_test(:, i));
end

% Standadize the data
[U_train, U_test_stand] = standardize_input(U_train,U_test);

% Add noise to the training data
Y_train = Y_train + noise_level * (2*rand(n_outputs, N_train)-1);
Y_test = Y_test + noise_level * (2*rand(n_outputs, N_test)-1);

test_err_min = inf;

% Training loop
for epoch = 1:epochs

    % Reshuffle training data
    idx = randperm(N_train);
    U_train = U_train(:,idx);
    Y_train = Y_train(:,idx);

    loss_train_ = 0; loss_test_ = 0;

    % Mini-Batch training
    for batch_start = 1:N_batch:N_train

        batch_end = min(batch_start+N_batch-1, N_train);
        u_batch = U_train(:,batch_start:batch_end);
        y_batch = Y_train(:,batch_start:batch_end);        

        % Backpropagation based training

        % Forward pass
        Z1 = W1 * u_batch + b1;
        A1 = tanh(Z1);
        Z2 = W2 * A1 + b2;
        A2 = tanh(Z2);
        Z3 = W3 * A2 + b3;
        Y_pred = Z3;
        
        % Calculate the loss (MSE)
        loss_train_ = loss_train_ + mean(sum((Y_pred - y_batch).^2, 1));
        
        % Backward pass (Gradient descent)
        % 3rd Layer
        delta_3 = (Y_pred - y_batch);
        dW3 = 1 / N_batch * delta_3 * A2';
        db3 = 1 / N_batch * sum(delta_3, 2);
        
        % 2nd Layer
        delta_2 = (W3'*delta_3).*tanh_gradient(Z2);
        dW2 = 1 / N_batch * delta_2 * A1';
        db2 = 1 / N_batch * sum(delta_2,2);

        % 1st Layer
        delta_1 = (W2'*delta_2).*tanh_gradient(Z1);
        dW1 = 1 / N_batch * delta_1 * u_batch';
        db1 = 1 / N_batch * sum(delta_1,2);

        % Update weights and biases
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;
        W3 = W3 - learning_rate * dW3;
        b3 = b3 - learning_rate * db3;

        % Forward pass with test data
        Z1 = W1 * U_test_stand + b1;
        A1 = tanh(Z1);
        Z2 = W2 * A1 + b2;
        A2 = tanh(Z2);       
        Z3 = W3 * A2 + b3;
        Y_test_pred = Z3;
        loss_test_ = loss_test_ + mean(sum((Y_test_pred - Y_test).^2, 1));

              
    end

    % Average the losses
    loss_train(epoch) = loss_train_/N_batch;
    loss_test(epoch) = loss_test_/N_batch;
  
    % Check if current test loss is less then minimum test loss
    if loss_test(epoch) <= test_err_min
        test_err_min = loss_test(epoch);        
        W1_min = W1; b1_min = b1; W2_min = W2; b2_min = b2; W3_min = W3; b3_min = b3;
    end

   
    % Plot the loss curves
    if epoch == 1
        figure
        loss_train_plt = semilogy(epoch,loss_train(epoch),'o-', MarkerSize=4); hold on
        loss_test_plt = semilogy(epoch,loss_test(epoch) ,'o-', MarkerSize=4); hold on
        legend({'train', 'test'});
        xlabel('Epochs'); ylabel('Loss');

        % Plot for current output using test data
        figure
        plot(-1:0.01:1, f(-1:0.01:1), 'blue'); hold on
        plot_y_pred_test = scatter(U_test, Y_test_pred, 'red', MarkerFaceColor='red'); hold on
        legend({'True function', 'BP'})
        title('Predicted Outputs Test data')

        drawnow;
    else
        loss_train_plt.XData = 1:epoch;
        loss_test_plt.XData =  1:epoch;
        loss_train_plt.YData = loss_train(1:epoch);
        loss_test_plt.YData =  loss_test(1:epoch);
       
        plot_y_pred_test.YData = Y_test_pred;
 
        drawnow;

    end

    % Display the loss for every 10 epochs
    if mod(epoch, 10) == 0
        disp(['Backpropagation: Epoch ', num2str(epoch),'/', num2str(epochs), ' - Loss train: ', num2str(loss_train(epoch)), ' - Loss test: ', num2str(loss_test(epoch))]);
    end

end


% Predict using test data and best weights and biases

% Test Backprop-based trained NN
Z1 = W1_min * U_test + b1_min;
A1 = tanh(Z1);
Z2 = W2_min * A1 + b2_min;
A2 = tanh(Z2);
Z3 = W3_min * A2 + b3_min;
Y_test_pred = Z3;

% Compute and display the test loss
loss_test = mean(sum((Y_test_pred - Y_test).^2, 1));

plot_y_pred_test.YData = Y_test_pred;

disp(['Loss test: ', num2str(loss_test)]);


