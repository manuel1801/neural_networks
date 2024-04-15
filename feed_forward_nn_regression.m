clc; clear; close all;

rng('default')

addpath('functions/')

% Define the neural network architecture
% Togeter with the input layer we have N_hidden_layers+1 layers
% Define the number of neurons in each layer
layer_size(1) = 1;      % Input layer
layer_size(2) = 10;     % hidden layer 1
layer_size(3) = 10;     % hidden layer 2
layer_size(4) = 1;      % Output layer

N_hidden_layers = numel(layer_size)-1;

% Hyperparameters
learning_rate = 0.05;
epochs = 3000;
N_train = 500;
N_test = 200;
N_batch = 10;
noise_level =  0.05;
act_fun_type = 'tanh'; % 'relu', 'sigmoid' or 'tanh'

% Initialize weights and biases randomly
for i = 1:N_hidden_layers
    W{i} = 1 * (2*rand(layer_size(i+1), layer_size(i)) - 1);
    b{i} = 1 * (2*rand(layer_size(i+1), 1) - 1);
end

% Function to approximate
f = @(x) (1/8)*(63*x.^5 - 70*x.^3 + 15*x);

% Generate training data
U_train = 1 * (2*rand(layer_size(1), N_train) - 1); % random values between -1 and 1
U_test = 1 * (2*rand(layer_size(1), N_test) - 1);
for i = 1:N_train
    Y_train(:, i) = f(U_train(:, i));
end
for i = 1:N_test
    Y_test(:, i) = f(U_test(:, i));
end

% Add noise to the training data
Y_train = Y_train + noise_level * (2*rand(layer_size(end), N_train)-1);

test_err_min = inf;   % to track the minimum test error

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


        % Forward pass through all layers
        u{1} = u_batch;
        for l = 1:N_hidden_layers
            Z{l} = W{l} * u{l} + b{l};
            if l < N_hidden_layers
                u{l+1} = act_fun(Z{l}, act_fun_type);
            else
                u{l+1} = Z{l};
            end                  
        end
        Y_pred = u{l+1};
  
        % Calculate the training loss (MSE)
        Error = Y_pred - y_batch;
        loss_train_ = loss_train_ + mean(sum(Error.^2, 1));


        % Backpropagation through all the layers
        delta = Error;
        for l = N_hidden_layers:-1:1
                
            % for dW, db not neccessary cell array required
            dW{l} = 1 / N_batch * delta * u{l}';
            db{l} = 1 / N_batch * sum(delta, 2);
                            
            if l > 1
                delta = (W{l}'*delta).*act_fun_gradient(Z{l-1}, act_fun_type);
            end
           
            % Update the weights
            W{l} = W{l} - learning_rate * dW{l};
            b{l} = b{l} - learning_rate * db{l};
    
        end 

        % Forward pass with test data
        u{1} = U_test;
        for l = 1:N_hidden_layers
            Z{l} = W{l} * u{l} + b{l};
            if l < N_hidden_layers
                u{l+1} = act_fun(Z{l}, act_fun_type);
            else
                u{l+1} = Z{l};       
            end               
        end
        Y_test_pred = u{l+1};

        % Calculate the test loss (MSE)
        loss_test_ = loss_test_ + mean(sum((Y_test_pred - Y_test).^2, 1)); 

    end

    % Average the losses
    loss_train(epoch) = loss_train_/N_batch;
    loss_test(epoch) = loss_test_/N_batch;

    % Check if current test loss is less then minimum test loss
    if loss_test(epoch) <= test_err_min
        test_err_min = loss_test(epoch);  
        for l = 1:N_hidden_layers
            W_min{l} = W{l}; 
            b_min{l} = b{l};
        end
    end

    % Plot the loss curves und predicted function values
    if epoch == 1
        figure
        loss_train_plt = semilogy(epoch,loss_train(epoch),'o-'); hold on
        loss_test_plt = semilogy(epoch,loss_test(epoch) ,'o-'); hold on
        legend({'train', 'test'});
        xlabel('Epochs'); ylabel('Loss');

        figure
        plot(-1:0.01:1, f(-1:0.01:1)); hold on
        y_pred_test_plt = scatter(U_test, Y_test_pred, 'red', MarkerFaceColor='red'); hold on
        legend({'True function', 'Predicted'})

        drawnow;


    else
        loss_train_plt.XData = 1:epoch;
        loss_test_plt.XData =  1:epoch;
        loss_train_plt.YData = loss_train(1:epoch);
        loss_test_plt.YData =  loss_test(1:epoch);

        y_pred_test_plt.YData = Y_test_pred;
        drawnow;
    end   
   
    % Display the loss for every 10 epochs
    if mod(epoch, 10) == 0
        disp(['Epoch ', num2str(epoch),'/', num2str(epochs), ' - Loss train: ', num2str(loss_train(epoch)), ' - Loss test: ', num2str(loss_test(epoch))]);
    end
end


% Test the network on new input
u{1} = U_test;
for l = 1:N_hidden_layers
    Z{l} = W_min{l} * u{l} + b_min{l};
    if l < N_hidden_layers
        u{l+1} = act_fun(Z{l}, act_fun_type);
    else
        u{l+1} = Z{l};
    end           
end
Y_pred_test = u{l+1};

loss_test = mean(sum((Y_test_pred - Y_test).^2, 1));


% Plot predicted function values
y_pred_test_plt.YData = Y_test_pred;


% activation functions and their gradient
function y = act_fun(x, type)
    if strcmp(type,'relu')
        y = relu(x);
    elseif strcmp(type,'sigmoid')
        y = sigmoid(x);
    elseif strcmp(type,'tanh')
        y = tanh(x);   
    end
end

function y = act_fun_gradient(x, type)    
    if strcmp(type,'relu')
        y = relu_gradient(x);
    elseif strcmp(type,'sigmoid')        
        y = sigmoid_gradient(x);
    elseif strcmp(type,'tanh')
        y = tanh_gradient(x);     
    end
end
