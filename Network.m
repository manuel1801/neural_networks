classdef Network
    properties
        layers % Cell array to store layers
    end
    
    methods
        % Constructor to initialize the network with an empty cell array
        function obj = Network()
            obj.layers = {};
        end
        
        % Add a layer to the network
        function obj = addLayer(obj, layer)
            obj.layers{end + 1} = layer;
        end

        % Train the network using batch gradient descent
        function train(obj, X_train, Y_train, X_test, Y_test, opts)
            N_train = size(X_train, 2);

            if isfield(opts,'epochs')
                epochs = opts.epochs;
            else
                epochs = 1000;
                disp('Warning: Nr of epochs not specified, setting to 1000')
            end

            if isfield(opts, 'learning_rate')
                learning_rate = opts.learning_rate;
            else
                learning_rate = 0.01;
                disp('Warning: Learning_rate not specified, setting to 0.01')
            end

            if isfield(opts, 'batch_size')
                batch_size = opts.batch_size;
            else
                batch_size = 10;
                disp('Warning: Batch size not specified, setting to 10')
            end

            if isfield(opts, 'use_momentum')
                use_momentum = opts.use_momentum;
            else
                use_momentum = false;                
            end

            % test_err_min = inf;   % to track the minimum test error
                              
            for epoch = 1:epochs

                % Shuffle training data
                idx = randperm(N_train);
                X_train = X_train(:,idx);
                Y_train = Y_train(:,idx);               
                
                loss_train_ = 0; loss_test_ = 0;


                % Mini-Batch training
                for batch_start = 1:batch_size:N_train
            
                    batch_end = min(batch_start+batch_size-1, N_train);
                    x_batch = X_train(:,batch_start:batch_end);
                    y_batch = Y_train(:,batch_start:batch_end);

           
                    % Forward pass with training data
                    output = x_batch;
            
                    for layer = obj.layers
                        output = layer{1}.forward(output);
                    end
            
                    % Calculate the training loss (MSE)
                    loss_train_ = loss_train_ + mean(sum((output - y_batch).^2, 1));
            
                    % Backward pass with the traiaing data
                    delta = output - y_batch;
                    is_output_layer = true;
                    for layer = fliplr(obj.layers)                    
                        delta = layer{1}.backward(delta, learning_rate,N_train,is_output_layer, use_momentum);
                        is_output_layer = false;
                    end


                    % Forward pass with test data
                    output = X_test;
            
                    for layer = obj.layers
                        output = layer{1}.forward(output);
                    end

                    % Calculate the test loss (MSE)
                    loss_test_ = loss_test_ + mean(sum((output - Y_test).^2, 1));
                   
                end

                % Average the losses
                loss_train(epoch) = loss_train_/batch_size;
                loss_test(epoch) = loss_test_/batch_size;

                % % Check if current test loss is less then minimum test loss
                % if loss_test(epoch) <= test_err_min
                %     test_err_min = loss_test(epoch);            
                %     for layer = obj.layers
                %         layer{1}.save_optimal_weights();
                %     end
                % end

                % Plot the loss curves und predicted function values
                if epoch == 1
                    figure
                    loss_train_plt = semilogy(epoch,loss_train(epoch),'o-'); hold on
                    loss_test_plt = semilogy(epoch,loss_test(epoch) ,'o-'); hold on
                    legend({'train', 'test'});
                    xlabel('Epochs'); ylabel('Loss');           
                    drawnow;                        
                else
                    loss_train_plt.XData = 1:epoch;
                    loss_test_plt.XData =  1:epoch;
                    loss_train_plt.YData = loss_train(1:epoch);
                    loss_test_plt.YData =  loss_test(1:epoch);            
                    drawnow;
                end
                       
                % Display the loss for every 1000 epochs
                if mod(epoch, 10) == 0
                    disp(['Epoch ', num2str(epoch),'/', num2str(epochs), ' - Loss train: ', num2str(loss_train(epoch)), ' - Loss test: ', num2str(loss_test(epoch))]);
                end               
            end

        end

        function output = predict(obj, X_test)

            % Forward pass for the entire dataset
            output = X_test;
    
            for layer = obj.layers
                output = layer{1}.forward(output);
            end
                       
        end



    end
end
