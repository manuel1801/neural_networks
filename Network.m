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
        function train(obj, X_train, Y_train, learning_rate, epochs)
            N = size(X_train, 2);
        
            for epoch = 1:epochs
        
                % Forward pass for the entire dataset
                output = X_train;
        
                for layer = obj.layers
                    output = layer{1}.forward(output);
                end
        
                % Calculate loss (MSE)
                loss(epoch) = mean(sum((output - Y_train).^2, 1));
        
                % Backward pass for the entire dataset
                gradient = output - Y_train;
                is_output_layer = true;
                for layer = fliplr(obj.layers)                    
                    gradient = layer{1}.backward(gradient, learning_rate,N,is_output_layer);
                    is_output_layer = false;
                end
                       
                % Display the loss for every 1000 epochs
                if mod(epoch, 1000) == 0
                    disp(['Epoch ', num2str(epoch), ' - Loss: ', num2str(loss(epoch))]);
                end               
            end

            plot(1:epochs,loss)
            xlabel('Epochs'); ylabel('Loss');
            title('Error');
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
