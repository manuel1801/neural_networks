classdef FullyConnectedLayer < handle
    properties
        input_size
        output_size
        weights
        biases
        input
        output
        Z
        activation_function
        activation_function_gradient
        is_output_layer
        % optimal_weights
        % optimal_biases
        v_W_prev
        v_b_prev
        gamma
    end
    
    methods
        % Constructor to initialize the layer with input and output sizes
        function obj = FullyConnectedLayer(input_size, output_size, activation_function)
            obj.input_size = input_size;
            obj.output_size = output_size;

            if isempty(activation_function)
                obj.activation_function = @(x) x;
                obj.activation_function_gradient = @(x) x;
            else
                obj.activation_function = str2func(activation_function);
                obj.activation_function_gradient = str2func([activation_function,'_gradient']);
            end

            % Initialize weights with small random values
            obj.weights = randn(output_size, input_size);
            obj.biases = randn(output_size, 1);

            % Initialize v for momentum
            obj.v_W_prev = zeros(output_size, input_size);
            obj.v_b_prev = zeros(output_size, 1);

            obj.gamma = 0.9;

        end
        
        % Forward pass through the layer
        function output = forward(obj, input)
            obj.input = input;
            obj.Z = obj.weights * input + obj.biases;   
            obj.output = obj.activation_function(obj.Z);
            output = obj.output;
        end

        
        % Backward pass through the layer
        function gradient = backward(obj, gradient, learning_rate, N, is_output_layer, use_momentum)
            
            % Calculate gradients for weights and biases  
            if is_output_layer
                dZ = gradient;
            else
                dZ = gradient.* obj.activation_function_gradient(obj.Z);
            end
            dW = 1 / N * dZ * obj.input';
            db = 1 / N * sum(dZ, 2);  

            % Calculate gradient to be passed to the previous layer
            gradient = obj.weights' * dZ;

            % Update weights and biases
            if use_momentum

                v_W = obj.gamma*obj.v_W_prev + learning_rate * dW;
                v_b = obj.gamma*obj.v_b_prev + learning_rate * db;
                
                obj.weights = obj.weights - v_W;
                obj.biases = obj.biases - v_b;

                obj.v_W_prev = v_W;
                obj.v_b_prev = v_b;
            else               
                obj.weights = obj.weights - learning_rate * dW;
                obj.biases = obj.biases - learning_rate * db;
            end
        end

        % function save_optimal_weights(obj)
        %     obj.optimal_weights = obj.weights;
        %     obj.optimal_biases = obj.biases;
        % end
        
    end
end
