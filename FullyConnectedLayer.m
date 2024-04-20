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

        end
        
        % Forward pass through the layer
        function output = forward(obj, input)
            obj.input = input;
            obj.Z = obj.weights * input + obj.biases;   
            obj.output = obj.activation_function(obj.Z);
            output = obj.output;
        end
        
        % Backward pass through the layer
        function gradient = backward(obj, gradient, learning_rate, N, is_output_layer)
            
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
            obj.weights = obj.weights - learning_rate * dW;
            obj.biases = obj.biases - learning_rate * db;
        end
    end
end
