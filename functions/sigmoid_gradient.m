function y = sigmoid_gradient(x)
    sig_x = sigmoid(x);
    y = sig_x .* (1 - sig_x);
end