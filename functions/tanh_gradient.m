function y = tanh_gradient(x)
    y = 1 - tanh(x).^2;
end