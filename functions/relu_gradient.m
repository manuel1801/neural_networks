function y = relu_gradient(x)
    y = double(x > 0);
end