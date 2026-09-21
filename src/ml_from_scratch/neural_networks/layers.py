import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

        self.input = None
        self.dweights = None
        self.dbias = None

    def forward(self, X):
        X = np.asarray(X, dtype = float)

        self.input = X

        return X @ self.weights + self.bias

    def backward(self, doutput):
        self.dweights = self.input.T @ doutput
        self.dbias = np.sum(doutput, axis = 0, keepdims = True)

        dinput = doutput @ self.weights.T

        return dinput