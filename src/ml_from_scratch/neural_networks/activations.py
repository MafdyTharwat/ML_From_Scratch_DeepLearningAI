import numpy as np

class ReLU:
    def forward(self, X):
        X = np.asarray(X, dtype = float)

        self.input = X

        return np.maximum(0, X)

    def backward(self, doutput):
        drelu = (self.input > 0).astype(float)

        return doutput * drelu


class Sigmoid:
    def forward(self, X):
        X = np.asarray(X, dtype=float)

        self.output = 1 / (1 + np.exp(-X))

        return self.output

    def backward(self, doutput):
        return doutput * self.output * (1 - self.output)


class Tanh:
    def forward(self, X):
        X = np.asarray(X, dtype=float)

        self.output = np.tanh(X)

        return self.output

    def backward(self, doutput):
        return doutput * (1 - self.output ** 2)
