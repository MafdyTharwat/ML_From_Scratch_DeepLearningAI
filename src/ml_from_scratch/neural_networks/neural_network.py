import numpy as np

from .layers import Dense
from .activations import ReLU, Sigmoid
from .losses import BinaryCrossEntropyLoss


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size = 1,
                  learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.layers = [
            Dense(input_size, hidden_size),
            ReLU(),
            Dense(hidden_size, output_size),
            Sigmoid(),
        ]

        self.loss = BinaryCrossEntropyLoss()

        self.loss_history = []
        self._parameter_names = ['input_size', 'hidden_size', 'output_size', 'learning_rate', 'n_iterations']

    def forward(self, X):
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, dloss):
        gradient = dloss

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update_parameters(self):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights -= self.learning_rate * layer.dweights
                layer.bias -= self.learning_rate * layer.dbias

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape(-1, 1)

        self.loss_history = []

        for _ in range(self.n_iterations):

            y_hat = self.forward(X)

            loss = self.loss.forward(y, y_hat)

            self.loss_history.append(loss)

            dloss = self.loss.backward()

            self.backward(dloss)

            self.update_parameters()

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype = float)

        return self.forward(X)

    def predict(self, X, threshold = 0.5):
        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        predictions = self.predict(X).reshape(-1)

        return np.mean(predictions == y)