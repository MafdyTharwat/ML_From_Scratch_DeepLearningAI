import numpy as np
from ..base import BaseEstimator

class LogisticRegression(BaseEstimator):
    def __init__(self, learning_rate = 0.01, n_iterations = 1000, tolerance = 1e-7, verbose = False):
        if learning_rate <= 0:
            raise ValueError('learning rate must be greater than 0')

        if n_iterations <= 0:
            raise ValueError('no. iterations must be greater than 0')
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.w = None
        self.b = 0
        self.cost_history = []
        self.n_iter_ = 0
        self._parameter_names = ['learning_rate', 'n_iterations']

    def sigmoid(self, z):
        z = np.asarray(z, dtype = float)
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-1 * z))

    def compute_loss(self, y, y_hat):
        y = np.asarray(y, dtype = float).reshape(-1)
        y_hat = np.asarray(y_hat, dtype = float).reshape(-1)

        if y.shape != y_hat.shape:
            raise ValueError('y and predicted y must have the same shape')

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError('y must contain only binary values (0 and 1)')
        
        m = y.shape[0]
        epsilon = 1e-15

        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        total_loss = -1 * (1 / m) * np.sum((y * np.log(y_hat + epsilon)) + ((1 - y) * np.log(1 - y_hat + epsilon)))

        return total_loss

    def compute_gradients(self, X, y, y_hat):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape(-1)
        y_hat = np.asarray(y_hat, dtype = float).reshape(-1)

        m, n = X.shape

        if y.shape[0] != m:
            raise ValueError('X and y must contain the same number of samples')

        if y_hat.shape[0] != m:
            raise ValueError('X and predicted y must contain the same number of samples')

        error = y_hat - y

        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)

        return dw, db

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape(-1)

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError('y must contain only binary values (0 and 1)')

        m, n = X.shape

        if y.shape[0] != m:
            raise ValueError('X and y must contain the same number of samples')

        self.w = np.zeros(n)
        self.b = 0

        self.cost_history = []
        self.n_iter_ = 0

        for i in range(self.n_iterations):
            z = X @ self.w + self.b
            y_hat = self.sigmoid(z)

            loss = self.compute_loss(y, y_hat)
            self.cost_history.append(loss)

            dw, db = self.compute_gradients(X, y, y_hat)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            self.n_iter_ = i + 1

            if self.verbose:
                if i % 100 == 0 or i == self.n_iterations - 1:
                    print(f'Iteration {i:4d} | Loss: {loss:,.2f}')

            if i > 0:
                previous_loss = self.cost_history[-2]

                if abs(previous_loss - loss) < self.tolerance:
                    if self.verbose:
                        print(f'Early stopping at iteration {i + 1}')
                    break

        return self

    def predict_proba(self, X):
        if self.w is None:
            raise ValueError('Model must be fitted before prediction')

        X = np.asarray(X, dtype = float)

        z = X @ self.w + self.b

        return self.sigmoid(z)

    def predict(self, X, threshold = 0.5):
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')

        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)

    def log_loss(self, X, y):
        probabilities = self.predict_proba(X)

        return self.compute_loss(y, probabilities)

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)

        pred = self.predict(X)

        return np.mean(pred == y)