import numpy as np
from ml_from_scratch.metrics import r2_score
from ..base import BaseEstimator

class LinearRegression(BaseEstimator):
    def __init__(self, learning_rate = 0.01, n_iterations = 1000, tolerance = 1e-7, verbose = False):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

        if n_iterations <= 0:
            raise ValueError("n_iterations must be greater than 0.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.w = None
        self.b = 0.0
        self.cost_history = []
        self.n_iter_ = 0
        self._parameter_names = ['learning_rate', 'n_iterations']

    def compute_cost(self, y, pred):
        y = np.asarray(y).reshape(-1)
        pred = np.asarray(pred).reshape(-1)
        m = y.shape[0]
        cost = (1 / (2 * m) * np.sum((pred - y) ** 2))
        return cost

    def compute_gradients(self, X, y, pred):
        m = X.shape[0]

        error = pred - y

        dw = (1 / m) * (X.T @ error)
        db = (1 / m) * np.sum(error)

        return dw, db

    def fit(self, X, y):

        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape(-1)

        m, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        self.cost_history = []

        for i in range(self.n_iterations):
            y_pred = X @ self.w + self.b

            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)

            dw, db = self.compute_gradients(X, y, y_pred)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            self.n_iter_ = i + 1

            if self.verbose:
                if i % 100 == 0 or i == self.n_iterations - 1:
                    print(f'Iteration {i + 1:4d}, Cost: {cost:,.2f}')

            if i > 0:

                previous_cost = self.cost_history[-2]

                if abs(previous_cost - cost) < self.tolerance:
                    if self.verbose:
                        print(f'Early stopping at iteration {i + 1}.')
                    break
        return self

    def predict(self, X):
        return X @ self.w + self.b

    def score(self, X, y):
        y = np.asarray(y, dtype = float).reshape(-1)
        pred = self.predict(X)

        return r2_score(y, pred)