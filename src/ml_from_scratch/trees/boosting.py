import numpy as np

from .decision_tree_regressor import DecisionTreeRegressor
from ..metrics.regression import r2_score


class GradientBoostingRegressor:
    def __init__(self, n_estimators = 100, learning_rate = 0.1, 
                 max_depth = 3, min_samples_split = 2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.trees = []
        self.initial_prediction = None

        self._parameter_names = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split']

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape(-1)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must contain the same number of samples')

        if X.shape[0] == 0:
            raise ValueError('X and y must not be empty')

        if self.n_estimators < 1:
            raise ValueError('no. estimators must be at least 1')

        if self.learning_rate <= 0:
            raise ValueError('learning rate must be greater than 0')

        self.n_features_in_ = X.shape[1]

        self.initial_prediction = np.mean(y)

        current_predictions = np.full(y.shape, self.initial_prediction, dtype = float)

        self.trees = []

        for _ in range(self.n_estimators):

            residuals = y - current_predictions

            tree = DecisionTreeRegressor(max_depth = self.max_depth, min_samples_split = self.min_samples_split)

            tree.fit(X, residuals)

            tree_predictions = tree.predict(X)

            current_predictions += self.learning_rate * tree_predictions

            self.trees.append(tree)

        return self

    def predict(self, X):
        if self.initial_prediction is None:
            raise ValueError('Model must be fitted before prediction')

        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[1] != self.n_features_in_:
            raise ValueError('X must have the same number of features '
                'as the training data')

        predictions = np.full(X.shape[0], self.initial_prediction, dtype = float)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions

    def score(self, X, y):
        y = np.asarray(y, dtype = float).reshape(-1)

        predictions = self.predict(X)

        return r2_score(y, predictions)

