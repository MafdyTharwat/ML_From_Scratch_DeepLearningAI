import numpy as np
from ..metrics.regression import r2_score
from .decision_tree import Node

class DecisionTreeRegressor:
    def __init__(self, max_depth = 3, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self._parameter_names = ['max_depth', 'min_samples_split']

    def _mse(self, y):
        if len(y) == 0:
            return 0.0

        mean = np.mean(y)

        return np.mean((y - mean) ** 2)

    def _mse_reduction(self, y, y_left, y_right):
        n = len(y)

        if n == 0:
            return 0.0

        parent_mse = self._mse(y)

        left_weight = len(y_left) / n
        right_weight = len(y_right) / n

        children_mse = (left_weight * self._mse(y_left) + right_weight * self._mse(y_right))

        return parent_mse - children_mse

    def _find_best_split(self, X, y):
        n_features = X.shape[1]

        best_reduction = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                reduction = self._mse_reduction(y, y_left, y_right)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature
                    best_threshold = threshold

        return (best_feature, best_threshold, best_reduction)

    def _leaf_value(self, y):
        return np.mean(y)

    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        if (depth >= self.max_depth or n_samples < self.min_samples_split
            or np.all(y == y[0])):
            return Node(value=self._leaf_value(y))

        (best_feature, best_threshold, best_reduction) = self._find_best_split(X, y)

        if (best_feature is None or best_reduction <= 0):
            return Node(value=self._leaf_value(y))

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        X_left = X[left_mask]
        X_right = X[right_mask]

        y_left = y[left_mask]
        y_right = y[right_mask]

        left_subtree = self._build_tree(X_left, y_left, depth + 1)

        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature = best_feature, threshold = best_threshold,
                    left = left_subtree, right = right_subtree)

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape(-1)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must contain the same number of samples')

        if X.shape[0] == 0:
            raise ValueError('X and y must not be empty')

        self.n_features_in_ = X.shape[1]
        self.root = self._build_tree(X, y)

        return self

    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        if self.root is None:
            raise ValueError('Model must be fitted before prediction')

        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[1] != self.n_features_in_:
            raise ValueError('X must have the same number of features as the training data')

        predictions = np.array([self._predict_sample(x, self.root) for x in X])

        return predictions

    def score(self, X, y):
        y = np.asarray(y, dtype = float).reshape(-1)
        predictions = self.predict(X)

        return r2_score(y, predictions)