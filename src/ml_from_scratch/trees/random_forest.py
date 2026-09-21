import numpy as np

from .decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators = 100, max_depth = 10, min_samples_split = 2, max_features = 'sqrt', random_state = None):
        if n_estimators < 1:
            raise ValueError('no. estimators must be at least 1')

        if max_depth is not None and max_depth < 1:
            raise ValueError('max_depth must be at least 1')

        if min_samples_split < 2:
            raise ValueError('min_samples_split must be at least 2')

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self._parameter_names = ['n_estimators', 'max_depth', 'min_samples_split', 'max_features', 'random_state']

    def _bootstrap_sample(self, X, y, rng):
        n_samples = X.shape[0]

        indices = rng.choice(n_samples, size = n_samples, replace = True)

        return X[indices], y[indices]

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must contain the same number of samples')

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError('y must contain only binary values (0 and 1)')

        if self.n_estimators < 1:
            raise ValueError('no. estimators must be at least 1')

        rng = np.random.default_rng(self.random_state)

        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y, rng)

            tree = DecisionTreeClassifier(max_depth = self.max_depth, min_samples_split = self.min_samples_split,
                max_features = self.max_features, random_state = rng.integers(0, 1_000_000_000))

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        if len(self.trees) == 0:
            raise ValueError('Model must be fitted before prediction')

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[1] != self.n_features_in_:
            raise ValueError('X must contain the same number of features '
            'used during training')

        predictions = np.array([tree.predict(X) for tree in self.trees])

        probability_class_1 = np.mean(predictions, axis = 0)

        return np.column_stack([1 - probability_class_1, probability_class_1])

    def predict(self, X):
        if len(self.trees) == 0:
            raise ValueError('Model must be fitted before prediction')

        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        predictions = np.array([tree.predict(X) for tree in self.trees])

        return (np.mean(predictions, axis = 0) >= 0.5).astype(int)

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)

        predictions = self.predict(X)

        return np.mean(predictions == y)
