import numpy as np

class Node:
    def __init__(self, feature = None, threshold = None, left = None,
                 right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth = 10, min_samples_split = 2, max_features = None, random_state = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self._parameter_names = ['max_depth', 'min_samples_split', 'max_features', 'random_state']

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts = True)
        probabilities = counts / len(y)

        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, y, y_left, y_right):
        n = len(y)

        if n == 0:
            return 0

        left_weight = len(y_left) / n
        right_weight = len(y_right) / n

        parent_entropy = self._entropy(y)

        child_entropy = left_weight * self._entropy(y_left) + right_weight * self._entropy(y_right)

        return parent_entropy - child_entropy
    
    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        X_left = X[left_mask]
        X_right = X[right_mask]

        y_left = y[left_mask]
        y_right = y[right_mask]

        return X_left, X_right, y_left, y_right

    def _get_feature_subset(self, n_features, rng):
        if self.max_features is None:
            return np.arange(n_features)

        if self.max_features == 'sqrt':
            n_selected = max(1, int(np.sqrt(n_features)))

        elif self.max_features == 'log2':
            n_selected = max(1, int(np.log2(n_features)))

        elif isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError('max_features must be at least 1')
            n_selected = min(self.max_features, n_features)

        else:
            raise ValueError('max_features must be None, \'sqrt\', \'log2\', or an integer')

        return rng.choice(n_features, size = n_selected, replace = False)

    def _find_best_split(self, X, y, rng=None):
        n_samples, n_features = X.shape

        if rng is None:
            rng = np.random.default_rng(self.random_state)

        features = self._get_feature_subset(n_features, rng)

        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in features:
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                _, _, y_left, y_right = self._split(X, y, feature, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _most_common_label(self, y):
        labels, counts = np.unique(y, return_counts = True)
        return labels[np.argmax(counts)]

    def _build_tree(self, X, y, depth = 0, rng = None):
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        if n_classes == 1 or depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)

        best_feature, best_threshold, best_gain = self._find_best_split(X, y, rng)

        if best_feature is None or best_gain <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)

        X_left, X_right, y_left, y_right = self._split(X, y, best_feature, best_threshold)

        left_subtree = self._build_tree(X_left, y_left, depth + 1, rng)
        right_subtree = self._build_tree(X_right, y_right, depth + 1, rng)

        return Node(feature = best_feature, threshold = best_threshold,
                    left = left_subtree, right = right_subtree)

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError('X must be a 2-dimensional array')

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must contain the same number of samples')

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError('y must contain only binary values (0 and 1)')

        rng = np.random.default_rng(self.random_state)

        self.root = self._build_tree(X, y, rng = rng)

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

        predictions = np.array([self._predict_sample(x, self.root) for x in X])

        return predictions.astype(int)
    
    def score(self, X, y):
        y = np.asarray(y).reshape(-1)

        predictions = self.predict(X)

        return np.mean(predictions == y)