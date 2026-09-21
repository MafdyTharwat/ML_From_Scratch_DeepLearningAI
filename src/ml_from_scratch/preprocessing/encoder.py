import numpy as np
from ..base import BaseEstimator

class OneHotEncoder(BaseEstimator):
    _parameter_names = ['handle_unknown']

    def __init__(self, handle_unknown = 'error'):
        self.handle_unknown = handle_unknown

        if handle_unknown not in {'error', 'ignore'}:
            raise ValueError('handle_unknown must be either \'error\' or \'ignore\'')

    def fit(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError('X must be 2D array')

        self.n_features_in_ = X.shape[1]

        self.categories_ = []

        for feature in range(X.shape[1]):
            categories = np.unique(X[:, feature])
            self.categories_.append(categories)

        return self

    def transform(self, X):
        if not hasattr(self, 'categories_'):
            raise ValueError('Encoder must be fitted before transform')

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.ndim != 2:
            raise ValueError('X must be 2D array')
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                'X must have the same number of fatures '
            'as the training data')

        encoded_features = []

        for feature in range(X.shape[1]):
            categories = self.categories_[feature]

            encoded = np.zeros((X.shape[0], len(categories)), dtype = float)

            for category_index, category in enumerate(categories):
                mask = X[:, feature] == category
                encoded[mask, category_index] = 1.0

            unknown_mask = ~np.isin(X[:, feature], categories)

            if np.any(unknown_mask):
                if self.handle_unknown == 'error':
                    raise ValueError(f'Unknown category found in feature {feature}')

            encoded_features.append(encoded)

        return np.hstack(encoded_features)

    def fit_transform(self, X):
        return self.fit(X).transform(X)