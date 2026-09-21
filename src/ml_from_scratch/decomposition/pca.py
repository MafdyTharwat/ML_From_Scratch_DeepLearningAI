import numpy as np
from ..base import BaseEstimator

class PCA(BaseEstimator):
    _parameter_names = ['n_components']
    def __init__(self, n_components = None):
        self.n_components = n_components

        if n_components is not None:
            if not isinstance(n_components, int) or isinstance(n_components, bool):
                raise ValueError('n_components must be an integer')

            if n_components < 1:
                raise ValueError('n_components must be at least one')

    def fit(self, X):
        X = np.asarray(X, dtype = float)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[0] < 2:
            raise ValueError('X must contain at least two samples')

        if X.shape[1] == 0:
            raise ValueError('X must contain at least 1 feature')

        n_samples, n_features = X.shape

        if self.n_components is not None and self.n_components > min(n_samples, n_features):
            raise ValueError('n_components can\'t be greater than min(n_samples, n_features)')

        self.n_features_in_ = n_features

        self.mean_ = np.mean(X, axis = 0)

        X_centered = X -self.mean_

        covariance_matrix = np.cov(X_centered, rowvar = False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = self.n_components

        self.explained_variance_ = eigenvalues[: n_components]

        self.components_ = eigenvectors[:, : n_components].T

        total_variance = np.sum(eigenvalues)

        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(n_components)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X):
        if not hasattr(self, 'components_'):
            raise ValueError('PCA must be fitted before transform')

        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[1] != self.n_features_in_:
            raise ValueError('X must have the same number of features as the training data')

        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
