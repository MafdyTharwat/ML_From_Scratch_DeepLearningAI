import numpy as np

from ..base import BaseEstimator

class MyStandardScaler(BaseEstimator):

    _parameter_names = []

    def fit(self, X):
        X = np.asarray(X, dtype = float)

        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)

        self.std = np.where(self.std == 0, 1.0, self.std)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype = float)

        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)