import numpy as np

from ..base import BaseEstimator

class Pipeline(BaseEstimator):
    _parameter_names = ['steps']
    
    def __init__(self, steps):
        self.steps = steps
        
        if not isinstance(steps, list):
            raise ValueError('steps must be a list')

        if len(steps) == 0:
            raise ValueError('steps must not be empty')

        for step in steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise ValueError('Each step must be a tuple: (name, estimator)')

            name, estimator = step

            if not isinstance(name, str):
                raise ValueError('Step name must be a string')

            if not hasattr(estimator, "fit"):
                raise ValueError(f'Step \'{name}\' must implement fit()')

    def fit(self, X, y):
        X_current = np.asarray(X)
        y = np.asarray(y)

        self.fitted_steps_ = []

        for name, step in self.steps[:-1]:
            step.fit(X_current)
            X_current = step.transform(X_current)

            self.fitted_steps_.append((name, step))

        final_name, final_step = self.steps[-1]

        final_step.fit(X_current, y)

        self.fitted_steps_.append((final_name, final_step))

        self.n_features_in_ = X.shape[1]

        return self
    
    def transform(self, X):
        X_current = np.asarray(X)

        for _, step in self.fitted_steps_[:-1]:
            X_current = step.transform(X_current)

        return X_current

    def predict(self, X):
        if not hasattr(self, 'fitted_steps_'):
            raise ValueError(
                'Pipeline must be fitted before prediction')

        X_current = self.transform(X)

        _, model = self.fitted_steps_[-1]

        return model.predict(X_current)

    def score(self, X, y):
        if not hasattr(self, 'fitted_steps_'):
            raise ValueError('Pipeline must be fitted before scoring')

        X_current = self.transform(X)

        _, model = self.fitted_steps_[-1]

        return model.score(X_current, y)