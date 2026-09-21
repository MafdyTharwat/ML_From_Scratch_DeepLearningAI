import itertools

import numpy as np

from .cross_validation import cross_val_score
from ..base import clone


class GridSearch:

    def __init__(self, estimator, param_grid, cv = 5, scoring = 'maximize', shuffle = True, random_state = None,):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.shuffle = shuffle
        self.random_state = random_state

        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def _validate_param_grid(self):

        if not isinstance(self.param_grid, dict):
            raise ValueError('param_grid must be a dictionary')

        if len(self.param_grid) == 0:
            raise ValueError('param_grid must not be empty')

        for parameter, values in self.param_grid.items():

            if not isinstance(values, (list, tuple, np.ndarray)):
                raise ValueError(
                    f'Values for \'{parameter}\' must be a '
                    'list, tuple, or numpy array')

            if len(values) == 0:
                raise ValueError(
                    f'Parameter \'{parameter}\' must contain '
                    'at least one value')

    def _generate_parameter_combinations(self):

        parameter_names = list(self.param_grid.keys())

        parameter_values = [
            self.param_grid[name]
            for name in parameter_names
        ]

        combinations = itertools.product(*parameter_values)

        return [dict(zip(parameter_names, combination)) for combination in combinations]

    def fit(self, X, y):

        if self.scoring not in {'maximize', 'minimize',}:
            raise ValueError(
                'scoring must be either '
                '\'maximize\' or \'minimize\'')

        self._validate_param_grid()

        combinations = (self._generate_parameter_combinations())

        all_results = []

        best_score = None
        best_params = None

        for params in combinations:

            model = clone(self.estimator)

            model.set_params(**params)

            scores = cross_val_score(model = model, X = X, y = y, cv = self.cv, shuffle = self.shuffle, random_state = self.random_state,)

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            result = {
                'params': params,
                'scores': scores,
                'mean_score': mean_score,
                'std_score': std_score,
            }

            all_results.append(result)

            if best_score is None:
                best_score = mean_score
                best_params = params

            elif self.scoring == 'maximize':
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

            else:

                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score

        self.best_estimator_ = clone(self.estimator)

        self.best_estimator_.set_params(**self.best_params_)

        self.best_estimator_.fit(X, y)

        self.cv_results_ = all_results

        return self