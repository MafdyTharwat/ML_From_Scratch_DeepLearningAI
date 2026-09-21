class BaseEstimator:

    _parameter_names = []

    def get_params(self):
        return {
            key: getattr(self, key)
            for key in self._parameter_names
        }

    def set_params(self, **params):
        for key, value in params.items():

            if key not in self._parameter_names:
                raise ValueError(
                    f'Invalid parameter \'{key}\' '
                    f'for {type(self).__name__}'
                )

            setattr(self, key, value)

        return self


def clone(estimator):

    if hasattr(estimator, 'get_params'):

        params = estimator.get_params()

        cloned_params = {
            key: clone(value)
            for key, value in params.items()
        }

        return type(estimator)(**cloned_params)

    if isinstance(estimator, list):
        return [clone(value) for value in estimator]

    if isinstance(estimator, tuple):
        return tuple(clone(value) for value in estimator)

    if isinstance(estimator, dict):
        return {
            key: clone(value)
            for key, value in estimator.items()
        }

    return estimator