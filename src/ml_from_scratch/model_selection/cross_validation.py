import numpy as np
from ..base import clone

def cross_val_score(model, X, y, cv = 5, shuffle = True, random_state = None):
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError('X must be a 2D array')

    if y.ndim == 0:
        raise ValueError('y must be at least 1D')

    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must contain the same number of samples')

    n_samples = X.shape[0]

    if n_samples < 2:
        raise ValueError('X and y must contain at least 2 samples')

    if not isinstance(cv, int) or isinstance(cv, bool):
        raise ValueError('cv must be an integer')

    if cv < 2:
        raise ValueError('cv must be at least 2')

    if cv > n_samples:
        raise ValueError('cv cannot be greater than the number of samples')

    if not hasattr(model, 'get_params'):
        raise ValueError('model must implement get_params()')

    if not hasattr(model, 'fit') or not hasattr(model, 'score'):
        raise ValueError('model must implement fit() and score()')

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    folds = np.array_split(indices, cv)

    scores = []

    for i in range(cv):
        validation_indices = folds[i]

        training_indices = np.concatenate([folds[j] for j in range(cv) if j != i])

        X_train = X[training_indices]
        y_train = y[training_indices]

        X_validation = X[validation_indices]
        y_validation = y[validation_indices]

        fold_model = clone(model)

        fold_model.fit(X_train, y_train)

        score = fold_model.score(X_validation, y_validation)

        scores.append(score)

    return np.asarray(scores, dtype = float)