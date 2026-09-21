import numpy as np

def train_test_split(X, y, test_size = 0.2, random_state = None, shuffle = True):
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 0:
        raise ValueError('X must be at least 1D')

    if y.ndim == 0:
        raise ValueError('y must be at least 1D')

    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must contain the same number of samples')

    n_samples = X.shape[0]

    if n_samples < 2:
        raise ValueError('X and y must contain at least 2 samples')

    if not isinstance(test_size, (float, int)):
        raise ValueError('test size must be a float or integer')

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError('Float test_size must be between 0 and 1')

        n_test = int(np.ceil(n_samples * test_size))

    else:
        if not 0 < test_size < n_samples:
            raise ValueError('Integer test_size must be between 1 and n_samples - 1')

        n_test = int(test_size)

    n_train = n_samples - n_test

    if n_train < 1:
        raise ValueError('test_size leaves no samples for training')

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

