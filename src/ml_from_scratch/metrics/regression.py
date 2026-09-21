import numpy as np

def mean_squared_error(y, pred):
    y = np.asarray(y, dtype = float).reshape(-1)
    pred = np.asarray(pred, dtype = float).reshape(-1)

    if y.shape != pred.shape:
        raise ValueError('y and predicted y must have the same shape')

    return np.mean((y - pred) ** 2)

def root_mean_squared_error(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

def mean_abs_error(y, pred):
    y = np.asarray(y, dtype = float).reshape(-1)
    pred = np.asarray(pred, dtype = float).reshape(-1)

    if y.shape != pred.shape:
        raise ValueError('y and predicted y must have the same shape')

    return np.mean(np.abs(y - pred))

def r2_score(y, pred):
    y = np.asarray(y, dtype = float).reshape(-1)
    pred = np.asarray(pred, dtype = float).reshape(-1)

    if y.shape != pred.shape:
            raise ValueError('y and predicted y must have the same shape')

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
         return 0
    return 1 - (ss_res / ss_tot)

def test_all_metrices(y, pred):
    mse = mean_squared_error(y, pred)
    rmse = root_mean_squared_error(y, pred)
    mae = mean_abs_error(y, pred)
    r2 = r2_score(y, pred)

    print(f'MSE  : {mse:.6f}')
    print(f'RMSE : {rmse:.6f}')
    print(f'MAE  : {mae:.6f}')
    print(f'R2   : {r2:.6f}')