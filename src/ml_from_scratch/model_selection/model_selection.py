import numpy as np

from .cross_validation import cross_val_score


def select_model(models, X, y, cv = 5, scoring = 'maximize', shuffle = True, random_state = None):
    if not isinstance(models, dict):
        raise ValueError('models must be a dictionary')

    if len(models) == 0:
        raise ValueError('models must not be empty')

    if scoring not in {'maximize', 'minimize'}:
        raise ValueError('scoring must be either \'maximize\' or \'minimize\'')

    results = {}

    for name, model in models.items():
        scores = cross_val_score(model = model, X = X, y = y, cv = cv, shuffle = shuffle, random_state = random_state)

        results[name] = {'scores': scores, 'mean': np.mean(scores), 'std': np.std(scores)}

    if scoring == 'maximize':
        best_name = max(results, key = lambda name: results[name]['mean'])
    else:
        best_name = min(results, key = lambda name: results[name]['mean'])

    return best_name, results