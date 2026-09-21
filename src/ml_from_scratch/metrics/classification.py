import numpy as np

def validate_classification_inputs(y, y_hat):
    y = np.asarray(y, dtype = float).reshape(-1)
    y_hat = np.asarray(y_hat, dtype = float).reshape(-1)

    if y.shape != y_hat.shape:
        raise ValueError('y and predicted y must have the same shape')

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError('y must contain only binary numbers (0 or 1)')
    
    if not np.all(np.isin(y_hat, [0, 1])):
        raise ValueError('predicted y must contain only binary numbers (0 or 1)')
    
    return y, y_hat

def validate_probability_inputs(y, y_hat):
    y = np.asarray(y, dtype = float).reshape(-1)
    y_hat = np.asarray(y_hat, dtype = float).reshape(-1)
    
    if y.shape != y_hat.shape:
        raise ValueError('y and predicted y must have the same shape')
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError('y must contain only binary numbers (0 or 1)')
        
    if np.any((y_hat < 0) | (y_hat > 1)):
        raise ValueError('predicted probabilities must be between 0 and 1')
        
    return y, y_hat

def accuracy_score(y, y_hat):
    y, y_hat = validate_classification_inputs(y, y_hat)

    return np.mean(y == y_hat)

def confusion_matrix(y, y_hat):
    y, y_hat = validate_classification_inputs(y, y_hat)

    tn = np.sum((y == 0) & (y_hat == 0))
    tp = np.sum((y == 1) & (y_hat == 1))
    fn = np.sum((y == 1) & (y_hat == 0))
    fp = np.sum((y == 0) & (y_hat == 1))

    return np.array([
        [tn, fp],
        [fn, tp]
    ])

def precision_score(y, y_hat):
    con_matrix = confusion_matrix(y, y_hat)

    tn, fp = con_matrix[0]
    fn, tp = con_matrix[1]

    denominator = tp + fp

    if denominator == 0:
        return 0
    return tp / denominator

def recall_score(y, y_hat):
    y, y_hat = validate_classification_inputs(y, y_hat)

    con_matrix = confusion_matrix(y, y_hat)

    tn, fp = con_matrix[0]
    fn, tp = con_matrix[1]

    denominator = tp + fn

    if denominator == 0:
        return 0
    return tp / denominator

def f1_score(y, y_hat):
    precision = precision_score(y, y_hat)
    recall = recall_score(y, y_hat)

    denominator = precision + recall

    if denominator == 0:
        return 0
    return 2 * (precision * recall) / denominator

def log_loss(y, y_hat):
    y, y_hat  = validate_probability_inputs(y, y_hat)

    epsilon = 1e-7
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

