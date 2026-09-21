import numpy as np
import pytest

from ml_from_scratch.metrics.classification import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

def test_accuracy_score():
    y = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    result = accuracy_score(y, y_pred)

    assert result == pytest.approx(0.8)

def test_accuracy_perfect_prediction():
    y = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    result = accuracy_score(y, y_pred)

    assert result == pytest.approx(1.0)

def test_accuracy_zero_prediction():
    y = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1])

    result = accuracy_score(y, y_pred)

    assert result == pytest.approx(0.0)

def test_confusion_matrix():
    y = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    result = confusion_matrix(y, y_pred)

    expected = np.array([
        [2, 1],
        [1, 2]
    ])

    assert np.array_equal(result, expected)

def test_confusion_matrix_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    result = confusion_matrix(y, y_pred)

    expected = np.array([
        [2, 0],
        [0, 2]
    ])

    assert np.array_equal(result, expected)

def test_precision_score():

    y = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    result = precision_score(y, y_pred)

    assert result == pytest.approx(2 / 3)

def test_precision_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    result = precision_score(y, y_pred)

    assert result == pytest.approx(1.0)

def test_precision_no_positive_predictions():
    y = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])

    result = precision_score(y, y_pred)

    assert result == pytest.approx(0.0)

def test_recall_score():
    y = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    result = recall_score(y, y_pred)

    assert result == pytest.approx(2 / 3)

def test_recall_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    result = recall_score(y, y_pred)

    assert result == pytest.approx(1.0)

def test_recall_no_actual_positive_samples():
    y = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1])

    result = recall_score(y, y_pred)

    assert result == pytest.approx(0.0)

def test_f1_score():
    precision = 2 / 3
    recall = 2 / 3

    expected = 2 * (precision * recall) / (precision + recall)

    y = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    result = f1_score(y, y_pred)

    assert result == pytest.approx(expected)

def test_f1_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    result = f1_score(y, y_pred)

    assert result == pytest.approx(1.0)

def test_f1_zero_when_no_true_positive():
    y = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])

    result = f1_score(y, y_pred)

    assert result == pytest.approx(0.0)

def test_shape_mismatch_raises_error():
    y = np.array([0, 1, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError):
        accuracy_score(y, y_pred)

def test_shape_mismatch_precision_raises_error():
    y = np.array([0, 1, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError):
        precision_score(y, y_pred)

def test_shape_mismatch_recall_raises_error():
    y = np.array([0, 1, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError):
        recall_score(y, y_pred)

def test_shape_mismatch_f1_raises_error():
    y = np.array([0, 1, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError):
        f1_score(y, y_pred)

def test_confusion_matrix_rejects_non_binary_true_labels():
    y = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 0])

    with pytest.raises(ValueError):
        confusion_matrix(y, y_pred)

def test_confusion_matrix_rejects_non_binary_predictions():
    y = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 2, 0])

    with pytest.raises(ValueError):
        confusion_matrix(y, y_pred)

def test_metrics_are_consistent_with_confusion_matrix():
    y = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    cm = confusion_matrix(y, y_pred)

    tn, fp = cm[0]
    fn, tp = cm[1]

    expected_accuracy = (tp + tn) / (tp + tn + fp + fn)
    expected_precision = tp / (tp + fp)
    expected_recall = tp / (tp + fn)

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    assert accuracy == pytest.approx(expected_accuracy)
    assert precision == pytest.approx(expected_precision)
    assert recall == pytest.approx(expected_recall)

def test_log_loss_perfect_predictions():
    y = np.array([0, 1, 0, 1])
    y_hat = np.array([0.01, 0.99, 0.01, 0.99])

    result = log_loss(y, y_hat)

    assert result < 0.02

def test_log_loss_bad_predictions():
    y = np.array([0, 1, 0, 1])
    y_hat = np.array([0.99, 0.01, 0.99, 0.01])

    result = log_loss(y, y_hat)

    assert result > 4

def test_log_loss_known_value():
    y = np.array([1, 0])
    y_hat = np.array([0.9, 0.8])

    expected = -(
        np.log(0.9) +
        np.log(1 - 0.8)
    ) / 2

    assert log_loss(y, y_hat) == pytest.approx(expected)

def test_log_loss_probability_range():
    y = np.array([0, 1])
    y_hat = np.array([-0.1, 1.1])

    with pytest.raises(ValueError):
        log_loss(y, y_hat)


def test_log_loss_shape_mismatch():
    y = np.array([0, 1, 1])
    y_hat = np.array([0.2, 0.8])

    with pytest.raises(ValueError):
        log_loss(y, y_hat)

def test_log_loss_binary_target_validation():
    y = np.array([0, 1, 2])
    y_hat = np.array([0.2, 0.8, 0.5])

    with pytest.raises(ValueError):
        log_loss(y, y_hat)

