import numpy as np
import pytest

from ml_from_scratch.linear_models.linear_regression import LinearRegression

def test_model_initialization():
    model = LinearRegression(learning_rate = 0.01, n_iterations = 1000)

    assert model.learning_rate == 0.01
    assert model.n_iterations == 1000

    assert model.w is None
    assert model.b == 0.0

    assert model.cost_history == []
    assert model.n_iter_ == 0

def test_compute_cost():
    model = LinearRegression()

    y = np.array([2, 4, 6])
    y_pred = np.array([2, 4, 6])

    cost = model.compute_cost(y, y_pred)

    assert cost == pytest.approx(0.0)

def test_compute_cost_with_error():
    model = LinearRegression()

    y = np.array([2, 4, 6])
    y_pred = np.array([3, 5, 7])

    cost = model.compute_cost(y, y_pred)

    expected_cost = 0.5

    assert cost == pytest.approx(expected_cost)

def test_fit():
    X = np.array([[1], [2], [3], [4], [5]])

    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression(learning_rate = 0.01, n_iterations = 1000)

    model.fit(X, y)

    assert model.w is not None

    assert model.w[0] == pytest.approx(2.0, abs = 0.1)

    assert model.b == pytest.approx(0.0, abs = 0.1)

def test_cost_decreases():
    X = np.array([[1], [2], [3], [4], [5]])

    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression(learning_rate = 0.01, n_iterations = 1000)

    model.fit(X, y)

    initial_cost = model.cost_history[0]
    final_cost = model.cost_history[-1]

    assert final_cost < initial_cost

def test_predict():
    X = np.array([[1], [2], [3], [4], [5]])

    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression(learning_rate = 0.01, n_iterations = 1000)

    model.fit(X, y)

    predictions = model.predict(X)

    expected = np.array([2, 4, 6, 8, 10])

    assert predictions == pytest.approx(expected, abs = 0.1)

def test_prediction_shape():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

    y = np.array([5, 8, 11, 14])

    model = LinearRegression(learning_rate = 0.01, n_iterations = 1000)

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (4,)

def test_score():
    X = np.array([[1], [2], [3], [4], [5]])

    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression(learning_rate = 0.01, n_iterations = 1000)

    model.fit(X, y)

    r2 = model.score(X, y)

    assert r2 > 0.99

def test_predict_before_fit():

    model = LinearRegression()

    X = np.array([[1], [2]])

    with pytest.raises(ValueError):
        model.predict(X)

def test_invalid_X_shape():

    model = LinearRegression()

    X = np.array([1, 2, 3])
    y = np.array([2, 4, 6])

    with pytest.raises(ValueError):
        model.fit(X, y)

def test_mismatched_samples():

    model = LinearRegression()

    X = np.array([[1], [2], [3]])

    y = np.array([2, 4])

    with pytest.raises(ValueError):
        model.fit(X, y)

def test_invalid_learning_rate():

    with pytest.raises(ValueError):
        LinearRegression(learning_rate=0).fit(np.array([[1], [2]]),np.array([2, 4]))

def test_early_stopping():

    X = np.array([[1], [2], [3], [4], [5]])

    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression(learning_rate = 0.01, n_iterations = 10000, tolerance = 1e-7)

    model.fit(X, y)

    assert model.n_iter_ < 10000

def test_gradient_checking():
    model = LinearRegression()

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ])

    y = np.array([3.0, 5.0, 7.0])

    model.w = np.array([0.5, -0.5])
    model.b = 0.2

    y_pred = X @ model.w + model.b

    analytical_dw, analytical_db = model.compute_gradients(X, y, y_pred)

from ml_from_scratch.metrics.regression import (
    mean_squared_error,
    root_mean_squared_error,
    mean_abs_error,
    r2_score,
)


def test_mean_squared_error():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 3, 5])

    assert mean_squared_error(y_true, y_pred) == pytest.approx(5 / 3)


def test_root_mean_squared_error():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 3, 5])

    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
        np.sqrt(5 / 3)
    )


def test_mean_absolute_error():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 3, 5])

    assert mean_abs_error(y_true, y_pred) == pytest.approx(1)


def test_r2_score():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 3, 5])

    expected = 1 - (5 / 2)

    assert r2_score(y_true, y_pred) == pytest.approx(expected)


def test_perfect_predictions():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    assert mean_squared_error(y_true, y_pred) == pytest.approx(0)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(0)
    assert mean_abs_error(y_true, y_pred) == pytest.approx(0)
    assert r2_score(y_true, y_pred) == pytest.approx(1)


def test_mismatched_shapes():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])

    with pytest.raises(ValueError):
        mean_squared_error(y_true, y_pred)

    with pytest.raises(ValueError):
        mean_abs_error(y_true, y_pred)

    with pytest.raises(ValueError):
        r2_score(y_true, y_pred)