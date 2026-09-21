import numpy as np
import pytest

from ml_from_scratch.linear_models.logistic_regression import LogisticRegression

def test_model_initialization():
    model = LogisticRegression(learning_rate = 0.01, n_iterations = 1000, tolerance = 1e-7, verbose = False)

    assert model.learning_rate == 0.01
    assert model.n_iterations == 1000

    assert model.w is None
    assert model.b == 0

    assert model.cost_history == []
    assert model.n_iter_ == 0

def test_sigmoid():
    model = LogisticRegression()

    z = np.array([-1, 0, 1])

    result = model.sigmoid(z)

    expected = np.array([
        0.26894142,
        0.5,
        0.73105858
    ])

    assert np.allclose(result, expected, atol = 1e-7)

def test_sigmoid_output_range():
    model = LogisticRegression()

    z = np.array([-10, -2, 0, 2, 10])

    result = model.sigmoid(z)

    assert np.all(result >= 0)
    assert np.all(result <= 1)

def test_sigmoid_extreme_values():
    model = LogisticRegression()

    z = np.array([-1000, -500, 0, 500, 1000])

    result = model.sigmoid(z)

    assert np.all(np.isfinite(result))
    assert result[0] < 1e-10
    assert result[2] == pytest.approx(0.5)
    assert result[-1] > 1 - 1e-10

def test_compute_loss_good_predictions():
    model = LogisticRegression()

    y = np.array([0, 1, 0, 1])
    y_hat = np.array([0.01, 0.99, 0.01, 0.99])

    loss = model.compute_loss(y, y_hat)

    assert loss < 0.1

def test_compute_loss_bad_predictions():
    model = LogisticRegression()

    y = np.array([0, 1, 0, 1])
    y_hat = np.array([0.99, 0.01, 0.99, 0.01])

    loss = model.compute_loss(y, y_hat)

    assert loss > 4

def test_compute_loss_shape_mismatch():
    model = LogisticRegression()

    y = np.array([0, 1, 0])
    y_hat = np.array([0.1, 0.9])

    with pytest.raises(ValueError):
        model.compute_loss(y, y_hat)

def test_compute_gradients_shapes():
    model = LogisticRegression()

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ])

    y = np.array([0, 1, 1])
    y_hat = np.array([0.2, 0.7, 0.8])

    dw, db = model.compute_gradients(X, y, y_hat)

    assert dw.shape == (2,)
    assert np.isscalar(db)

def test_compute_gradients_known_values():
    model = LogisticRegression()

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ])

    y = np.array([0.0, 1.0, 1.0])
    y_hat = np.array([0.2, 0.7, 0.8])

    dw, db = model.compute_gradients(X, y, y_hat)

    error = y_hat - y

    expected_dw = (1 / 3) * (X.T @ error)
    expected_db = (1 / 3) * np.sum(error)

    assert np.allclose(dw, expected_dw)
    assert np.isclose(db, expected_db)

def test_fit_initializes_parameters():
    model = LogisticRegression(
        learning_rate = 0.1,
        n_iterations = 100
    )

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ])

    y = np.array([0, 0, 1, 1])

    model.fit(X, y)

    assert model.w is not None
    assert model.w.shape == (2,)
    assert np.isscalar(model.b)


def test_fit_records_cost_history():
    model = LogisticRegression(
        learning_rate = 0.1,
        n_iterations = 100
    )

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y = np.array([0, 0, 1, 1])

    model.fit(X, y)

    assert len(model.cost_history) > 0
    assert len(model.cost_history) <= model.n_iterations


def test_fit_loss_decreases():
    model = LogisticRegression(
        learning_rate = 0.1,
        n_iterations = 1000
    )

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y = np.array([0, 0, 1, 1])

    model.fit(X, y)

    assert model.cost_history[-1] < model.cost_history[0]


def test_fit_updates_n_iter():
    model = LogisticRegression(
        learning_rate = 0.1,
        n_iterations = 100
    )

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y = np.array([0, 0, 1, 1])

    model.fit(X, y)

    assert model.n_iter_ > 0
    assert model.n_iter_ <= model.n_iterations

def test_predict_proba_shape():
    model = LogisticRegression()

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ])

    y = np.array([0, 0, 1])

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (3,)


def test_predict_proba_range():
    model = LogisticRegression()

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ])

    y = np.array([0, 0, 1])

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_predict_returns_binary_values():
    model = LogisticRegression()

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y = np.array([0, 0, 1, 1])

    model.fit(X, y)

    predictions = model.predict(X)

    assert np.all(np.isin(predictions, [0, 1]))


def test_predict_threshold():
    model = LogisticRegression()

    model.w = np.array([1.0])
    model.b = 0.0

    X = np.array([
        [-2.0],
        [2.0]
    ])

    predictions = model.predict(X, threshold=0.5)

    assert np.array_equal(
        predictions,
        np.array([0, 1])
    )

def test_score_returns_accuracy():
    model = LogisticRegression(
        learning_rate = 0.1,
        n_iterations = 1000
    )

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y = np.array([0, 0, 1, 1])

    model.fit(X, y)

    score = model.score(X, y)

    assert 0 <= score <= 1
    assert score > 0.75

def test_non_binary_target_raises_error():
    model = LogisticRegression()

    X = np.array([
        [1.0],
        [2.0],
        [3.0]
    ])

    y = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_invalid_learning_rate_raises_error():
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=0)


def test_invalid_iterations_raises_error():
    with pytest.raises(ValueError):
        LogisticRegression(n_iterations=0)


def test_predict_before_fit_raises_error():
    model = LogisticRegression()

    X = np.array([
        [1.0, 2.0]
    ])

    with pytest.raises(ValueError):
        model.predict(X)


def test_predict_proba_before_fit_raises_error():
    model = LogisticRegression()

    X = np.array([
        [1.0, 2.0]
    ])

    with pytest.raises(ValueError):
        model.predict_proba(X)


def test_invalid_threshold_raises_error():
    model = LogisticRegression()

    model.w = np.array([1.0])
    model.b = 0.0

    X = np.array([
        [1.0]
    ])

    with pytest.raises(ValueError):
        model.predict(X, threshold = -0.1)

    with pytest.raises(ValueError):
        model.predict(X, threshold = 1.1)

def test_gradient_checking():
    model = LogisticRegression()

    X = np.array([
        [0.5, 1.0],
        [1.0, 1.5],
        [1.5, 2.0]
    ])

    y = np.array([0.0, 1.0, 1.0])

    w = np.array([0.2, -0.3])
    b = 0.1

    # Analytical gradients
    z = X @ w + b
    y_hat = model.sigmoid(z)

    dw, db = model.compute_gradients(X, y, y_hat)

    # Numerical gradient
    epsilon = 1e-5

    numerical_dw = np.zeros_like(w)

    for j in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()

        w_plus[j] += epsilon
        w_minus[j] -= epsilon

        loss_plus = model.compute_loss(
            y,
            model.sigmoid(X @ w_plus + b)
        )

        loss_minus = model.compute_loss(
            y,
            model.sigmoid(X @ w_minus + b)
        )

        numerical_dw[j] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

    loss_plus_b = model.compute_loss(
        y,
        model.sigmoid(X @ w + (b + epsilon))
    )

    loss_minus_b = model.compute_loss(
        y,
        model.sigmoid(X @ w + (b - epsilon))
    )

    numerical_db = (
        loss_plus_b - loss_minus_b
    ) / (2 * epsilon)

    assert np.allclose(
        dw,
        numerical_dw,
        atol=1e-5
    )

    assert np.isclose(
        db,
        numerical_db,
        atol=1e-5
    )

def test_log_loss_matches_metric():
    from ml_from_scratch.metrics.classification import log_loss

    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ])

    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=1000
    )

    model.fit(X, y)

    expected = log_loss(y, model.predict_proba(X))
    result = model.log_loss(X, y)

    assert result == pytest.approx(expected)

def test_log_loss_before_fit():
    model = LogisticRegression()

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0]
    ])

    y = np.array([0, 1])

    with pytest.raises(ValueError):
        model.log_loss(X, y)

def test_compute_loss_matches_log_loss():
    from ml_from_scratch.metrics.classification import log_loss

    y = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.8, 0.9, 0.2])

    model = LogisticRegression()

    expected = log_loss(y, probabilities)
    result = model.compute_loss(y, probabilities)

    assert result == pytest.approx(expected)