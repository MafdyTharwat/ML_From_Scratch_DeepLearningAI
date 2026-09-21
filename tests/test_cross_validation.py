import numpy as np
import pytest

from ml_from_scratch.linear_models.linear_regression import LinearRegression
from ml_from_scratch.model_selection.cross_validation import cross_val_score


def test_cross_val_score_returns_correct_number_of_scores():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)

    model = LinearRegression(
        learning_rate=0.01,
        n_iterations=100,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        random_state=42,
    )

    assert scores.shape == (5,)


def test_cross_val_scores_are_finite():
    X = np.arange(40).reshape(20, 2).astype(float)
    y = np.arange(20).astype(float)

    model = LinearRegression(
        learning_rate=0.01,
        n_iterations=100,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        random_state=42,
    )

    assert np.all(np.isfinite(scores))


def test_cross_val_is_reproducible():
    X = np.arange(40).reshape(20, 2).astype(float)
    y = np.arange(20).astype(float)

    model = LinearRegression(
        learning_rate=0.01,
        n_iterations=100,
    )

    scores_1 = cross_val_score(
        model,
        X,
        y,
        cv=5,
        random_state=42,
    )

    scores_2 = cross_val_score(
        model,
        X,
        y,
        cv=5,
        random_state=42,
    )

    np.testing.assert_array_equal(
        scores_1,
        scores_2,
    )


def test_different_random_states_can_change_scores():
    X = np.arange(40).reshape(20, 2).astype(float)
    y = np.arange(20).astype(float)

    model = LinearRegression(
        learning_rate=0.01,
        n_iterations=100,
    )

    scores_1 = cross_val_score(
        model,
        X,
        y,
        cv=5,
        random_state=42,
    )

    scores_2 = cross_val_score(
        model,
        X,
        y,
        cv=5,
        random_state=123,
    )

    assert not np.array_equal(scores_1, scores_2)


def test_invalid_cv():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)

    model = LinearRegression()

    with pytest.raises(ValueError):
        cross_val_score(model, X, y, cv=1)

    with pytest.raises(ValueError):
        cross_val_score(model, X, y, cv=11)


def test_mismatched_samples():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(9).astype(float)

    model = LinearRegression()

    with pytest.raises(ValueError):
        cross_val_score(model, X, y)


def test_invalid_model():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)

    class InvalidModel:
        pass

    with pytest.raises(ValueError):
        cross_val_score(
            InvalidModel(),
            X,
            y,
        )