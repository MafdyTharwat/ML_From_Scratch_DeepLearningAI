import numpy as np
import pytest

from ml_from_scratch.linear_models.linear_regression import (
    LinearRegression,
)
from ml_from_scratch.model_selection import GridSearch


def create_dataset():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 3))

    y = (
        3 * X[:, 0]
        - 2 * X[:, 1]
        + 0.5 * X[:, 2]
        + rng.normal(scale=0.1, size=100)
    )

    return X, y


def test_number_of_combinations():
    X, y = create_dataset()

    model = LinearRegression()

    search = GridSearch(
        estimator=model,
        param_grid={
            "learning_rate": [0.01, 0.05],
            "n_iterations": [500, 1000],
        },
        cv=3,
    )

    search.fit(X, y)

    assert len(search.cv_results_) == 4


def test_best_params_are_returned():
    X, y = create_dataset()

    model = LinearRegression()

    search = GridSearch(
        estimator=model,
        param_grid={
            "learning_rate": [0.005, 0.01],
            "n_iterations": [500, 1000],
        },
        cv=3,
        scoring="maximize",
        random_state=42,
    )

    search.fit(X, y)

    assert search.best_params_ is not None
    assert "learning_rate" in search.best_params_
    assert "n_iterations" in search.best_params_


def test_best_score_is_finite():
    X, y = create_dataset()

    model = LinearRegression()

    search = GridSearch(
        estimator=model,
        param_grid={
            "learning_rate": [0.01, 0.05],
            "n_iterations": [500, 1000],
        },
        cv=3,
        random_state=42,
    )

    search.fit(X, y)

    assert np.isfinite(search.best_score_)


def test_best_estimator_is_fitted():
    X, y = create_dataset()

    model = LinearRegression()

    search = GridSearch(
        estimator=model,
        param_grid={
            "learning_rate": [0.01],
            "n_iterations": [500],
        },
        cv=3,
        random_state=42,
    )

    search.fit(X, y)

    predictions = search.best_estimator_.predict(X)

    assert predictions.shape == (100,)


def test_cv_results_structure():
    X, y = create_dataset()

    model = LinearRegression()

    search = GridSearch(
        estimator=model,
        param_grid={
            "learning_rate": [0.01, 0.05],
            "n_iterations": [500],
        },
        cv=3,
        random_state=42,
    )

    search.fit(X, y)

    assert len(search.cv_results_) == 2

    for result in search.cv_results_:
        assert "params" in result
        assert "scores" in result
        assert "mean_score" in result
        assert "std_score" in result

        assert result["scores"].shape == (3,)


def test_empty_grid():
    X, y = create_dataset()

    model = LinearRegression()

    search = GridSearch(
        estimator=model,
        param_grid={},
    )

    with pytest.raises(ValueError):
        search.fit(X, y)


def test_empty_parameter_values():
    X, y = create_dataset()

    model = LinearRegression()

    search = GridSearch(
        estimator=model,
        param_grid={
            "learning_rate": [],
        },
    )

    with pytest.raises(ValueError):
        search.fit(X, y)


def test_invalid_parameter_values():
    X, y = create_dataset()

    model = LinearRegression()

    search = GridSearch(
        estimator=model,
        param_grid={
            "learning_rate": 0.01,
        },
    )

    with pytest.raises(ValueError):
        search.fit(X, y)


def test_invalid_scoring():
    X, y = create_dataset()

    model = LinearRegression()

    search = GridSearch(
        estimator=model,
        param_grid={
            "learning_rate": [0.01],
        },
        scoring="invalid",
    )

    with pytest.raises(ValueError):
        search.fit(X, y)