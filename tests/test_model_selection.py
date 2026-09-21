import numpy as np
import pytest

from ml_from_scratch.linear_models.linear_regression import (
    LinearRegression,
)
from ml_from_scratch.model_selection import select_model


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


def test_select_model_returns_best_model_name():
    X, y = create_dataset()

    models = {
        "model_a": LinearRegression(
            learning_rate=0.01,
            n_iterations=1000,
        ),
        "model_b": LinearRegression(
            learning_rate=0.005,
            n_iterations=2000,
        ),
    }

    best_model, results = select_model(
        models,
        X,
        y,
        cv=5,
        scoring="maximize",
        random_state=42,
    )

    assert best_model in models
    assert set(results.keys()) == set(models.keys())


def test_results_contain_cv_information():
    X, y = create_dataset()

    models = {
        "model_a": LinearRegression(
            learning_rate=0.01,
            n_iterations=1000,
        ),
    }

    _, results = select_model(
        models,
        X,
        y,
        cv=5,
        random_state=42,
    )

    result = results["model_a"]

    assert "scores" in result
    assert "mean" in result
    assert "std" in result

    assert result["scores"].shape == (5,)
    assert np.isfinite(result["mean"])
    assert np.isfinite(result["std"])


def test_minimize_scoring():
    class DummyModel:
        def __init__(self, score_value):
            self.score_value = score_value
            self._parameter_names = ["score_value"]

        def get_params(self):
            return {
                "score_value": self.score_value
            }

        def fit(self, X, y):
            return self

        def score(self, X, y):
            return self.score_value

    X = np.ones((20, 2))
    y = np.ones(20)

    models = {
        "bad": DummyModel(0.8),
        "good": DummyModel(0.2),
    }

    best_model, _ = select_model(
        models,
        X,
        y,
        cv=5,
        scoring="minimize",
        random_state=42,
    )

    assert best_model == "good"


def test_empty_models():
    X, y = create_dataset()

    with pytest.raises(ValueError):
        select_model({}, X, y)


def test_invalid_scoring():
    X, y = create_dataset()

    models = {
        "model": LinearRegression(),
    }

    with pytest.raises(ValueError):
        select_model(
            models,
            X,
            y,
            scoring="invalid",
        )