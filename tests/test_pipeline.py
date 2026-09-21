import numpy as np
import pytest

from ml_from_scratch.preprocessing.scaler import MyStandardScaler
from ml_from_scratch.linear_models.linear_regression import LinearRegression
from ml_from_scratch.pipeline.pipeline import Pipeline


def create_dataset():
    rng = np.random.default_rng(42)

    X = rng.normal(
        loc=10,
        scale=5,
        size=(100, 3),
    )

    y = (
        3 * X[:, 0]
        - 2 * X[:, 1]
        + 0.5 * X[:, 2]
    )

    return X, y


def test_pipeline_fit():
    X, y = create_dataset()

    pipeline = Pipeline([
        ("scaler", MyStandardScaler()),
        (
            "model",
            LinearRegression(
                learning_rate=0.01,
                n_iterations=1000,
            ),
        ),
    ])

    result = pipeline.fit(X, y)

    assert result is pipeline


def test_pipeline_predict():
    X, y = create_dataset()

    pipeline = Pipeline([
        ("scaler", MyStandardScaler()),
        (
            "model",
            LinearRegression(
                learning_rate=0.01,
                n_iterations=1000,
            ),
        ),
    ])

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert predictions.shape == (100,)
    assert np.all(np.isfinite(predictions))


def test_pipeline_score():
    X, y = create_dataset()

    pipeline = Pipeline([
        ("scaler", MyStandardScaler()),
        (
            "model",
            LinearRegression(
                learning_rate=0.01,
                n_iterations=1000,
            ),
        ),
    ])

    pipeline.fit(X, y)

    score = pipeline.score(X, y)

    assert score > 0.9


def test_pipeline_requires_fit():
    X, _ = create_dataset()

    pipeline = Pipeline([
        ("scaler", MyStandardScaler()),
        ("model", LinearRegression()),
    ])

    with pytest.raises(ValueError):
        pipeline.predict(X)


def test_pipeline_transform():
    X, y = create_dataset()

    pipeline = Pipeline([
        ("scaler", MyStandardScaler()),
        ("model", LinearRegression()),
    ])

    pipeline.fit(X, y)

    X_transformed = pipeline.transform(X)

    assert X_transformed.shape == X.shape

    np.testing.assert_allclose(
        np.mean(X_transformed, axis=0),
        np.zeros(X.shape[1]),
        atol=1e-10,
    )


def test_pipeline_steps_are_preserved():
    X, y = create_dataset()

    scaler = MyStandardScaler()
    model = LinearRegression()

    pipeline = Pipeline([
        ("scaler", scaler),
        ("model", model),
    ])

    pipeline.fit(X, y)

    assert pipeline.fitted_steps_[0][0] == "scaler"
    assert pipeline.fitted_steps_[1][0] == "model"