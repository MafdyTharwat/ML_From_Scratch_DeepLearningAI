import numpy as np
import pytest

from ml_from_scratch.trees.boosting import GradientBoostingRegressor


def test_initial_prediction():
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([
        10,
        20,
        30,
        40
    ])

    model = GradientBoostingRegressor(
        n_estimators=1
    )

    model.fit(X, y)

    assert model.initial_prediction == 25.0


def test_number_of_trees():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])

    y = np.array([
        2,
        4,
        6,
        8,
        10
    ])

    model = GradientBoostingRegressor(
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2
    )

    model.fit(X, y)

    assert len(model.trees) == 5


def test_predict_shape():
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([
        2,
        4,
        6,
        8
    ])

    model = GradientBoostingRegressor(
        n_estimators=5,
        max_depth=2
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (4,)
    assert np.all(np.isfinite(predictions))


def test_predict_before_fit():
    model = GradientBoostingRegressor()

    X = np.array([
        [1],
        [2]
    ])

    with pytest.raises(ValueError):
        model.predict(X)


def test_wrong_number_of_features():
    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([
        2,
        4,
        6
    ])

    model = GradientBoostingRegressor(
        n_estimators=3
    )

    model.fit(X, y)

    X_wrong = np.array([
        [1, 2],
        [3, 4]
    ])

    with pytest.raises(ValueError):
        model.predict(X_wrong)


def test_score():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])

    y = np.array([
        2,
        4,
        6,
        8,
        10
    ])

    model = GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.1,
        max_depth=2
    )

    model.fit(X, y)

    score = model.score(X, y)

    assert score > 0.8


def test_more_estimators_reduce_training_error():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
        [6]
    ])

    y = np.array([
        2,
        4,
        6,
        8,
        10,
        12
    ])

    model_small = GradientBoostingRegressor(
        n_estimators=1,
        learning_rate=0.1,
        max_depth=2
    )

    model_large = GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.1,
        max_depth=2
    )

    model_small.fit(X, y)
    model_large.fit(X, y)

    predictions_small = model_small.predict(X)
    predictions_large = model_large.predict(X)

    mse_small = np.mean(
        (y - predictions_small) ** 2
    )

    mse_large = np.mean(
        (y - predictions_large) ** 2
    )

    assert mse_large < mse_small


def test_learning_rate_validation():
    with pytest.raises(ValueError):
        GradientBoostingRegressor(
            learning_rate=0
        ).fit(
            np.array([[1], [2]]),
            np.array([1, 2])
        )


def test_n_estimators_validation():
    with pytest.raises(ValueError):
        GradientBoostingRegressor(
            n_estimators=0
        ).fit(
            np.array([[1], [2]]),
            np.array([1, 2])
        )


def test_fit_returns_self():
    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([
        2,
        4,
        6
    ])

    model = GradientBoostingRegressor(
        n_estimators=3
    )

    result = model.fit(X, y)

    assert result is model