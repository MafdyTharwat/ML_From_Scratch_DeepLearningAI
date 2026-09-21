import numpy as np
import pytest

from ml_from_scratch.trees.random_forest import RandomForestClassifier


def test_bootstrap_sample_shape():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1] * 5)

    forest = RandomForestClassifier()

    rng = np.random.default_rng(22)

    X_sample, y_sample = forest._bootstrap_sample(X, y, rng)

    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape


def test_bootstrap_is_reproducible():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1] * 5)

    forest = RandomForestClassifier()

    rng1 = np.random.default_rng(22)
    rng2 = np.random.default_rng(22)

    X1, y1 = forest._bootstrap_sample(X, y, rng1)
    X2, y2 = forest._bootstrap_sample(X, y, rng2)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_random_forest_creates_trees():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = RandomForestClassifier(n_estimators = 5, max_depth = 3, random_state = 22)

    model.fit(X, y)

    assert len(model.trees) == 5


def test_random_forest_predict_before_fit():
    model = RandomForestClassifier()

    X = np.array([[1, 2]])

    with pytest.raises(ValueError):
        model.predict(X)


def test_random_forest_predict():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = RandomForestClassifier(n_estimators = 10, max_depth = 5, random_state = 22)

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.all(np.isin(predictions, [0, 1]))


def test_random_forest_score():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = RandomForestClassifier(n_estimators = 10, max_depth = 5, random_state = 22)

    model.fit(X, y)

    score = model.score(X, y)

    assert 0 <= score <= 1