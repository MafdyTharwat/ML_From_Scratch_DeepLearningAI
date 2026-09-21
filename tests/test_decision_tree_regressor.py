import numpy as np
import pytest

from ml_from_scratch.trees.decision_tree import Node
from ml_from_scratch.trees.decision_tree_regressor import DecisionTreeRegressor


def test_mse_pure_values():
    tree = DecisionTreeRegressor()

    y = np.array([5, 5, 5, 5])

    assert tree._mse(y) == 0.0


def test_mse_mixed_values():
    tree = DecisionTreeRegressor()

    y = np.array([1, 2, 3, 4])

    assert np.isclose(tree._mse(y), 1.25)


def test_mse_reduction():
    tree = DecisionTreeRegressor()

    y = np.array([1, 2, 10, 11])
    y_left = np.array([1, 2])
    y_right = np.array([10, 11])

    reduction = tree._mse_reduction(
        y,
        y_left,
        y_right
    )

    assert np.isclose(reduction, 20.25)


def test_leaf_value():
    tree = DecisionTreeRegressor()

    y = np.array([2, 4, 6, 8])

    assert tree._leaf_value(y) == 5.0


def test_find_best_split():
    tree = DecisionTreeRegressor()

    X = np.array([
        [1],
        [2],
        [3],
        [10],
        [11],
        [12]
    ])

    y = np.array([
        1,
        2,
        3,
        10,
        11,
        12
    ])

    feature, threshold, reduction = tree._find_best_split(X, y)

    assert feature == 0
    assert threshold in [3, 10]
    assert reduction > 0


def test_pure_node_becomes_leaf():
    tree = DecisionTreeRegressor()

    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([
        5,
        5,
        5
    ])

    tree.fit(X, y)

    assert tree.root.is_leaf()
    assert tree.root.value == 5.0


def test_tree_builds_internal_node():
    tree = DecisionTreeRegressor(max_depth=2)

    X = np.array([
        [1],
        [2],
        [10],
        [11]
    ])

    y = np.array([
        1,
        2,
        10,
        11
    ])

    tree.fit(X, y)

    assert not tree.root.is_leaf()


def test_predict():
    tree = DecisionTreeRegressor(max_depth=2)

    X = np.array([
        [1],
        [2],
        [10],
        [11]
    ])

    y = np.array([
        1,
        2,
        10,
        11
    ])

    tree.fit(X, y)

    predictions = tree.predict(X)

    assert predictions.shape == (4,)
    assert np.all(np.isfinite(predictions))


def test_predict_before_fit():
    tree = DecisionTreeRegressor()

    X = np.array([
        [1],
        [2]
    ])

    with pytest.raises(ValueError):
        tree.predict(X)


def test_predict_wrong_number_of_features():
    tree = DecisionTreeRegressor()

    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([
        1,
        2,
        3
    ])

    tree.fit(X, y)

    X_wrong = np.array([
        [1, 2],
        [3, 4]
    ])

    with pytest.raises(ValueError):
        tree.predict(X_wrong)


def test_score():
    tree = DecisionTreeRegressor(max_depth=3)

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

    tree.fit(X, y)

    score = tree.score(X, y)

    assert score > 0.9


def test_fit_returns_self():
    tree = DecisionTreeRegressor()

    X = np.array([
        [1],
        [2]
    ])

    y = np.array([
        2,
        4
    ])

    result = tree.fit(X, y)

    assert result is tree