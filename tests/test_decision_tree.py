import numpy as np
import pytest 

from ml_from_scratch.trees.decision_tree import (
    DecisionTreeClassifier, Node)


def test_entropy_pure_class():
    tree = DecisionTreeClassifier()

    y = np.array([0, 0, 0, 0])

    assert tree._entropy(y) == 0.0


def test_entropy_balanced_classes():
    tree = DecisionTreeClassifier()

    y = np.array([0, 0, 1, 1])

    assert np.isclose(tree._entropy(y), 1.0)


def test_entropy_mixed_classes():
    tree = DecisionTreeClassifier()

    y = np.array([0, 0, 0, 1])

    assert np.isclose(tree._entropy(y), 0.811278)


def test_node_leaf():
    node = Node(value = 1)

    assert node.is_leaf()


def test_node_internal():
    node = Node(feature = 0, threshold = 0.5)

    assert not node.is_leaf()

def test_information_gain_perfect_split():
    tree = DecisionTreeClassifier()

    y = np.array([0, 0, 0, 1, 1, 1])
    y_left = np.array([0, 0, 0])
    y_right = np.array([1, 1, 1])

    gain = tree._information_gain(y, y_left, y_right)

    assert np.isclose(gain, 1.0)

def test_information_gain_no_improvement():
    tree = DecisionTreeClassifier()

    y = np.array([0, 0, 1, 1])
    y_left = np.array([0, 1])
    y_right = np.array([0, 1])

    gain = tree._information_gain(y, y_left, y_right)

    assert np.isclose(gain, 0.0)

def test_split():
    tree = DecisionTreeClassifier()

    X = np.array([
        [2, 10],
        [3, 12],
        [8, 20],
        [9, 22],
    ])

    y = np.array([0, 0, 1, 1])

    X_left, X_right, y_left, y_right = tree._split(X, y, feature = 0, threshold = 5)

    np.testing.assert_array_equal(
        X_left,
        np.array([
            [2, 10],
            [3, 12],
        ])
    )

    np.testing.assert_array_equal(
        X_right,
        np.array([
            [8, 20],
            [9, 22],
        ])
    )

    np.testing.assert_array_equal(
        y_left,
        np.array([0, 0])
    )

    np.testing.assert_array_equal(
        y_right,
        np.array([1, 1])
    )

def test_find_best_split():
    tree = DecisionTreeClassifier()

    X = np.array([
        [2],
        [3],
        [8],
        [9],
    ])

    y = np.array([0, 0, 1, 1])

    feature, threshold, gain = tree._find_best_split(X, y)

    assert feature == 0
    assert threshold in [2, 3, 8]
    assert gain > 0

def test_most_common_label():
    tree = DecisionTreeClassifier()

    y = np.array([0, 1, 1, 1, 0])

    assert tree._most_common_label(y) == 1

def test_build_tree_pure_node():
    tree = DecisionTreeClassifier()

    X = np.array([
        [1],
        [2],
        [3],
    ])

    y = np.array([1, 1, 1])

    node = tree._build_tree(X, y)

    assert node.is_leaf()
    assert node.value == 1

def test_build_tree_creates_internal_node():
    tree = DecisionTreeClassifier(max_depth = 3)

    X = np.array([
        [1],
        [2],
        [8],
        [9],
    ])

    y = np.array([0, 0, 1, 1])

    tree.fit(X, y)

    assert tree.root is not None
    assert not tree.root.is_leaf()

    assert tree.root.left is not None
    assert tree.root.right is not None

def test_predict():
    tree = DecisionTreeClassifier(max_depth = 3)

    X = np.array([
        [1],
        [2],
        [8],
        [9],
    ])

    y = np.array([0, 0, 1, 1])

    tree.fit(X, y)

    predictions = tree.predict(X)

    np.testing.assert_array_equal(predictions, y)

def test_predict_new_samples():
    tree = DecisionTreeClassifier(max_depth = 3)

    X = np.array([
        [1],
        [2],
        [8],
        [9],
    ])

    y = np.array([0, 0, 1, 1])

    tree.fit(X, y)

    X_new = np.array([
        [1.5],
        [8.5],
    ])

    predictions = tree.predict(X_new)

    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_predict_before_fit():
    tree = DecisionTreeClassifier()

    X = np.array([
        [1],
        [2],
    ])

    with pytest.raises(ValueError):
        tree.predict(X)

def test_score():
    tree = DecisionTreeClassifier(max_depth = 3)

    X = np.array([
        [1],
        [2],
        [8],
        [9],
    ])

    y = np.array([0, 0, 1, 1])

    tree.fit(X, y)

    assert tree.score(X, y) == 1.0
