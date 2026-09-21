import numpy as np
import pytest

from ml_from_scratch.preprocessing.splitting import train_test_split


def test_split_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    assert X_train.shape == (8, 2)
    assert X_test.shape == (2, 2)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)


def test_split_preserves_samples():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    combined_X = np.vstack([X_train, X_test])
    combined_y = np.concatenate([y_train, y_test])

    assert set(map(tuple, combined_X)) == set(map(tuple, X))
    assert set(combined_y) == set(y)


def test_split_keeps_X_and_y_aligned():
    X = np.arange(10).reshape(5, 2)
    y = np.array([10, 20, 30, 40, 50])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    for row, target in zip(X_train, y_train):
        assert target == row[0] * 5 + 10

    for row, target in zip(X_test, y_test):
        assert target == row[0] * 5 + 10


def test_random_state_is_reproducible():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    split_1 = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    split_2 = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    for a, b in zip(split_1, split_2):
        np.testing.assert_array_equal(a, b)


def test_different_random_states_can_change_split():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    split_1 = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    split_2 = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=123,
    )

    assert not np.array_equal(split_1[1], split_2[1])


def test_shuffle_false_preserves_order():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        shuffle=False,
    )

    np.testing.assert_array_equal(
        y_train,
        np.array([0, 1, 2]),
    )

    np.testing.assert_array_equal(
        y_test,
        np.array([3, 4]),
    )


def test_integer_test_size():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=3,
        random_state=42,
    )

    assert len(X_train) == 7
    assert len(X_test) == 3


def test_invalid_test_size():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=0)

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.0)

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=5)


def test_mismatched_samples():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(4)

    with pytest.raises(ValueError):
        train_test_split(X, y)


def test_too_few_samples():
    X = np.array([[1, 2]])
    y = np.array([1])

    with pytest.raises(ValueError):
        train_test_split(X, y)