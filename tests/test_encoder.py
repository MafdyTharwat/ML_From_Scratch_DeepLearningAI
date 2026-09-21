import numpy as np
import pytest

from ml_from_scratch.preprocessing.encoder import OneHotEncoder


def test_fit_creates_categories():
    X = np.array([
        ["Egypt"],
        ["Germany"],
        ["Egypt"],
        ["France"],
    ])

    encoder = OneHotEncoder()

    encoder.fit(X)

    assert len(encoder.categories_) == 1

    np.testing.assert_array_equal(
        encoder.categories_[0],
        np.array(["Egypt", "France", "Germany"]),
    )


def test_transform_shape():
    X = np.array([
        ["Egypt"],
        ["Germany"],
        ["France"],
        ["Egypt"],
    ])

    encoder = OneHotEncoder()

    X_encoded = encoder.fit_transform(X)

    assert X_encoded.shape == (4, 3)


def test_transform_values():
    X = np.array([
        ["Egypt"],
        ["Germany"],
        ["France"],
    ])

    encoder = OneHotEncoder()

    X_encoded = encoder.fit_transform(X)

    expected = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])

    np.testing.assert_array_equal(
        X_encoded,
        expected,
    )


def test_multiple_features():
    X = np.array([
        ["Egypt", "Male"],
        ["Germany", "Female"],
        ["Egypt", "Female"],
    ])

    encoder = OneHotEncoder()

    X_encoded = encoder.fit_transform(X)

    assert X_encoded.shape == (3, 4)


def test_unknown_category_error():
    X_train = np.array([
        ["Egypt"],
        ["Germany"],
    ])

    X_test = np.array([
        ["Japan"],
    ])

    encoder = OneHotEncoder(
        handle_unknown="error"
    )

    encoder.fit(X_train)

    with pytest.raises(ValueError):
        encoder.transform(X_test)


def test_unknown_category_ignore():
    X_train = np.array([
        ["Egypt"],
        ["Germany"],
    ])

    X_test = np.array([
        ["Japan"],
    ])

    encoder = OneHotEncoder(
        handle_unknown="ignore"
    )

    encoder.fit(X_train)

    X_encoded = encoder.transform(X_test)

    assert X_encoded.shape == (1, 2)

    np.testing.assert_array_equal(
        X_encoded,
        np.array([[0, 0]]),
    )


def test_transform_before_fit():
    X = np.array([
        ["Egypt"],
    ])

    encoder = OneHotEncoder()

    with pytest.raises(ValueError):
        encoder.transform(X)


def test_wrong_number_of_features():
    X_train = np.array([
        ["Egypt"],
        ["Germany"],
    ])

    X_test = np.array([
        ["Egypt", "Male"],
    ])

    encoder = OneHotEncoder()

    encoder.fit(X_train)

    with pytest.raises(ValueError):
        encoder.transform(X_test)


def test_invalid_handle_unknown():
    with pytest.raises(ValueError):
        OneHotEncoder(
            handle_unknown="something"
        )