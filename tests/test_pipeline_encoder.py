import numpy as np

from ml_from_scratch.pipeline.pipeline import Pipeline
from ml_from_scratch.preprocessing.encoder import OneHotEncoder
from ml_from_scratch.linear_models.logistic_regression import LogisticRegression


def test_pipeline_with_one_hot_encoder():
    X = np.array([
        ["Egypt"],
        ["Germany"],
        ["Egypt"],
        ["France"],
        ["Germany"],
        ["Egypt"],
    ])

    y = np.array([0, 1, 0, 1, 1, 0])

    pipeline = Pipeline([
        ("encoder", OneHotEncoder()),
        ("model", LogisticRegression(
            learning_rate=0.1,
            n_iterations=1000,
        )),
    ])

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert predictions.shape == y.shape
    assert np.all(np.isin(predictions, [0, 1]))


def test_pipeline_encoder_preserves_feature_count():
    X = np.array([
        ["Egypt", "Male"],
        ["Germany", "Female"],
        ["Egypt", "Female"],
        ["France", "Male"],
    ])

    y = np.array([0, 1, 0, 1])

    pipeline = Pipeline([
        ("encoder", OneHotEncoder()),
        ("model", LogisticRegression(
            learning_rate=0.1,
            n_iterations=1000,
        )),
    ])

    pipeline.fit(X, y)

    transformed = pipeline.transform(X)

    # Country: 3 categories
    # Gender: 2 categories
    assert transformed.shape == (4, 5)


def test_pipeline_unknown_category_ignore():
    X_train = np.array([
        ["Egypt"],
        ["Germany"],
        ["Egypt"],
    ])

    y_train = np.array([0, 1, 0])

    X_test = np.array([
        ["Japan"],
    ])

    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("model", LogisticRegression(
            learning_rate=0.1,
            n_iterations=1000,
        )),
    ])

    pipeline.fit(X_train, y_train)

    transformed = pipeline.transform(X_test)

    assert transformed.shape == (1, 2)
    np.testing.assert_array_equal(
        transformed,
        np.array([[0.0, 0.0]])
    )

def test_pipeline_does_not_leak_validation_categories():
    X_train = np.array([
        ["Egypt"],
        ["Germany"],
        ["Egypt"],
        ["Germany"],
    ])

    y_train = np.array([0, 1, 0, 1])

    X_validation = np.array([
        ["Japan"],
        ["Egypt"],
    ])

    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("model", LogisticRegression(
            learning_rate=0.1,
            n_iterations=1000,
        )),
    ])

    pipeline.fit(X_train, y_train)

    encoder = pipeline.fitted_steps_[0][1]

    # Japan must NOT have been learned during fit.
    np.testing.assert_array_equal(
        encoder.categories_[0],
        np.array(["Egypt", "Germany"])
    )

    transformed = pipeline.transform(X_validation)

    # Japan -> [0, 0]
    # Egypt -> [1, 0]
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ])
    )