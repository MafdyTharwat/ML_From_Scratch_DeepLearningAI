import numpy as np
import pytest

from ml_from_scratch.decomposition.pca import PCA


@pytest.fixture
def simple_data():
    return np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, 10.0],
    ])


@pytest.fixture
def three_feature_data():
    return np.array([
        [1.0, 2.0, 10.0],
        [2.0, 4.0, 20.0],
        [3.0, 6.0, 30.0],
        [4.0, 8.0, 40.0],
        [5.0, 10.0, 50.0],
    ])


def test_invalid_n_components():
    with pytest.raises(ValueError):
        PCA(n_components=0)

    with pytest.raises(ValueError):
        PCA(n_components=-1)

    with pytest.raises(ValueError):
        PCA(n_components=1.5)

    with pytest.raises(ValueError):
        PCA(n_components=True)


def test_fit_creates_expected_attributes(simple_data):
    model = PCA(n_components=1)

    model.fit(simple_data)

    assert hasattr(model, "mean_")
    assert hasattr(model, "components_")
    assert hasattr(model, "explained_variance_")
    assert hasattr(model, "explained_variance_ratio_")
    assert hasattr(model, "n_features_in_")


def test_mean_shape(simple_data):
    model = PCA(n_components=1)

    model.fit(simple_data)

    assert model.mean_.shape == (2,)


def test_components_shape(simple_data):
    model = PCA(n_components=1)

    model.fit(simple_data)

    assert model.components_.shape == (1, 2)


def test_explained_variance_shape(simple_data):
    model = PCA(n_components=1)

    model.fit(simple_data)

    assert model.explained_variance_.shape == (1,)


def test_explained_variance_ratio_shape(simple_data):
    model = PCA(n_components=1)

    model.fit(simple_data)

    assert model.explained_variance_ratio_.shape == (1,)


def test_explained_variance_ratio_is_valid(simple_data):
    model = PCA(n_components=2)

    model.fit(simple_data)

    assert np.all(model.explained_variance_ratio_ >= 0)
    assert np.all(model.explained_variance_ratio_ <= 1)

    np.testing.assert_allclose(
        np.sum(model.explained_variance_ratio_),
        1.0,
        atol=1e-10,
    )


def test_first_component_explains_most_variance(simple_data):
    model = PCA(n_components=2)

    model.fit(simple_data)

    assert (
        model.explained_variance_ratio_[0]
        >= model.explained_variance_ratio_[1]
    )


def test_components_are_orthonormal(three_feature_data):
    model = PCA(n_components=3)

    model.fit(three_feature_data)

    identity = (
        model.components_
        @ model.components_.T
    )

    np.testing.assert_allclose(
        identity,
        np.eye(3),
        atol=1e-10,
    )


def test_transform_shape(three_feature_data):
    model = PCA(n_components=2)

    transformed = model.fit_transform(
        three_feature_data
    )

    assert transformed.shape == (5, 2)


def test_fit_transform_matches_transform(simple_data):
    model = PCA(n_components=1)

    transformed_1 = model.fit_transform(
        simple_data
    )

    transformed_2 = model.transform(
        simple_data
    )

    np.testing.assert_allclose(
        transformed_1,
        transformed_2,
    )


def test_transformed_data_is_centered(simple_data):
    model = PCA(n_components=1)

    transformed = model.fit_transform(
        simple_data
    )

    np.testing.assert_allclose(
        np.mean(transformed, axis=0),
        np.zeros(1),
        atol=1e-10,
    )


def test_transform_before_fit(simple_data):
    model = PCA(n_components=1)

    with pytest.raises(ValueError):
        model.transform(simple_data)


def test_wrong_number_of_features(simple_data):
    model = PCA(n_components=1)

    model.fit(simple_data)

    X_wrong = np.array([
        [1.0, 2.0, 3.0],
    ])

    with pytest.raises(ValueError):
        model.transform(X_wrong)


def test_invalid_input_dimensions():
    model = PCA(n_components=1)

    X = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        model.fit(X)


def test_too_few_samples():
    model = PCA(n_components=1)

    X = np.array([
        [1.0, 2.0],
    ])

    with pytest.raises(ValueError):
        model.fit(X)


def test_empty_features():
    model = PCA(n_components=1)

    X = np.empty((5, 0))

    with pytest.raises(ValueError):
        model.fit(X)


def test_too_many_components(simple_data):
    model = PCA(n_components=3)

    with pytest.raises(ValueError):
        model.fit(simple_data)


def test_all_components_preserve_total_variance(simple_data):
    model = PCA(n_components=2)

    model.fit(simple_data)

    np.testing.assert_allclose(
        np.sum(model.explained_variance_ratio_),
        1.0,
        atol=1e-10,
    )


def test_principal_component_has_unit_length(
    simple_data,
):
    model = PCA(n_components=2)

    model.fit(simple_data)

    norms = np.linalg.norm(
        model.components_,
        axis=1,
    )

    np.testing.assert_allclose(
        norms,
        np.ones(2),
        atol=1e-10,
    )