import numpy as np
import pytest

from ml_from_scratch.clustering.kmeans import KMeans


@pytest.fixture
def simple_data():
    return np.array([
        [1.0, 1.0],
        [1.5, 2.0],
        [2.0, 1.5],
        [8.0, 8.0],
        [8.5, 9.0],
        [9.0, 8.5],
    ])


def test_invalid_n_clusters():
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)

    with pytest.raises(ValueError):
        KMeans(n_clusters=1.5)

    with pytest.raises(ValueError):
        KMeans(n_clusters=True)


def test_invalid_max_iter():
    with pytest.raises(ValueError):
        KMeans(max_iter=0)

    with pytest.raises(ValueError):
        KMeans(max_iter=1.5)

    with pytest.raises(ValueError):
        KMeans(max_iter=False)


def test_invalid_tol():
    with pytest.raises(ValueError):
        KMeans(tol=0)

    with pytest.raises(ValueError):
        KMeans(tol=-1)


def test_fit_creates_expected_attributes(simple_data):
    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model.fit(simple_data)

    assert hasattr(model, "cluster_centers_")
    assert hasattr(model, "labels_")
    assert hasattr(model, "inertia_")
    assert hasattr(model, "n_iter_")
    assert hasattr(model, "n_features_in_")


def test_cluster_centers_shape(simple_data):
    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model.fit(simple_data)

    assert model.cluster_centers_.shape == (2, 2)


def test_labels_shape_and_values(simple_data):
    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model.fit(simple_data)

    assert model.labels_.shape == (6,)
    assert np.all(np.isin(model.labels_, [0, 1]))


def test_fit_predict(simple_data):
    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    labels = model.fit_predict(simple_data)

    assert labels.shape == (6,)
    assert np.all(np.isin(labels, [0, 1]))
    np.testing.assert_array_equal(
        labels,
        model.labels_,
    )


def test_predict_after_fit(simple_data):
    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model.fit(simple_data)

    X_new = np.array([
        [1.2, 1.3],
        [8.8, 8.7],
    ])

    predictions = model.predict(X_new)

    assert predictions.shape == (2,)
    assert np.all(np.isin(predictions, [0, 1]))


def test_predict_before_fit(simple_data):
    model = KMeans(n_clusters=2)

    with pytest.raises(ValueError):
        model.predict(simple_data)


def test_wrong_number_of_features(simple_data):
    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model.fit(simple_data)

    X_wrong = np.array([
        [1.0, 1.0, 1.0],
    ])

    with pytest.raises(ValueError):
        model.predict(X_wrong)


def test_empty_data():
    model = KMeans(n_clusters=2)

    X = np.empty((0, 2))

    with pytest.raises(ValueError):
        model.fit(X)


def test_too_many_clusters(simple_data):
    model = KMeans(
        n_clusters=10,
        random_state=42,
    )

    with pytest.raises(ValueError):
        model.fit(simple_data)


def test_reproducibility(simple_data):
    model_1 = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model_2 = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model_1.fit(simple_data)
    model_2.fit(simple_data)

    np.testing.assert_allclose(
        model_1.cluster_centers_,
        model_2.cluster_centers_,
    )

    np.testing.assert_array_equal(
        model_1.labels_,
        model_2.labels_,
    )

    assert model_1.inertia_ == model_2.inertia_
    assert model_1.n_iter_ == model_2.n_iter_


def test_inertia_is_non_negative(simple_data):
    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model.fit(simple_data)

    assert model.inertia_ >= 0


def test_inertia_computation():
    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [10.0, 0.0],
        [12.0, 0.0],
    ])

    labels = np.array([0, 0, 1, 1])

    centroids = np.array([
        [1.0, 0.0],
        [11.0, 0.0],
    ])

    model = KMeans(n_clusters=2)

    inertia = model._compute_inertia(
        X,
        labels,
        centroids,
    )

    assert inertia == 4.0


def test_n_iter_is_valid(simple_data):
    model = KMeans(
        n_clusters=2,
        max_iter=20,
        random_state=42,
    )

    model.fit(simple_data)

    assert 1 <= model.n_iter_ <= 20