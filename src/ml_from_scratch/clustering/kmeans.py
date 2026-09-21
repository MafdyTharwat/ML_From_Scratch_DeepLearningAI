import numpy as np
from ..base import BaseEstimator

class KMeans:
    _parameter_names = ['n_clusters', 'max_iter', 'tol', 'random_state']
    def __init__(self, n_clusters = 3, max_iter = 100, tol = 1e-4, random_state = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
            raise ValueError('n_clusters must be an integer')
        if n_clusters < 1:
            raise ValueError('n_clusters must be at least 1')
        if not isinstance(max_iter, int) or isinstance(max_iter, bool):
            raise ValueError('max_iter must be an integer')
        if max_iter < 1:
            raise ValueError('max_iter must be at least 1')
        if tol <= 0:
            raise ValueError('tol must be greater than 0')

    def _initialize_centroids(self, X, rng):
        indices = rng.choice(X.shape[0], size = self.n_clusters, replace = False)
        return X[indices].copy()

    def _compute_distances(self, X, centroids):
        distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis = 2,)
        return distances
    
    def _assign_clusters(self, X, centroids):
        distances = self._compute_distances(X, centroids)
        return np.argmin(distances, axis = 1)

    def _update_centroids(self, X, labels, rng):
        centroids = np.zeros((self.n_clusters, X.shape[1]), dtype = float,)

        for cluster in range(self.n_clusters):
            cluster_points = X[labels == cluster]
            if len(cluster_points) == 0:
                centroids[cluster] = X[rng.integers(0, X.shape[0])]
            else:
                centroids[cluster] = np.mean(cluster_points, axis = 0,)

        return centroids

    def _compute_inertia(self, X, labels, centroids):
        distances = X - centroids[labels]
        return np.sum(distances ** 2)

    def fit(self, X):
        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[0] == 0:
            raise ValueError('X must not be empty')

        if self.n_clusters > X.shape[0]:
            raise ValueError('n_clusters can\'t be greater than the number of samples')

        self.n_features_in_ = X.shape[1]

        rng = np.random.default_rng(self.random_state)

        centroids = self._initialize_centroids(X, rng)

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)

            new_centroids = self._update_centroids(X, labels, rng,)

            centroid_shift = np.linalg.norm(new_centroids - centroids)

            centroids = new_centroids

            if centroid_shift <= self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.cluster_centers_ = centroids
        self.labels_ = self._assign_clusters(X, self.cluster_centers_,)

        self.inertia_ = self._compute_inertia(X, self.labels_, self.cluster_centers_,)

        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError('Model must be fitted before prediction')

        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if X.shape[1] != self.n_features_in_:
            raise ValueError('X must have the same number of features as the training data')

        return self._assign_clusters(X, self.cluster_centers_,)
        
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_