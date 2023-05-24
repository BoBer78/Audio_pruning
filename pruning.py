import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
from matplotlib.pyplot import cm
from sklearn.inspection import DecisionBoundaryDisplay

import gc
import csv
import os


class Pruning:
    """Generic class for pruning a given dataset based on a dimensional reduction and clustering analysis."""

    def __init__(self, X: str, y: str):
        self.X = self._read(X)
        self.y = self._read(y)
        self.memory_management = True
        self.stacker(
            frac=1.0
        )  # we call stacker directly in the constructor. Needed anyways.

    def _read(self, filename) -> np.ndarray:
        """Reads a .npy binary array"""
        return np.load(filename, allow_pickle=True)

    def stacker(self, randomize=False, frac=1.0) -> np.ndarray:
        """Builds a matrix in which each column is a vectorised sample."""
        # reduce a redundant dimension
        if len(self.X.shape) > 3:
            self.X = self.X[:, 0, :, :]
        nsamples = self.X.shape[0]
        samples = np.arange(nsamples)

        # randomize the samples
        if randomize:
            np.random.shuffle(samples)
        # take a fraction of all samples
        if not np.isclose(frac, 1.0):
            nsamples = int(nsamples * frac)
            samples = samples[:nsamples]

        self.M = np.zeros(
            (len(self.X[0, :, :].reshape(-1)), nsamples), dtype=np.float16
        )
        for n, sample in enumerate(samples):
            self.M[:, n] = self.X[sample].reshape(-1)

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        self.M = scaler.fit_transform(self.M)

        if self.memory_management:
            self.X = 0
            gc.collect()

    def reduce_PCA(self, **params):
        """Reduce the matrix to two-column array with PCA. **params passed according to https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html."""
        if not hasattr(self, "M"):
            self._stacker()
        self.PCA_ = PCA(**params).fit(self.M.T)
        self.M_transformed = self.PCA_.transform(self.M.T)

        if self.memory_management:
            self.M = 0
            gc.collect()

    def reduce_tSNE(self, **params):
        """Reduce the matrix to two-column array with t-SNE. **params passed according to https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html."""
        if not hasattr(self, "M"):
            self._stacker()
        tsne = manifold.TSNE(n_components=2, **params)
        self.M_transformed = tsne.fit_transform(self.M.T)

        if self.memory_management:
            self.M = 0
            gc.collect()

    def cluster_kmeans(self, **params):
        """K-Means clustering. **params passed according to https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html."""
        if not hasattr(self, "M"):
            raise ValueError("You need to use a dimensional reduction first! Use reduce_PCA() or reduce_tSNE() function.")
        self.kmeans = KMeans(**params).fit(self.M_transformed)
        self.y_kmeans = self.kmeans.predict(self.M_transformed)
        self.centers = self.kmeans.cluster_centers_
        return self.kmeans, self.y_kmeans, self.centers

    def cluster_kmeans_full(self, **params):
        """K-Means clustering on the complete dataset, not on a reducion"""
        if not hasattr(self, "M"):
            raise ValueError("You need to stack first")
        self.kmeans = KMeans(**params).fit(self.M.T)
        self.y_kmeans = self.kmeans.predict(self.M.T)
        self.centers = self.kmeans.cluster_centers_

        return self.kmeans, self.y_kmeans, self.centers

    def remove_points_from_clusters(self, label=-1, frac=1.0, order="smallest") -> np.ndarray:
        """For each cluster (labeled according to the y_kmeans), the distance (L2) between the center and every point in the cluster is computed in order to identify which points are close or far from the center.
        :params: frac - defines the fraction of a data (from [0, 1] range) that will be removed.
        :params: order - defines whether the points closest to the center ('smallest') or the farthest 'largest') will be removed.
        If label = -1, the code performs the analysis for all the labels (constrained by n_clusters parameter in kmeans).
        """

        if not hasattr(self, "kmeans"):
            raise ValueError("Call the function cluster_kmeans() before analyzing the clusters!")
        if order not in ["smallest", "largest"]:
            raise ValueError("Wrong order: valid options are smaller or largest!")
        labels = [label] if not label == -1 else np.arange(len(self.centers))

        # list for keeping the indices of relevant entries
        reduced = []

        for l in labels:
            # find the points labeled by l
            idx = np.flatnonzero(self.y_kmeans == l)
            center = self.centers[l]
            clusters = self.M_transformed[idx, :]
            # compute the norm between the center and all the points in a cluster
            norms = np.linalg.norm(center - clusters, axis=1)
            # find the indices depending on the norm in an ascending order
            norms_idx_sorted = np.argsort(norms)
            cutoff = int(frac * len(clusters))
            if order == "smallest":
                reduced.append(idx[norms_idx_sorted[cutoff:]])
            else:
                reduced.append(idx[norms_idx_sorted[:-cutoff]])
        return np.concatenate(reduced)

    def remove_points_from_clusters_full(self, label=-1, frac=1.0, order="smallest") -> np.ndarray:
        """same function as "remove_points_from_clusters", works on the full dimension"""

        if not hasattr(self, "kmeans"):
            raise ValueError("Call the function cluster_kmeans() before analyzing the clusters!" )
        if order not in ["smallest", "largest"]:
            raise ValueError("Wrong order: valid options are smaller or largest!")
        labels = [label] if not label == -1 else np.arange(len(self.centers))

        # list for keeping the indices of relevant entries
        reduced = []

        for l in labels:
            # find the points labeled by l
            idx = np.flatnonzero(self.y_kmeans == l)
            center = self.centers[l]
            clusters = self.M.T[idx, :]
            # compute the norm between the center and all the points in a cluster
            norms = np.linalg.norm(center - clusters, axis=1)
            # find the indices depending on the norm in an ascending order
            norms_idx_sorted = np.argsort(norms)
            cutoff = int(frac * len(clusters))
            if order == "smallest":
                reduced.append(idx[norms_idx_sorted[cutoff:]])
            else:
                reduced.append(idx[norms_idx_sorted[:-cutoff]])
        return np.concatenate(reduced)


# ==================================================================================
# Init the pruning class.
# ==================================================================================

K = 10
current_path = os.getcwd()
x_path = os.path.join(current_path, 'wav2vec_features', "x_wav2vec.npy")
y_path = os.path.join(current_path, 'wav2vec_features', "y_wav2vec.npy")

pruning = Pruning(X=x_path, y=y_path)
print("init done")

# ==================================================================================
# K-means on the complete data + pruning with different ratios
# ==================================================================================

kmeans, y_kmeans, centers = pruning.cluster_kmeans_full(n_clusters=K)
print("full k-means done")

to_reduce = [0.05, 0.1, 0.15, 0.2, 0.3]

for ratio in to_reduce:
    simple_pruning = pruning.remove_points_from_clusters_full(label=-1, frac=ratio, order="smallest")
    np.save(
        os.path.join(
            current_path,"y_pruned", "y_simple_pruning_FULL_k{}_{}.npy".format(K, ratio),
        ),
        simple_pruning,
    )

for ratio in to_reduce:
    hard_pruning = pruning.remove_points_from_clusters_full(label=-1, frac=ratio, order="largest")
    np.save(
        os.path.join(
            current_path, "y_pruned", "y_hard_pruning_FULL_k{}_{}.npy".format(K, ratio)
        ),
        hard_pruning,
    )

# ==================================================================================
# K-means after PCA only done for plots and testing. (can be commented, not necessary)
# ==================================================================================

pruning.reduce_PCA(n_components = 2, whiten=True)
print('PCA done')

kmeans, y_kmeans, centers = pruning.cluster_kmeans(n_clusters = K)
print('k-means done')

reduced_hard = pruning.remove_points_from_clusters(label = -1, frac = 0.5, order = 'smallest')
reduced_easy = pruning.remove_points_from_clusters(label = -1, frac = 0.5, order = 'largest')

X_projected = pruning.M_transformed
proj_centers = centers

fig, axes = plt.subplots(ncols = 2, nrows = 2, sharey = True, sharex = True)
axes = axes.flat

axes[0].scatter(X_projected[:, 0], X_projected[:, 1], c = y_kmeans, s = 2, cmap = 'viridis')
axes[0].scatter(proj_centers[:, 0] ,proj_centers[:, 1], marker = 'x', color = 'k')
axes[0].set_title('full data')

axes[2].scatter(X_projected[reduced_hard , 0], X_projected[reduced_hard, 1], c = y_kmeans[reduced_hard], s = 2, cmap = 'viridis')
axes[2].scatter(proj_centers[:, 0] ,proj_centers[:, 1], marker = 'x', color = 'k')
axes[2].set_title('removing 30$\%$ points close to centroids')

axes[3].scatter(X_projected[reduced_easy , 0], X_projected[reduced_easy, 1], c = y_kmeans[reduced_easy], s = 2, cmap = 'viridis')
axes[3].scatter(proj_centers[:, 0] ,proj_centers[:, 1], marker = 'x', color = 'k')
axes[3].set_title('removing 30$\%$ points far from centroids')
plt.show()
