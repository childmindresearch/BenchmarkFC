import numpy as np
from brainspace.gradient import LaplacianEigenmaps
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment


def fc_to_affinity(mat: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Threshold a functional connectivity matrix to get a sparse graph affinity.

    Assumes that large values mean more connected.

    Args:
        mat: functional connectivity matrix, shape (n_nodes, n_nodes)
        threshold: quantile sparsity threshold

    Returns:
        sparse non-negative affinity matrix, shape (n_nodes, n_nodes)
    """
    mat = 0.5 * (mat + mat.T)
    threshold = np.quantile(mat, threshold)
    aff = np.where(mat >= threshold, mat, 0.0)
    return aff


def fit_gradients(aff: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Fit principal gradients.

    Note, the matrix should be already thresholded so that it's sparse and non-negative.

    Args:
        aff: graph affinity matrix, shape (n_nodes, n_nodes)

    Returns:
        Array of gradients, shape (n_nodes, n_components)
    """
    embed = LaplacianEigenmaps(n_components, random_state=0)
    embed.fit(aff)
    grad = embed.maps_
    return grad


def calc_gradient_similarity(grad1: np.ndarray, grad2: np.ndarray) -> float:
    """Estimate the similarity between two sets of principal gradients.

    Computes the canonical correlation between the subspaces spanned by the two sets of
    principal gradients.

    Canonical correlation is defined as the mean of squared cosine principal angles.

    .. math::

        \text{Similarity} = \frac{1}{k} \sum_{i=1}^k \cos^2(\theta_i).

    Args:
        grad1: first principal gradients, shape (n_nodes, n_components)
        grad2: second principal gradients, shape (n_nodes, n_components)

    Returns:
        Gradient similarity score.
    """
    grad1, _ = np.linalg.qr(grad1)
    grad2, _ = np.linalg.qr(grad2)
    crossdot = grad1.T @ grad2
    s = np.linalg.svd(crossdot, compute_uv=False)
    return np.mean(s**2)


def calc_affinity_iou(aff1: np.ndarray, aff2: np.ndarray):
    """Calculate the intersection over union for two sparse affinity matrices.

    Note, the matrices should be sparse and non-negative.
    """
    mask1 = aff1 > 0
    mask2 = aff2 > 0
    iou = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
    return iou


def fit_clustering(aff: np.ndarray, n_clusters: int = 7) -> np.ndarray:
    """Cluster a graph affinity into networks using spectral clustering.

    Args:
        aff: graph affinity matrix, shape (n_nodes, n_nodes)
        n_clusters: number of clusters

    Returns:
        Array of cluster labels, shape (n_nodes,)
    """
    clust = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=0
    )
    clust.fit(aff)
    labels = clust.labels_
    return labels


def calc_cluster_similarity(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Calculate the similarity between two clusterings.

    Measures the cluster "accuracy" (i.e. fraction of samples with the same labels),
    up to permutation of the labels.

    Args:
        labels1: first cluster labels
        labels2: second cluster labels

    Returns:
        Cluster similarity score

    Note:
        You can also use any other cluster similarity metric, e.g. rand score or mutual
        information.
    """
    n_clusters = max(np.max(labels1), np.max(labels2)) + 1

    # Convert to one-hot
    src_one_hot = np.eye(n_clusters)[labels1]
    target_one_hot = np.eye(n_clusters)[labels2]

    # Compute confusion matrix
    confusion = src_one_hot.T @ target_one_hot

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(-confusion)
    accuracy = confusion[row_ind, col_ind].sum() / confusion.sum()
    return accuracy


def calc_graph_spectral_entropy(aff: np.ndarray, bins: int = 20) -> float:
    """Compute the graph spectral entropy (i.e. entropy of the eigenvalue distribution).

    Args:
        aff: graph affinity matrix, shape (n_nodes, n_nodes)
        bins: number of bins for estimating the eigenvalue histogram

    Returns:
        Spectral entropy score.

    References:
        https://www.rdocumentation.org/packages/statGraph/versions/0.5.0/topics/graph.entropy
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0049949
    """
    aff = (aff > 0).astype(int)
    evals = np.linalg.eigvalsh(aff)
    entropy = density_entropy(evals, bins=20)
    return entropy


def density_entropy(values: np.ndarray, bins: int = 100) -> float:
    """Compute the empirical entropy of a set of values."""
    prob, edges = np.histogram(values, bins=bins, density=True)
    width = np.mean(np.diff(edges))
    entropy = -width * np.sum(prob * np.log(prob + 1e-8))
    return entropy
