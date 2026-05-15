"""
Graph and matrix utilities for functional connectivity analysis.
 
Shared primitives (import_matrix, sparse graph builders) are defined here
and imported by reliability.py and info_density.py.
"""

import networkx as nx
import numpy as np
from brainspace.gradient import LaplacianEigenmaps
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment

from arfcexp.matrices import import_matrix
 
 
def _extract_triangle_as_symmetric(A: np.ndarray, triangle: str) -> np.ndarray:
    """Extract one triangle of A and mirror it to make a symmetric matrix.
 
    For an asymmetric matrix A[i,j] != A[j,i], the upper triangle captures
    the i->j direction and the lower triangle captures the j->i direction.
    Each triangle is then mirrored to produce a valid symmetric undirected graph.
 
    Args:
        A: square matrix, shape (n, n).
        triangle: "upper" uses A[i,j] for i < j;
                  "lower" uses A[j,i] for i < j (i.e. the lower triangle values).
 
    Returns:
        Symmetric matrix, shape (n, n).
    """
    n = A.shape[0]
    S = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    if triangle == "upper":
        S[iu] = A[iu]
    else:
        # lower triangle: A[j, i] for each (i, j) pair where i < j
        S[iu] = A[iu[1], iu[0]]
    return S + S.T

def sparse_undirected_graph(
    matrix: np.ndarray,
    sparsity: float = 0.0,
    triangle: str = "upper",
) -> tuple[nx.Graph, float, float]:
    """Build a sparse weighted undirected graph from one triangle of a matrix.
 
     For symmetric matrices both triangles are identical so ``triangle`` has no
    effect. For asymmetric matrices, pass ``triangle="upper"`` or
    ``triangle="lower"`` to compute metrics on each direction separately.

    Keeps the top ``(1 - sparsity)`` fraction of directed edges by absolute weight.
 
    Args:
        matrix: connectivity matrix (flat or 2-D).
        sparsity: fraction of edges to zero out (default 0.0)
        triangle: "upper" or "lower" - which triangle to use
 
    Returns:
        G: weighted undirected NetworkX graph
        abs_thr: absolute weight threshold applied
        density: resulting graph density
    """
    A = import_matrix(matrix)
    W = _extract_triangle_as_symmetric(A, triangle=triangle)
    n = W.shape[0]
 
    m_keep = int(round((1.0 - sparsity) * n * (n - 1) // 2))
    iu = np.triu_indices(n, k=1)
    w_abs = np.abs(W[iu])
    valid = w_abs > 0
 
    if m_keep <= 0 or not np.any(valid):
        return nx.empty_graph(n), np.inf, 0.0
 
    w_abs_valid = w_abs[valid]
    if w_abs_valid.size <= m_keep:
        selected = np.where(valid)[0]
        abs_thr = float(w_abs_valid.min())
    else:
        top_idx = np.argpartition(w_abs_valid, -m_keep)[-m_keep:]
        selected = np.where(valid)[0][top_idx]
        abs_thr = float(w_abs_valid[top_idx].min())
 
    B = np.zeros((n, n))
    ii, jj = iu[0][selected], iu[1][selected]
    B[ii, jj] = W[ii, jj]
    B[jj, ii] = W[jj, ii]
 
    G = nx.from_numpy_array(B)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G, abs_thr, nx.density(G)
 
 
def sparse_directed_graph(
    matrix: np.ndarray,
    sparsity: float = 0.0,
) -> tuple[nx.DiGraph, float, float]:
    """Build a sparse weighted directed graph from a connectivity matrix.
 
    Keeps the top ``(1 - sparsity)`` fraction of directed edges by absolute weight.
 
    Returns:
        G: weighted directed NetworkX graph
        abs_thr: absolute weight threshold applied
        density: resulting graph density
    """
    A = import_matrix(matrix)
    n = A.shape[0]
 
    m_keep = int(round((1.0 - sparsity) * n * (n - 1)))
    ii, jj = np.where(~np.eye(n, dtype=bool))
    w_abs = np.abs(A[ii, jj])
    valid = w_abs > 0
 
    if m_keep <= 0 or not np.any(valid):
        return nx.empty_graph(n, create_using=nx.DiGraph), np.inf, 0.0
 
    w_abs_valid = w_abs[valid]
    if w_abs_valid.size <= m_keep:
        selected = np.where(valid)[0]
        abs_thr = float(w_abs_valid.min())
    else:
        top_idx = np.argpartition(w_abs_valid, -m_keep)[-m_keep:]
        selected = np.where(valid)[0][top_idx]
        abs_thr = float(w_abs_valid[top_idx].min())
 
    B = np.zeros((n, n))
    si, sj = ii[selected], jj[selected]
    B[si, sj] = A[si, sj]
 
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G, abs_thr, nx.density(G)


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
