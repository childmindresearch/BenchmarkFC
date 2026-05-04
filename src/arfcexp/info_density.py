"""
Information density metrics for functional connectivity matrices.
 
All functions operate on a *single averaged matrix* — they measure
intrinsic properties of the graph, not consistency across repetitions.
 
Metrics
-------
    compute_sve - singular value entropy
    compute_stable_rank - stable rank 
    compute_rich_club - rich-club coefficient + scalar summaries
    compute_small_worldness - small-world sigma (C/Cr) / (L/Lr)
    compute_tsp_cost - approximate TSP tour cost
    compute_trophic_incoherence - trophic incoherence parameter q
    compute_core_depth - maximum k-core number
 
    compute_all - wrapper returning a flat dict of all scalar
        metrics for one matrix
"""
  
from typing import NamedTuple
 
import networkx as nx
import numpy as np
from networkx.algorithms.approximation import traveling_salesman
from scipy.stats import entropy
 
from arfcexp.graph_metrics import sparse_directed_graph, sparse_undirected_graph
from arfcexp.matrices import collapse_maxabs, import_matrix


def compute_sve(matrix: np.ndarray) -> float:
    """Singular value entropy (bits).
 
    Measures how evenly the variance is spread across singular values —
    higher entropy means more distributed, lower-rank structure.
 
    Args:
        matrix: connectivity matrix (flat or 2-D).
 
    Returns:
        Singular value entropy in bits.
    """
    A = import_matrix(matrix)
    S = np.linalg.svd(A, compute_uv=False)
    S = S[S > 0]
    if S.size == 0:
        return 0.0
    p = S / S.sum()
    return float(entropy(p, base=2))

def compute_stable_rank(matrix: np.ndarray, eps: float = 1e-12) -> float:
    """Stable rank = ||M||_F² / ||M||_2²
 
    Continuous approximation of matrix rank; robust to small singular values.
 
    Args:
        matrix: connectivity matrix (flat or 2-D).
        eps: floor for the spectral norm to avoid division by zero.
 
    Returns:
        Stable rank (≥ 1).
    """
    A = import_matrix(matrix)
    frob_sq = float(np.linalg.norm(A, ord="fro") ** 2)
    spec_sq = float(np.linalg.norm(A, ord=2) ** 2)
    return frob_sq / spec_sq if spec_sq > eps else 0.0

class RichClubResult(NamedTuple):
    """Scalar summaries of the rich-club coefficient curve.
 
    Attributes:
        auc: area under rho(k) vs k.
        k_at_max: degree k where rho(k) is maximal.
        sig_k_range: max(k) - min(k) where rho(k) > sig_thr (0 if none).
        mean_rho: mean rho(k) across all k.
        rc_dict: raw {degree: rho} dictionary from NetworkX.
    """
    auc: float
    k_at_max: float
    sig_k_range: float
    mean_rho: float
    rc_dict: dict
 
 
def compute_rich_club(
    matrix: np.ndarray,
    *,
    sparsity: float = 0.8,
    normalized: bool = True,
    seed: int = 70,
    Q: int = 10,
    sig_thr: float = 1.0,
) -> RichClubResult:
    """Rich-club coefficient and scalar summaries.
 
    Args:
        matrix: connectivity matrix (flat or 2-D).
        sparsity: sparsity level for graph thresholding.
        normalized: whether to normalise rho(k) by random graph expectation.
        seed: RNG seed for random graph generation.
        Q: number of edge swaps per edge for random graph.
        sig_thr: significance threshold for ``sig_k_range`` (default 1.0).
 
    Returns:
        :class:`RichClubResult` named tuple.
    """
    G, _, _ = sparse_undirected_graph(matrix, sparsity=sparsity)
 
    if G.number_of_edges() == 0:
        return RichClubResult(np.nan, np.nan, np.nan, np.nan, {})
 
    try:
        rc = nx.rich_club_coefficient(G, normalized=normalized, seed=seed, Q=Q)
    except (ZeroDivisionError, nx.NetworkXError):
        rc = nx.rich_club_coefficient(G, normalized=False)
 
    if not rc:
        return RichClubResult(np.nan, np.nan, np.nan, np.nan, {})
 
    ks = np.array(sorted(rc), dtype=float)
    vals = np.array([rc[k] for k in ks], dtype=float)
    good = np.isfinite(ks) & np.isfinite(vals)
    ks, vals = ks[good], vals[good]
 
    if ks.size == 0:
        return RichClubResult(np.nan, np.nan, np.nan, np.nan, rc)
 
    auc = float(np.trapz(vals, ks))
    k_at_max = float(ks[int(np.argmax(vals))])
    sig_mask = vals > sig_thr
    sig_k_range = float(ks[sig_mask].max() - ks[sig_mask].min()) if sig_mask.any() else 0.0
    mean_rho = float(np.mean(vals))
 
    return RichClubResult(auc, k_at_max, sig_k_range, mean_rho, rc)


def compute_small_worldness(
    matrix: np.ndarray,
    *,
    sparsity: float = 0.8,
    seed: int = 70,
    nrand: int = 5,
    nswap: int = 2000,
    use_lcc: bool = True,
) -> float:
    """Small-world sigma = (C / C_r) / (L / L_r).
 
    σ > 1 -> small-world topology.
    σ ≈ 1 -> random-like.
    σ < 1 -> not small-world.
 
    Args:
        matrix: connectivity matrix (flat or 2-D).
        sparsity: sparsity level for graph thresholding.
        seed: RNG seed for random reference graphs.
        nrand: number of random reference graphs to average over.
        nswap: number of edge swaps per randomisation.
        use_lcc: restrict to the largest connected component when the graph
            is disconnected (required for shortest-path computation).
 
    Returns:
        Small-world sigma, or NaN if the graph is degenerate.
    """
    G, _, _ = sparse_undirected_graph(matrix, sparsity=sparsity)
 
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return float("nan")
 
    if use_lcc and not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
 
    C = nx.transitivity(G)
    L = nx.average_shortest_path_length(G)
 
    rng = np.random.default_rng(seed)
    Crs, Lrs = [], []
    for _ in range(nrand):
        Gr = G.copy()
        nx.double_edge_swap(
            Gr, nswap=nswap, max_tries=nswap * 20,
            seed=int(rng.integers(0, 2 ** 31 - 1)),
        )
        if use_lcc and not nx.is_connected(Gr):
            Gr = Gr.subgraph(max(nx.connected_components(Gr), key=len)).copy()
        if Gr.number_of_nodes() < 3 or Gr.number_of_edges() == 0:
            continue
        Crs.append(nx.transitivity(Gr))
        Lrs.append(nx.average_shortest_path_length(Gr))
 
    if not Crs:
        return float("nan")
 
    Cr, Lr = float(np.mean(Crs)), float(np.mean(Lrs))
    if Cr == 0 or L == 0 or Lr == 0:
        return float("nan")
 
    return float((C / Cr) / (L / Lr))


def compute_tsp_cost(
    matrix: np.ndarray,
    *,
    nodes: list[int] | None = None,
) -> float:
    """Approximate TSP tour cost on a distance graph derived from FC weights.
 
    Edge distances are ``1 - |w|``, so strongly connected pairs are "close".
    A shorter tour -> more structured, information-dense connectivity.
 
    Args:
        matrix: connectivity matrix (flat or 2-D).
        nodes: subset of node indices (defaults to all nodes).
 
    Returns:
        Total tour length.
    """
    A = import_matrix(matrix)
    W = collapse_maxabs(A)
    n = W.shape[0]
    if nodes is None:
        nodes = list(range(n))
 
    D = 1.0 - np.abs(W)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i_idx, i in enumerate(nodes):
        for j in nodes[i_idx + 1:]:
            G.add_edge(i, j, weight=max(float(D[i, j]), 0.0))
 
    tour = traveling_salesman.traveling_salesman_problem(G, nodes=nodes, cycle=True)
    cost = sum(G[u][v]["weight"] for u, v in zip(tour, tour[1:]))
    if tour[0] != tour[-1]:
        cost += G[tour[-1]][tour[0]]["weight"]
    return float(cost)

def compute_trophic_incoherence(
    matrix: np.ndarray,
    *,
    sparsity: float = 0.8,
    absolute_weights: bool = False,
) -> float:
    """Trophic incoherence parameter q of the sparse directed graph.
 
    Args:
        matrix: connectivity matrix (flat or 2-D).
        sparsity: sparsity level for directed graph thresholding.
        absolute_weights: if True, use |weight| - removes sign-driven outliers.
 
    Returns:
        Trophic incoherence parameter q, or NaN if the graph has no edges.
    """
    G, _, _ = sparse_directed_graph(matrix, sparsity=sparsity)
    if G.number_of_edges() == 0:
        return float("nan")
    if absolute_weights:
        for _, _, d in G.edges(data=True):
            d["weight"] = abs(float(d["weight"]))
    try:
        return float(nx.trophic_incoherence_parameter(G, weight="weight"))
    except Exception:
        return float("nan")
    

def compute_core_depth(matrix: np.ndarray, *, sparsity: float = 0.8) -> float:
    """Maximum k-core number of the sparse undirected graph.
 
    Higher values indicate a denser, more hierarchically organised core.
 
    Args:
        matrix: connectivity matrix (flat or 2-D).
        sparsity: sparsity level for graph thresholding.
 
    Returns:
        Maximum core number (0 if graph is empty).
    """
    G, _, _ = sparse_undirected_graph(matrix, sparsity=sparsity)
    core_numbers = nx.core_number(G)
    return float(max(core_numbers.values())) if core_numbers else 0.0


def compute_all(matrix, *, sparsity=0.8, tsp_nodes=None, 
                small_world_kwargs=None, rich_club_kwargs=None):
    sw_kw = small_world_kwargs or {"seed": 70, "nrand": 1, "nswap": 2000}
    rc_kw = rich_club_kwargs or {}
    rc = compute_rich_club(matrix, sparsity=0.0, **rc_kw)

    return {
        "sv_entropy": compute_sve(matrix),
        "stable_rank": compute_stable_rank(matrix),
        "rich_club_auc": rc.auc,
        "rich_club_k_at_max": rc.k_at_max,
        "rich_club_sig_k_range": rc.sig_k_range,
        "rich_club_mean_rho": rc.mean_rho,
        "small_worldness": compute_small_worldness(matrix, sparsity=0.0, **sw_kw),
        "tsp_cost": compute_tsp_cost(matrix, nodes=tsp_nodes),
        "trophic_incoherence": compute_trophic_incoherence(matrix, sparsity=0.0),
        "trophic_incoherence_abs": compute_trophic_incoherence(matrix, sparsity=0.0, 
                                                                absolute_weights=True),
        "core_depth": compute_core_depth(matrix, sparsity=0.0),
    }