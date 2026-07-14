from functools import lru_cache

import numpy as np

from arfcexp.schaefer_metadata import (
    get_schaefer_network_hemisphere_indices,
    get_schaefer_network_order,
)


OVERALL_NETWORK_LABEL = "all"


def _as_square_matrix(mat: np.ndarray, parc_size: int) -> np.ndarray:
    mat = np.asarray(mat)
    if mat.ndim == 2:
        if mat.shape != (parc_size, parc_size):
            raise ValueError(
                f"Expected matrix shape ({parc_size}, {parc_size}), got {mat.shape}."
            )
        return mat

    if mat.ndim != 1:
        raise ValueError(f"Expected 1D flattened or 2D matrix input, got ndim={mat.ndim}.")

    n_elements = len(mat)
    n = int(np.sqrt(n_elements))
    if n * n != n_elements:
        raise ValueError(f"Cannot reshape vector of length {n_elements} to square matrix.")
    if n != parc_size:
        raise ValueError(f"Matrix size {n} does not match requested parc_size={parc_size}.")

    return mat.reshape(n, n)


def _cartesian_edges(row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int32)
    cols = np.asarray(col_indices, dtype=np.int32)
    if rows.size == 0 or cols.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack(
        [np.repeat(rows, cols.size), np.tile(cols, rows.size)]
    ).astype(np.int32, copy=False)


def _coerce_edge_indices(edge_indices: np.ndarray, parc_size: int) -> np.ndarray:
    edges = np.asarray(edge_indices, dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected edge_indices with shape (n_edges, 2), got {edges.shape}.")
    if np.any(edges < 0) or np.any(edges >= parc_size):
        raise ValueError("edge_indices contains out-of-bounds parcel indices.")
    return edges


@lru_cache(maxsize=4)
def get_schaefer_all_crosshemisphere_edges(parc_size: int = 200) -> np.ndarray:
    """Return all LH-to-RH cross-hemisphere parcel edges."""
    network_indices = get_schaefer_network_hemisphere_indices(parc_size=parc_size)
    lh_indices = np.concatenate([groups["L"] for groups in network_indices.values()])
    rh_indices = np.concatenate([groups["R"] for groups in network_indices.values()])
    return _cartesian_edges(lh_indices, rh_indices)


@lru_cache(maxsize=4)
def get_schaefer_within_network_edges(parc_size: int = 200) -> dict[str, np.ndarray]:
    """Return LH-to-RH cross-hemisphere edges inside each Schaefer-7 network."""
    network_indices = get_schaefer_network_hemisphere_indices(parc_size=parc_size)
    return {
        network: _cartesian_edges(network_indices[network]["L"], network_indices[network]["R"])
        for network in get_schaefer_network_order(parc_size=parc_size)
    }


@lru_cache(maxsize=4)
def get_schaefer_between_network_edges(parc_size: int = 200) -> dict[str, np.ndarray]:
    """Return network-local between-network cross-hemisphere reference edges.

    For network ``N``, the LH-to-RH reference contains cross-hemisphere edges
    touching ``N`` on exactly one side: ``LH_N x RH_notN`` and
    ``LH_notN x RH_N``. This keeps per-network references tied to the network
    being scored while excluding the within-network block.
    """
    network_indices = get_schaefer_network_hemisphere_indices(parc_size=parc_size)
    all_lh = np.concatenate([groups["L"] for groups in network_indices.values()])
    all_rh = np.concatenate([groups["R"] for groups in network_indices.values()])

    out: dict[str, np.ndarray] = {}
    for network in get_schaefer_network_order(parc_size=parc_size):
        lh = network_indices[network]["L"]
        rh = network_indices[network]["R"]
        lh_not = np.setdiff1d(all_lh, lh, assume_unique=True)
        rh_not = np.setdiff1d(all_rh, rh, assume_unique=True)
        out[network] = np.vstack(
            [
                _cartesian_edges(lh, rh_not),
                _cartesian_edges(lh_not, rh),
            ]
        ).astype(np.int32, copy=False)
    return out


@lru_cache(maxsize=4)
def get_schaefer_overall_between_network_edges(parc_size: int = 200) -> np.ndarray:
    """Return all LH-to-RH cross-hemisphere edges outside same-network blocks."""
    all_edges = get_schaefer_all_crosshemisphere_edges(parc_size=parc_size)
    within_edges = np.vstack(
        [
            edges
            for edges in get_schaefer_within_network_edges(parc_size=parc_size).values()
            if len(edges) > 0
        ]
    )
    within_keys = set(map(tuple, within_edges.tolist()))
    keep = np.asarray([tuple(edge) not in within_keys for edge in all_edges], dtype=bool)
    return all_edges[keep].copy()


def _compute_percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Compute tie-aware percentile ranks in [0, 1] with NaN propagation."""
    vals = np.asarray(values, dtype=np.float64)
    out = np.full(vals.shape, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        return out

    finite_vals = vals[finite_mask]
    n = finite_vals.size
    if n == 1:
        out[finite_mask] = 0.5
        return out

    order = np.argsort(finite_vals, kind="mergesort")
    sorted_vals = finite_vals[order]
    ranks_sorted = np.empty(n, dtype=np.float64)

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1

        avg_rank = ((start + 1) + end) / 2.0
        ranks_sorted[start:end] = avg_rank
        start = end

    percentiles_sorted = (ranks_sorted - 1.0) / (n - 1)
    percentiles = np.empty(n, dtype=np.float64)
    percentiles[order] = percentiles_sorted
    out[finite_mask] = percentiles

    return out


def rank_matrix_offdiagonal(
    mat: np.ndarray,
    *,
    parc_size: int = 200,
    use_abs_rank: bool = True,
) -> np.ndarray:
    """Rank off-diagonal matrix entries into percentile space [0, 1]."""
    mat2d = _as_square_matrix(mat, parc_size=parc_size).astype(np.float64, copy=False)
    work = np.abs(mat2d) if use_abs_rank else mat2d

    n = work.shape[0]
    offdiag_mask = ~np.eye(n, dtype=bool)
    ranked = np.full_like(work, np.nan, dtype=np.float64)
    ranked[offdiag_mask] = _compute_percentile_ranks(work[offdiag_mask])
    return ranked


def _get_reduce_fn(reducer: str):
    if reducer == "mean":
        return np.nanmean
    if reducer == "median":
        return np.nanmedian
    raise ValueError(f"Unsupported reducer={reducer!r}. Expected 'mean' or 'median'.")


def _safe_reduce(values: np.ndarray, reduce_fn) -> float:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0 or not np.isfinite(vals).any():
        return np.nan
    return float(reduce_fn(vals))


def _safe_std(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    if finite.sum() < 2:
        return np.nan
    return float(np.nanstd(vals, ddof=1))


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0 or not np.isfinite(vals).any():
        return np.nan
    return float(np.nanpercentile(vals, percentile))


def extract_edge_values_directional(
    mat: np.ndarray,
    *,
    parc_size: int = 200,
    edge_indices: np.ndarray,
    use_abs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract values for explicit cross-hemisphere edges and their reverse direction."""
    mat2d = _as_square_matrix(mat, parc_size=parc_size)
    edges = _coerce_edge_indices(edge_indices, parc_size=parc_size)

    upper = mat2d[edges[:, 0], edges[:, 1]].astype(np.float64, copy=False)
    lower = mat2d[edges[:, 1], edges[:, 0]].astype(np.float64, copy=False)

    if use_abs:
        upper = np.abs(upper)
        lower = np.abs(lower)

    return upper, lower


def _summarize_edge_values(
    upper: np.ndarray,
    lower: np.ndarray,
    *,
    reducer: str,
    is_directed: bool,
) -> dict[str, float | bool]:
    reduce_fn = _get_reduce_fn(reducer)
    primary = np.concatenate([upper, lower])

    if is_directed:
        mean_upper = _safe_reduce(upper, reduce_fn)
        mean_lower = _safe_reduce(lower, reduce_fn)
        q25_upper = _safe_percentile(upper, 25)
        q75_upper = _safe_percentile(upper, 75)
        q25_lower = _safe_percentile(lower, 25)
        q75_lower = _safe_percentile(lower, 75)
    else:
        mean_upper = np.nan
        mean_lower = np.nan
        q25_upper = np.nan
        q75_upper = np.nan
        q25_lower = np.nan
        q75_lower = np.nan

    return {
        "mean": _safe_reduce(primary, reduce_fn),
        "std": _safe_std(primary),
        "q25": _safe_percentile(primary, 25),
        "q75": _safe_percentile(primary, 75),
        "mean_upper": mean_upper,
        "mean_lower": mean_lower,
        "q25_upper": q25_upper,
        "q75_upper": q75_upper,
        "q25_lower": q25_lower,
        "q75_lower": q75_lower,
        "is_directed": bool(is_directed),
    }


def summarize_homotopic_fc_ranked(
    mat: np.ndarray,
    *,
    parc_size: int = 200,
    edge_indices: np.ndarray,
    reducer: str = "mean",
    use_abs_rank: bool = True,
    is_directed: bool = False,
) -> dict[str, float | bool]:
    """Summarize within-network cross-hemisphere FC percentile ranks."""
    ranked = rank_matrix_offdiagonal(
        mat,
        parc_size=parc_size,
        use_abs_rank=use_abs_rank,
    )
    return summarize_homotopic_fc_ranked_edges(
        ranked,
        parc_size=parc_size,
        edge_indices=edge_indices,
        reducer=reducer,
        is_directed=is_directed,
    )


def summarize_homotopic_fc_ranked_edges(
    ranked_mat: np.ndarray,
    *,
    parc_size: int = 200,
    edge_indices: np.ndarray,
    reducer: str = "mean",
    is_directed: bool = False,
) -> dict[str, float | bool]:
    """Summarize within-network cross-hemisphere edges from a ranked matrix."""
    upper, lower = extract_edge_values_directional(
        ranked_mat,
        parc_size=parc_size,
        edge_indices=edge_indices,
        use_abs=False,
    )
    summary = _summarize_edge_values(upper, lower, reducer=reducer, is_directed=is_directed)
    return {
        "homotopic_score": summary["mean"],
        "homotopic_score_upper": summary["mean_upper"],
        "homotopic_score_lower": summary["mean_lower"],
        "is_directed": summary["is_directed"],
    }


def summarize_between_network_reference_fc_ranked(
    mat: np.ndarray,
    *,
    parc_size: int = 200,
    edge_indices: np.ndarray,
    reducer: str = "mean",
    use_abs_rank: bool = True,
    is_directed: bool = False,
) -> dict[str, float | bool]:
    """Summarize deterministic between-network cross-hemisphere reference ranks."""
    ranked = rank_matrix_offdiagonal(
        mat,
        parc_size=parc_size,
        use_abs_rank=use_abs_rank,
    )
    return summarize_between_network_reference_ranked_edges(
        ranked,
        parc_size=parc_size,
        edge_indices=edge_indices,
        reducer=reducer,
        is_directed=is_directed,
    )


def summarize_between_network_reference_ranked_edges(
    ranked_mat: np.ndarray,
    *,
    parc_size: int = 200,
    edge_indices: np.ndarray,
    reducer: str = "mean",
    is_directed: bool = False,
) -> dict[str, float | bool]:
    """Summarize deterministic between-network reference edges from a ranked matrix."""
    upper, lower = extract_edge_values_directional(
        ranked_mat,
        parc_size=parc_size,
        edge_indices=edge_indices,
        use_abs=False,
    )
    summary = _summarize_edge_values(upper, lower, reducer=reducer, is_directed=is_directed)
    return {
        "between_network_reference_mean": summary["mean"],
        "between_network_reference_std": summary["std"],
        "between_network_reference_q25": summary["q25"],
        "between_network_reference_q75": summary["q75"],
        "between_network_reference_mean_upper": summary["mean_upper"],
        "between_network_reference_mean_lower": summary["mean_lower"],
        "between_network_reference_q25_upper": summary["q25_upper"],
        "between_network_reference_q75_upper": summary["q75_upper"],
        "between_network_reference_q25_lower": summary["q25_lower"],
        "between_network_reference_q75_lower": summary["q75_lower"],
        "is_directed": summary["is_directed"],
    }
