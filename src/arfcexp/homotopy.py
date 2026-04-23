from functools import lru_cache
from typing import Literal

import numpy as np


SUPPORTED_SCHAEFER_PARC_SIZES = (200,)


def _validate_parc_size(parc_size: int) -> None:
    if parc_size not in SUPPORTED_SCHAEFER_PARC_SIZES:
        raise ValueError(
            f"Unsupported Schaefer parcellation size {parc_size}. "
            f"Supported sizes: {SUPPORTED_SCHAEFER_PARC_SIZES}."
        )


@lru_cache(maxsize=4)
def get_schaefer_homotopic_pairs(
    parc_size: int = 200,
    *,
    zero_indexed: bool = True,
) -> np.ndarray:
    """Return Schaefer homotopic LH-RH ROI index pairs.

    For Schaefer 7-network CIFTI ordering, parcels are arranged as all LH parcels
    followed by all RH parcels. Homotopic pairs are therefore given by
    ``(i, i + parc_size / 2)`` for LH index ``i``.

    Args:
        parc_size: Number of parcels in the parcellation.
        zero_indexed: Whether to return indices in 0-based convention.

    Returns:
        Array of shape ``(parc_size // 2, 2)`` with ``(lh, rh)`` index pairs.
    """
    _validate_parc_size(parc_size)

    if parc_size % 2 != 0:
        raise ValueError(f"Expected an even parc_size, got {parc_size}.")

    half = parc_size // 2
    lh_idx = np.arange(half, dtype=np.int32)
    rh_idx = lh_idx + half
    pairs = np.stack([lh_idx, rh_idx], axis=1)

    if not zero_indexed:
        pairs = pairs + 1
    return pairs


def get_homotopic_partner_indices(parc_size: int = 200) -> np.ndarray:
    """Return partner index for every ROI index in Schaefer ordering.

    Args:
        parc_size: Number of parcels in the parcellation.

    Returns:
        Array ``partner`` where ``partner[i]`` is the homotopic index of ``i``.
    """
    _validate_parc_size(parc_size)

    half = parc_size // 2
    partner = np.empty(parc_size, dtype=np.int32)
    partner[:half] = np.arange(half, parc_size, dtype=np.int32)
    partner[half:] = np.arange(half, dtype=np.int32)
    return partner


def validate_schaefer_homotopic_pairs(parc_size: int = 200) -> bool:
    """Validate homotopic pair construction invariants."""
    pairs = get_schaefer_homotopic_pairs(parc_size=parc_size, zero_indexed=True)
    partner = get_homotopic_partner_indices(parc_size=parc_size)

    expected = parc_size // 2
    if len(pairs) != expected:
        return False

    for lh_idx, rh_idx in pairs:
        if lh_idx < 0 or rh_idx >= parc_size:
            return False
        if partner[lh_idx] != rh_idx:
            return False
        if partner[rh_idx] != lh_idx:
            return False
    return True


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


def extract_homotopic_values_directional(
    mat: np.ndarray,
    *,
    parc_size: int = 200,
    use_abs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract directional homotopic values from an FC matrix.

    Returns:
        Tuple ``(upper, lower)`` where:
        - ``upper`` is LH->RH values ``mat[i, i + parc_size // 2]``
        - ``lower`` is RH->LH values ``mat[i + parc_size // 2, i]``
    """
    mat2d = _as_square_matrix(mat, parc_size=parc_size)
    pairs = get_schaefer_homotopic_pairs(parc_size=parc_size, zero_indexed=True)

    upper = mat2d[pairs[:, 0], pairs[:, 1]].astype(np.float64, copy=False)
    lower = mat2d[pairs[:, 1], pairs[:, 0]].astype(np.float64, copy=False)

    if use_abs:
        upper = np.abs(upper)
        lower = np.abs(lower)

    return upper, lower


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

        # Average 1-based rank for the tied block.
        avg_rank = ((start + 1) + end) / 2.0
        ranks_sorted[start:end] = avg_rank
        start = end

    # Convert to percentile in [0, 1].
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
    """Rank off-diagonal matrix entries into percentile space [0, 1].

    Ranking is performed globally across all off-diagonal entries.
    Diagonal entries are set to NaN.
    """
    mat2d = _as_square_matrix(mat, parc_size=parc_size).astype(np.float64, copy=False)
    if use_abs_rank:
        work = np.abs(mat2d)
    else:
        work = mat2d

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


def _sample_non_partner_indices(
    n_pairs: int,
    *,
    rng: np.random.Generator,
    base_offset: int,
) -> np.ndarray:
    """Sample one non-partner index per homotopic pair.

    For pair i, partner offset is i. This returns sampled offsets j != i, then
    converts to absolute indices by adding ``base_offset``.
    """
    # Draw from [0, n_pairs - 2], then shift to skip each pair's partner index.
    draws = rng.integers(0, n_pairs - 1, size=n_pairs, dtype=np.int32)
    offsets = draws + (draws >= np.arange(n_pairs, dtype=np.int32))
    return offsets + base_offset


def summarize_heterotopic_null_fc_ranked(
    mat: np.ndarray,
    *,
    parc_size: int = 200,
    reducer: str = "mean",
    use_abs_rank: bool = True,
    is_directed: bool = False,
    n_perm: int = 200,
    seed: int = 2142,
) -> dict[str, float]:
    """Summarize heterotopic null FC using ranked off-diagonal percentiles.

    Null sampling strategy:
    - For each homotopic pair (L_i, R_i), sample one non-homotopic
      inter-hemispheric edge (L_i, R_j) with j != i.
    - Repeat ``n_perm`` times.
    - Aggregate each repetition with ``reducer`` and summarize the resulting
      distribution.

    For directed matrices, upper (L->R) and lower (R->L) null summaries are
    returned separately in addition to the combined summary.
    """
    if n_perm <= 0:
        raise ValueError(f"Expected n_perm > 0, got {n_perm}.")

    ranked = rank_matrix_offdiagonal(
        mat,
        parc_size=parc_size,
        use_abs_rank=use_abs_rank,
    )
    pairs = get_schaefer_homotopic_pairs(parc_size=parc_size, zero_indexed=True)
    n_pairs = pairs.shape[0]
    reduce_fn = _get_reduce_fn(reducer)

    rng = np.random.default_rng(seed)
    null_reps = np.empty(n_perm, dtype=np.float64)
    null_reps_upper = np.empty(n_perm, dtype=np.float64)
    null_reps_lower = np.empty(n_perm, dtype=np.float64)

    for rep in range(n_perm):
        sampled_rh = _sample_non_partner_indices(n_pairs, rng=rng, base_offset=n_pairs)
        sampled_lh = _sample_non_partner_indices(n_pairs, rng=rng, base_offset=0)

        upper = ranked[pairs[:, 0], sampled_rh]
        lower = ranked[pairs[:, 1], sampled_lh]

        null_reps_upper[rep] = float(reduce_fn(upper))
        null_reps_lower[rep] = float(reduce_fn(lower))
        null_reps[rep] = float(reduce_fn(np.concatenate([upper, lower])))

    mean = float(np.nanmean(null_reps))
    std = float(np.nanstd(null_reps, ddof=1)) if n_perm > 1 else np.nan
    q25 = float(np.nanpercentile(null_reps, 25))
    q75 = float(np.nanpercentile(null_reps, 75))

    if is_directed:
        mean_upper = float(np.nanmean(null_reps_upper))
        mean_lower = float(np.nanmean(null_reps_lower))
        q25_upper = float(np.nanpercentile(null_reps_upper, 25))
        q75_upper = float(np.nanpercentile(null_reps_upper, 75))
        q25_lower = float(np.nanpercentile(null_reps_lower, 25))
        q75_lower = float(np.nanpercentile(null_reps_lower, 75))
    else:
        mean_upper = np.nan
        mean_lower = np.nan
        q25_upper = np.nan
        q75_upper = np.nan
        q25_lower = np.nan
        q75_lower = np.nan

    return {
        "heterotopic_null_mean": mean,
        "heterotopic_null_std": std,
        "heterotopic_null_q25": q25,
        "heterotopic_null_q75": q75,
        "heterotopic_null_mean_upper": mean_upper,
        "heterotopic_null_mean_lower": mean_lower,
        "heterotopic_null_q25_upper": q25_upper,
        "heterotopic_null_q75_upper": q75_upper,
        "heterotopic_null_q25_lower": q25_lower,
        "heterotopic_null_q75_lower": q75_lower,
    }


def summarize_homotopic_fc_ranked(
    mat: np.ndarray,
    *,
    parc_size: int = 200,
    reducer: str = "mean",
    use_abs_rank: bool = True,
    is_directed: bool = False,
) -> dict[str, float | bool]:
    """Summarize homotopic FC using global off-diagonal percentile ranks.

    For directed matrices, this returns separate directional means for upper and
    lower homotopic edges as well as the combined ranked score.
    For undirected matrices, only the combined score is returned and directional
    scores are set to NaN.
    """
    ranked = rank_matrix_offdiagonal(
        mat,
        parc_size=parc_size,
        use_abs_rank=use_abs_rank,
    )

    upper, lower = extract_homotopic_values_directional(
        ranked,
        parc_size=parc_size,
        use_abs=False,
    )

    reduce_fn = _get_reduce_fn(reducer)

    if is_directed:
        score_upper = float(reduce_fn(upper))
        score_lower = float(reduce_fn(lower))
        score = float(reduce_fn(np.concatenate([upper, lower])))
    else:
        # For symmetric matrices, upper and lower are equivalent; use all values.
        score = float(reduce_fn(np.concatenate([upper, lower])))
        score_upper = np.nan
        score_lower = np.nan

    return {
        "homotopic_score": score,
        "homotopic_score_upper": score_upper,
        "homotopic_score_lower": score_lower,
        "is_directed": bool(is_directed),
    }

