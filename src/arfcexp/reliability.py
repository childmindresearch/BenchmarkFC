"""
Reliability metrics for functional connectivity matrices.
 
    - Edge-wise ICC                     (wraps arfcexp.icc.batch_icc)
    - Gradient canonical correlation    (wraps arfcexp.graph_metrics.calc_gradient_similarity)
    - Subject identifiability index     (Amico & Goni 2018 / Tian & Zalesky 2021)
    - Discriminability                  (Bridgeford et al. 2021)
 
All metrics require *multiple runs per subject* and measure consistency
across repetitions. By default all metrics operate on raw (un-sparsified)
matrices. Pass sparsity as a float in (0, 1) to apply thresholding before
computation.
"""
  
from itertools import combinations
 
import numpy as np
import pandas as pd
 
from arfcexp.graph_metrics import calc_gradient_similarity, fc_to_affinity, fit_gradients
from arfcexp.icc import batch_icc, ICCResult
from arfcexp.matrices import apply_sparsity, import_matrix
from typing import NamedTuple
from scipy.stats import zscore
import logging

logger = logging.getLogger(__name__)

class TestRetestData(NamedTuple):
    subs: list
    test: np.ndarray
    retest: np.ndarray

def _build_test_retest(
    run_df: pd.DataFrame,
    *,
    mat_col: str,
) -> TestRetestData | None:
    """Average runs within each session and build test/retest matrix arrays.

    Args:
        run_df: DataFrame with columns [sub, ses, <mat_col>].
        mat_col: column holding the flat matrix arrays.

    Returns:
        TestRetestData(subs, test, retest) where:
            subs: list of subject IDs with >= 2 sessions
            test: shape (n_subs, n_edges), session-1 averaged matrices
            retest: shape (n_subs, n_edges), session-2 averaged matrices
        Returns None if fewer than 2 subjects have >= 2 sessions.
    """
    ses_avg = (
        run_df.groupby(["sub", "ses"])[mat_col]
        .apply(lambda x: np.mean(np.stack(x.values), axis=0))
        .reset_index()
        .sort_values(["sub", "ses"])
    )

    sub_counts = ses_avg.groupby("sub").size()
    valid_subs = sub_counts[sub_counts >= 2].index
    if len(valid_subs) < 2:
        return None
    
    filtered = ses_avg[ses_avg["sub"].isin(valid_subs)]

    test = np.stack([
        filtered[filtered["sub"] == s].iloc[0][mat_col]
        for s in valid_subs
    ])
    retest = np.stack([
        filtered[filtered["sub"] == s].iloc[1][mat_col]
        for s in valid_subs
    ])
    return TestRetestData(subs=list(valid_subs), test=test, retest=retest)

# ------------------------------------------------------------------
# ICC
# ------------------------------------------------------------------

def build_icc_input(
    run_df: pd.DataFrame,
    *,
    mat_col: str = "mat",
) -> np.ndarray:
    """Stack per-run matrices into shape (n_edges, n_subs, n_runs).
 
    Subjects with fewer than the maximum number of runs are padded with NaN
    so that ``batch_icc`` handles them via listwise deletion.
 
    Args:
        run_df: DataFrame with columns [sub, ses, run, <mat_col>].
                Each entry in mat_col must be a flat array of identical length.
        mat_col: column holding the flat matrix arrays.
 
    Returns:
        Array of shape (n_edges, n_subs, n_runs).
    """
    df = run_df.sort_values(["sub", "ses", "run"]).reset_index(drop=True)
    subs = df["sub"].unique()
    n_edges = len(df[mat_col].iloc[0])
    n_runs = int(df.groupby("sub").size().max())
 
    icc_input = np.full((n_edges, len(subs), n_runs), np.nan)
    
    sub_codes, _ = pd.factorize(df["sub"], sort=False)
    run_idx = df.groupby("sub", sort=False).cumcount().values

    mats = np.stack(df[mat_col].values)
    icc_input[:, sub_codes, run_idx] = mats.T
 
    return icc_input
 
 
def compute_icc(
    run_df: pd.DataFrame,
    *,
    is_symmetric: bool = False,
    sparsity: float | None = None,
    mat_col: str = "mat",
) -> ICCResult:
    """Compute edge-wise ICC2 (two-way random effects, absolute agreement) across all runs.
 
    If sparsity is provided, thresholding is applied per-run before stacking so that ICC 
    reflects the reliability of the thresholded representation.
 
    Args:
        run_df: DataFrame with columns [sub, ses, run, <mat_col>].
        is_symmetric: whether matrices are symmetric. Only used if sparsity is not None.
        sparsity: sparsity level passed to ``apply_sparsity``.
        mat_col: column holding the raw flat matrix arrays.
 
    Returns:
        ``ICCResult`` named tuple (icc1 … icc3k), each an array over edges.
    """
    if sparsity is not None:
        transformed = run_df[mat_col].apply(
            lambda m: apply_sparsity(np.asarray(m), is_symmetric=is_symmetric, sparsity=sparsity)
        )
        effective_df = run_df.assign(**{mat_col: transformed})
    else:
        effective_df = run_df

    icc_input = build_icc_input(effective_df, mat_col=mat_col)
    return batch_icc(icc_input)

# ------------------------------------------------------------------
# Gradient Canonical Correlation
# ------------------------------------------------------------------

def _gradient_sim_one_sub(
    sub: str,
    mats: list[np.ndarray],
    *,
    affinity_threshold: float,
    n_components: int,
) -> dict:
    """Compute mean pairwise gradient similarity for a single subject."""
    if len(mats) < 2:
        return {"sub": sub, "gradient_similarity": np.nan}

    gradients = []
    for mat in mats:
        try:
            A = import_matrix(mat)
            aff = fc_to_affinity(A, threshold=affinity_threshold)
            grad = fit_gradients(aff, n_components=n_components)
            gradients.append(grad)
        except Exception:
            logger.warning(
                "Gradient computation failed for subject %s; skipping run.",
                sub,
                exc_info=True,
            )
            continue

    if len(gradients) < 2:
        return {"sub": sub, "gradient_similarity": np.nan}

    sims = [
        calc_gradient_similarity(g1, g2)
        for g1, g2 in combinations(gradients, 2)
    ]
    return {"sub": sub, "gradient_similarity": float(np.mean(sims))}
 
 
def compute_gradient_reliability(
    run_df: pd.DataFrame,
    *,
    affinity_threshold: float = 0.8,
    n_components: int = 2,
    mat_col: str = "mat",
) -> pd.DataFrame:
    """Mean pairwise gradient canonical correlation per subject.

    Sparsity is not applied here — ``fc_to_affinity`` handles graph sparsification 
    internally via its affinity threshold.
 
    Args:
        run_df: DataFrame with columns [sub, ses, run, <mat_col>].
            Each entry in mat_col is a raw (un-sparsified) flat
            or 2-D connectivity matrix.
        affinity_threshold: quantile threshold for ``fc_to_affinity``.
        n_components: number of gradient components for Laplacian Eigenmaps.
        mat_col: column holding the connectivity matrices.
 
    Returns:
        DataFrame with columns [sub, gradient_similarity].
    """
    rows = [
        _gradient_sim_one_sub(
            sub, grp[mat_col].tolist(),
            affinity_threshold=affinity_threshold,
            n_components=n_components,
        )
        for sub, grp in run_df.groupby("sub")
    ]
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# Subject Identifiability Index
# ------------------------------------------------------------------

def compute_identifiability(
    run_df: pd.DataFrame,
    *,
    is_symmetric: bool = False,
    sparsity: float | None = None,
    mat_col: str = "mat",
) -> dict:
    """
    Differential identifiability (Amico & Goni 2018 / Tian & Zalesky 2021).

    Steps:
    1. Optionally apply sparsity to each run matrix (skipped if sparsity=None).
    2. Average within each session (test = ses 1, retest = ses 2).
    3. Build N x N identifiability matrix A of Pearson correlations.
    4. I_self = mean(diagonal)
    5. I_others = mean(off-diagonal)
    6. I_diff = (I_self - I_others) * 100
    7. Success rate = fraction correctly identified (diagonal is max in column)

    Args:
        run_df: DataFrame with columns [sub, ses, run, <mat_col>].
        is_symmetric: whether matrices are symmetric. Only used if sparsity is not None.
        sparsity: sparsity level passed to ``apply_sparsity``. 
        mat_col: column holding the raw flat matrix arrays.

    Returns:
        Dict with keys: I_self, I_others, I_diff, success_rate, identifiability_matrix (NxN array).
    """
    if sparsity is not None:
        transformed = run_df[mat_col].apply(
            lambda m: apply_sparsity(np.asarray(m), is_symmetric=is_symmetric, sparsity=sparsity)
        )
        effective_df = run_df.assign(**{mat_col: transformed})
    else:
        effective_df = run_df

    result = _build_test_retest(effective_df, mat_col=mat_col)
    if result is None:
        return {"I_self": np.nan, "I_others": np.nan, "I_diff": np.nan,
                "identifiability_matrix": None, "success_rate": np.nan}

    subs, test, retest = result

    # N×N Pearson correlation matrix between all test/retest pairs
    A = zscore(test, axis=1) @ zscore(retest, axis=1).T / test.shape[1]
    A = np.clip(A, -1, 1)

    I_self = np.mean(np.diag(A))
    mask_off = ~np.eye(len(subs), dtype=bool)
    I_others = np.mean(A[mask_off])
    I_diff = (I_self - I_others) * 100

    success_rate = np.mean(np.argmax(A, axis=0) == np.arange(len(subs)))

    return {
        "I_self": I_self,
        "I_others": I_others,
        "I_diff": I_diff,
        "identifiability_matrix": A,
        "success_rate": success_rate,
    }

# ------------------------------------------------------------------
# Discriminability
# ------------------------------------------------------------------

def compute_discriminability(
    run_df: pd.DataFrame,
    *,
    is_symmetric: bool = False,
    sparsity: float | None = None,
    mat_col: str = "mat",
    chunk_size: int = 10,
) -> float:
    """
    Discriminability (Bridgeford et al., PLoS Computational Biology 2021).

    Steps:
    1. Optionally apply sparsity to each run matrix (skipped if sparsity=None).
    2. Average within each session (test = ses 1, retest = ses 2)
    3. Compute within-subject Euclidean distance for each subject:
          d_within[i] = ||test_i - retest_i||
    4. For each subject i, compare d_within[i] against all between-subject
       distances D[i,j] = ||test_i - retest_j|| where j != i
    5. Count correct: d_within[i] < D[i,j] scores 1, tie scores 0.5
    6. Discriminability = total correct / total comparisons
       Range: [0, 1]. 

    Args:
        run_df: DataFrame with columns [sub, ses, run, <mat_col>].
        is_symmetric: whether matrices are symmetric (for sparsity thresholding).
        sparsity: sparsity level passed to apply_sparsity.
        mat_col: column holding the raw flat matrix arrays.
        chunk_size: number of subjects to process at once.

    Returns:
        Discriminability scalar in [0, 1], or NaN if insufficient data.
    """
    df = run_df.copy()
    if sparsity is not None:
        df[mat_col] = df[mat_col].apply(
            lambda m: apply_sparsity(np.asarray(m), is_symmetric=is_symmetric, sparsity=sparsity)
        )

    result = _build_test_retest(df, mat_col=mat_col)
    if result is None:
        return np.nan

    subs, test, retest = result
    n = len(subs)

    d_within = np.linalg.norm(test - retest, axis=1)  # (n_subs,)

    correct = 0.0
    total = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = slice(start, end)
        chunk_n = end - start

        diff = test[chunk, np.newaxis, :] - retest[np.newaxis, :, :]
        D_chunk = np.linalg.norm(diff, axis=2)

        # mask out self-comparisons
        global_idx = np.arange(start, end)
        self_mask = global_idx[:, np.newaxis] == np.arange(n)[np.newaxis, :]

        d_within_chunk = d_within[chunk, np.newaxis]

        correct += np.sum((D_chunk > d_within_chunk) & ~self_mask, dtype=float)
        correct += 0.5 * np.sum((D_chunk == d_within_chunk) & ~self_mask, dtype=float)
        total += chunk_n * (n - 1)

    return float(correct / total) if total > 0 else np.nan