"""
Reliability metrics for functional connectivity matrices.
 
    - Edge-wise ICC  (wraps arfcexp.icc.batch_icc)
    - Gradient canonical correlation  (wraps arfcexp.graph_metrics.calc_gradient_similarity)
 
Both metrics require *multiple runs per subject* and measure consistency
across repetitions, which is why they live separately from info_density.py.
"""
  
from itertools import combinations
 
import numpy as np
import pandas as pd
 
from arfcexp.graph_metrics import calc_gradient_similarity, fc_to_affinity, fit_gradients
from arfcexp.icc import batch_icc, ICCResult
from arfcexp.matrices import apply_sparsity, import_matrix


def build_icc_input(
    run_df: pd.DataFrame,
    *,
    mat_col: str = "mat",
) -> np.ndarray:
    """Stack per-run sparsified matrices into shape (n_edges, n_subs, n_runs).
 
    Subjects with fewer than the maximum number of runs are padded with NaN
    so that ``batch_icc`` handles them via listwise deletion.
 
    Args:
        run_df: DataFrame with columns [sub, ses, run, <mat_col>].
                Each entry in mat_col must be a flat sparsified array of
                identical length.
        mat_col: column holding the flat matrix arrays.
 
    Returns:
        Array of shape (n_edges, n_subs, n_runs).
    """
    subs = sorted(run_df["sub"].unique())
    n_edges = len(run_df[mat_col].iloc[0])
    n_runs = int(run_df.groupby("sub").size().max())
 
    icc_input = np.full((n_edges, len(subs), n_runs), np.nan)
    for i, sub in enumerate(subs):
        sub_rows = (
            run_df[run_df["sub"] == sub]
            .sort_values(["ses", "run"])
            .reset_index(drop=True)
        )
        for j, (_, row) in enumerate(sub_rows.iterrows()):
            if j >= n_runs:
                break
            icc_input[:, i, j] = row[mat_col]
 
    return icc_input
 
 
def compute_icc(
    run_df: pd.DataFrame,
    *,
    is_symmetric: bool = False,
    sparsity: float = 0.8,
    mat_col: str = "mat",
) -> ICCResult:
    """Compute edge-wise ICC across all runs.
 
    Applies sparsity thresholding to each raw run matrix before stacking,
    then calls ``batch_icc``.
 
    Args:
        run_df: DataFrame with columns [sub, ses, run, <mat_col>].
        is_symmetric: whether matrices are symmetric (affects sparsity thresholding).
        sparsity: sparsity level passed to ``apply_sparsity``.
        mat_col: column holding the raw flat matrix arrays.
 
    Returns:
        ``ICCResult`` named tuple (icc1 … icc3k), each an array over edges.
    """
    df = run_df.copy()
    df[mat_col] = df[mat_col].apply(
        lambda m: apply_sparsity(np.asarray(m), is_symmetric=is_symmetric, sparsity=sparsity)
    )
    icc_input = build_icc_input(df, mat_col=mat_col)
    return batch_icc(icc_input)


def _gradient_sim_one_sub(
    sub: str,
    mats: list[np.ndarray],
    *,
    affinity_threshold: float,
    n_components: int,
) -> dict:
    """Compute mean pairwise gradient similarity for a single subject."""
    if len(mats) < 2:
        return {"sub": sub, "gradient_similarity": float("nan")}

    gradients = []
    for mat in mats:
        try:
            A = import_matrix(mat)
            aff = fc_to_affinity(A, threshold=affinity_threshold)
            grad = fit_gradients(aff, n_components=n_components)
            gradients.append(grad)
        except Exception:
            continue  # skip this run if eigenmaps fails

    if len(gradients) < 2:
        return {"sub": sub, "gradient_similarity": float("nan")}

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
    rows = []
    for sub, grp in run_df.groupby("sub"):
        mats = grp[mat_col].tolist()
        rows.append(
            _gradient_sim_one_sub(
                sub, mats,
                affinity_threshold=affinity_threshold,
                n_components=n_components,
            )
        )
    return pd.DataFrame(rows)

def compute_identifiability(
    run_df: pd.DataFrame,
    *,
    is_symmetric: bool = False,
    sparsity: float = 0.8,
    mat_col: str = "mat",
) -> dict:
    """
    Differential identifiability (Amico & Goni 2018 / Tian & Zalesky 2021).

    Steps:
    1. Apply sparsity to each run matrix
    2. Average within each session (test = ses 1, retest = ses 2)
    3. Build N x N identifiability matrix A of Pearson correlations
    4. I_self  = mean(diagonal)
    5. I_others = mean(off-diagonal)
    6. I_diff  = (I_self - I_others) * 100
    7. Success rate = fraction correctly identified (diagonal is max in column)
    """
    df = run_df.copy()
    df[mat_col] = df[mat_col].apply(
        lambda m: apply_sparsity(np.asarray(m), is_symmetric=is_symmetric, sparsity=sparsity)
    )

    ses_avg = (
        df.groupby(["sub", "ses"])[mat_col]
        .apply(lambda x: np.mean(np.stack(x.values), axis=0))
        .reset_index()
    )

    subs = [s for s, g in ses_avg.groupby("sub") if len(g) >= 2]
    if len(subs) < 2:
        return {"I_self": np.nan, "I_others": np.nan, "I_diff": np.nan,
                "identifiability_matrix": None, "success_rate": np.nan}

    test = np.stack([
        ses_avg[ses_avg["sub"] == s].sort_values("ses").iloc[0][mat_col]
        for s in subs
    ])
    retest = np.stack([
        ses_avg[ses_avg["sub"] == s].sort_values("ses").iloc[1][mat_col]
        for s in subs
    ])

    def _zscore(X):
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        return np.where(sd > 0, (X - mu) / sd, 0.0)

    A = _zscore(test) @ _zscore(retest).T / test.shape[1]
    A = np.clip(A, -1, 1)

    I_self   = float(np.mean(np.diag(A)))
    mask_off = ~np.eye(len(subs), dtype=bool)
    I_others = float(np.mean(A[mask_off]))
    I_diff   = (I_self - I_others) * 100

    success_rate = float(np.mean(np.argmax(A, axis=0) == np.arange(len(subs))))

    return {
        "I_self": I_self,
        "I_others": I_others,
        "I_diff": I_diff,
        "identifiability_matrix": A,
        "success_rate": success_rate,
    }

def compute_discriminability(
    run_df: pd.DataFrame,
    *,
    is_symmetric: bool = False,
    sparsity: float = 0.8,
    mat_col: str = "mat",
    chunk_size: int = 10,
) -> float:
    """
    Discriminability (Bridgeford et al., PLoS Computational Biology 2021).

    Steps:
    1. Apply sparsity to each run matrix
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
    df[mat_col] = df[mat_col].apply(
        lambda m: apply_sparsity(np.asarray(m), is_symmetric=is_symmetric, sparsity=sparsity)
    )

    # average within each session per subject
    ses_avg = (
        df.groupby(["sub", "ses"])[mat_col]
        .apply(lambda x: np.mean(np.stack(x.values), axis=0))
        .reset_index()
    )

    # keep only subjects with at least 2 sessions
    subs = [s for s, g in ses_avg.groupby("sub") if len(g) >= 2]
    if len(subs) < 2:
        return float("nan")

    # build test (ses 1) and retest (ses 2) arrays — shape (n_subs, n_edges)
    test = np.stack([
        ses_avg[ses_avg["sub"] == s].sort_values("ses").iloc[0][mat_col]
        for s in subs
    ])
    retest = np.stack([
        ses_avg[ses_avg["sub"] == s].sort_values("ses").iloc[1][mat_col]
        for s in subs
    ])

    n = len(subs)

    # within-subject distances — shape (n_subs,)
    d_within = np.linalg.norm(test - retest, axis=1)

    correct = 0.0
    total = 0

    # chunk over rows
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        # D_chunk[i, j] = ||test[start:end][i] - retest[j]||
        diff = test[start:end, np.newaxis, :] - retest[np.newaxis, :, :]
        D_chunk = np.linalg.norm(diff, axis=2) 

        for local_i, global_i in enumerate(range(start, end)):
            for j in range(n):
                if global_i == j:
                    continue
                if d_within[global_i] < D_chunk[local_i, j]:
                    correct += 1.0
                elif d_within[global_i] == D_chunk[local_i, j]:
                    correct += 0.5
                total += 1

    return float(correct / total) if total > 0 else float("nan")