from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as pads
from sklearn.metrics.pairwise import cosine_similarity


def compute_pearson_kernel(X: np.ndarray) -> np.ndarray:
    # Center each sample
    X = X - np.nanmean(X, axis=1, keepdims=True)
    # Fill NaN.
    X = np.where(np.isnan(X), 0.0, X)
    # Cosine kernel, i.e. Pearson correlation since the samples are centered.
    K = cosine_similarity(X)
    return K


def load_avg_mats(mats_dir: Path, sub_list: list[str]) -> pd.DataFrame:
    """Load average FC matrices from an FC matrix dataset for a list of subjects.

    Return array of average matrices and the run counts. Subjects with missing data are
    given all zero matrices.
    """
    mats_ds = pads.dataset(sorted(mats_dir.rglob("*.arrow")), format="arrow")
    mats_df = mats_ds.to_table().to_pandas()

    # Average across sessions/runs
    avg_mats_df = mats_df.groupby(["sub"]).agg(
        {"success": "sum", "mat": average_matrices}
    )

    mat_shape, mat_dtype = next(
        (mat.shape, mat.dtype) for mat in avg_mats_df["mat"] if mat is not None
    )

    avg_mats = []
    counts = []
    for sub in sub_list:
        if sub in avg_mats_df.index:
            mat = avg_mats_df.loc[sub, "mat"]
            if mat is None:
                mat = np.zeros(mat_shape, dtype=mat_dtype)
            count = avg_mats_df.loc[sub, "success"]
        else:
            mat = np.zeros(mat_shape, dtype=mat_dtype)
            count = 0
        avg_mats.append(mat)
        counts.append(count)

    avg_mats_df = pd.DataFrame({"Count": counts, "Matrix": avg_mats}, index=sub_list)
    return avg_mats_df


def average_matrices(mats: list[np.ndarray]) -> np.ndarray:
    mats = [mat for mat in mats if mat is not None]
    if len(mats) == 0:
        return None
    return np.nanmean(np.stack(mats), axis=0)


def load_avg_mats_from_parquet(
    parquet_path: Path,
    method: str,
    func: str,
    sub_list: list[str],
    sparsity: float = 0.8,
) -> pd.DataFrame:
    """Load average FC matrices from parquet file with sparsity thresholding.

    Args:
        parquet_path: Path to the aggregated parquet file.
        method: Method name ("pyspi" or "skarf").
        func: Function name (e.g., "cov_EmpiricalCovariance", "linear_ridge").
        sub_list: List of subject IDs to load.
        sparsity: Sparsity level to impose (default 0.8 = keep top 20%).

    Returns:
        DataFrame with columns "Count" (number of runs) and "Matrix" (sparsity-thresholded
        averaged matrices), indexed by subject ID.
    """
    # Load and filter data using polars for memory efficiency
    df_pl = (
        pl.scan_parquet(parquet_path)
        .filter(
            pl.col("success")
            & (pl.col("method") == method)
            & (pl.col("func") == func)
        )
        .select(["sub", "ses", "run", "mat"])
        .collect()
    )

    # Convert to pandas for groupby operations
    df_pd = df_pl.to_pandas()

    if len(df_pd) == 0:
        # No data found for this method/func combination
        # Return empty DataFrame with correct structure
        avg_mats_df = pd.DataFrame(
            {"Count": [0] * len(sub_list), "Matrix": [None] * len(sub_list)},
            index=sub_list,
        )
        return avg_mats_df

    # Group by subject and average matrices across sessions/runs
    avg_mats_df = df_pd.groupby("sub").agg({"mat": ["count", average_matrices]})
    avg_mats_df.columns = ["Count", "Matrix"]

    # Apply sparsity threshold to each matrix
    def apply_sparsity(mat):
        if mat is None:
            return None
        mat = np.array(mat)
        # Threshold at the (sparsity * 100)th percentile on absolute values
        # Use nanpercentile to handle NaN values (e.g., diagonal in covariance matrices)
        threshold = np.nanpercentile(np.abs(mat), sparsity * 100)
        mat_sparse = np.where(np.abs(mat) >= threshold, mat, 0.0)
        return mat_sparse

    avg_mats_df["Matrix"] = avg_mats_df["Matrix"].apply(apply_sparsity)

    # Determine matrix shape and dtype from first valid matrix
    mat_shape = None
    mat_dtype = None
    for mat in avg_mats_df["Matrix"]:
        if mat is not None:
            mat_shape = mat.shape
            mat_dtype = mat.dtype
            break

    # Fill in missing subjects with zero matrices
    avg_mats = []
    counts = []
    for sub in sub_list:
        if sub in avg_mats_df.index:
            mat = avg_mats_df.loc[sub, "Matrix"]
            if mat is None and mat_shape is not None:
                mat = np.zeros(mat_shape, dtype=mat_dtype)
            count = avg_mats_df.loc[sub, "Count"]
        else:
            if mat_shape is not None:
                mat = np.zeros(mat_shape, dtype=mat_dtype)
            else:
                mat = None
            count = 0
        avg_mats.append(mat)
        counts.append(count)

    result_df = pd.DataFrame({"Count": counts, "Matrix": avg_mats}, index=sub_list)
    return result_df
