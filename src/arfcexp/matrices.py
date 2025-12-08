import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as pads
from sklearn.metrics.pairwise import cosine_similarity


def load_symmetry_lookup(project_root: Path | None = None) -> dict:
    """Load symmetry lookup dictionary from JSON file.
    
    Returns:
        Dictionary mapping "method__func" keys to boolean symmetry values.
    """
    if project_root is None:
        project_root = Path(os.environ["PROJECT_ROOT"])
    
    lookup_path = project_root / "resources/matrix_symmetry_lookup.json"
    
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"Symmetry lookup file not found: {lookup_path}\n"
            "Please run notebooks/detect_matrix_symmetry.ipynb to generate it."
        )
    
    with open(lookup_path, 'r') as f:
        symmetry_lookup = json.load(f)
    
    return symmetry_lookup


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


def apply_sparsity(mat: np.ndarray, is_symmetric: bool, sparsity: float = 0.8) -> np.ndarray:
    """Apply sparsity threshold to matrix.
    
    Args:
        mat: Flattened connectivity matrix.
        is_symmetric: Whether the matrix is symmetric.
        sparsity: Sparsity level to impose (default 0.8 = keep top 20%).
    
    Returns:
        Flattened sparsity-thresholded matrix.
    """
    # Reshape flattened array to 2D matrix
    n_elements = len(mat)
    n = int(np.sqrt(n_elements))
    if n * n != n_elements:
        raise ValueError(f"Cannot reshape array of length {n_elements} to square matrix")
    
    mat = mat.reshape(n, n)
    
    if is_symmetric:
        # For symmetric matrices, threshold on upper triangle only
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        upper_vals = mat[triu_indices]
        
        # Apply threshold to upper triangle
        threshold = np.nanpercentile(np.abs(upper_vals), sparsity * 100)
        upper_sparse = np.where(np.abs(upper_vals) >= threshold, upper_vals, 0.0)
        
        # Reconstruct symmetric matrix
        mat_sparse = np.zeros_like(mat)
        mat_sparse[triu_indices] = upper_sparse
        mat_sparse = mat_sparse + mat_sparse.T  # Mirror to lower triangle
        np.fill_diagonal(mat_sparse, mat.diagonal())  # Preserve diagonal
    else:
        # For non-symmetric matrices, threshold entire matrix
        threshold = np.nanpercentile(np.abs(mat), sparsity * 100)
        mat_sparse = np.where(np.abs(mat) >= threshold, mat, 0.0)
    
    # Flatten back to 1D for storage
    return mat_sparse.flatten()


def load_avg_mats_and_impose_sparsity(
    parquet_path: Path,
    method: str,
    func: str,
    sub_list: list[str],
    sparsity: float = 0.8,
    symmetry_lookup: dict | None = None,
    lag: int = 0,
) -> pd.DataFrame:
    """Load average FC matrices from parquet file with sparsity thresholding.

    Args:
        parquet_path: Path to the aggregated parquet file.
        method: Method name ("pyspi" or "skarf").
        func: Function name (e.g., "cov_EmpiricalCovariance", "linear_ridge").
        sub_list: List of subject IDs to load.
        sparsity: Sparsity level to impose (default 0.8 = keep top 20%).
        symmetry_lookup: Dictionary mapping "method__func" to boolean symmetry.
            If None, will not use symmetric matrix optimization.
        lag: Lag value for skarf methods (default 0). Only used for filtering data.

    Returns:
        DataFrame with columns "Count" (number of runs) and "Matrix" (sparsity-thresholded
        averaged matrices), indexed by subject ID.
    """
    # Load and filter data using polars for memory efficiency
    # For skarf methods, also filter by lag
    filter_conditions = (
        pl.col("success")
        & (pl.col("method") == method)
        & (pl.col("func") == func)
    )
    
    if method == "skarf":
        filter_conditions = filter_conditions & (pl.col("lag") == lag)
    
    df_pl = (
        pl.scan_parquet(parquet_path)
        .filter(filter_conditions)
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

    # Check if this method/func produces symmetric matrices
    is_symmetric = False
    if symmetry_lookup is not None:
        lookup_key = f"{method}__{func}"
        is_symmetric = symmetry_lookup.get(lookup_key, False)
    
    # Apply sparsity threshold to each matrix
    avg_mats_df["Matrix"] = avg_mats_df["Matrix"].apply(
        lambda mat: apply_sparsity(mat, is_symmetric, sparsity)
    )

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
