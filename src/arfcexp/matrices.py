from pathlib import Path

import numpy as np
import pandas as pd
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
