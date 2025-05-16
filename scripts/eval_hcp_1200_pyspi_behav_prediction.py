"""HCP behavioral prediction.

Follows the methods from Yeo Lab papers, e.g. Kong et al., NeuroImage, 2023.

References:
    https://github.com/ThomasYeoLab/CBIG/blob/v0.29.2-Kong2022_update/utilities/matlab/predictive_models/KernelRidgeRegression/
"""

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import typer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import check_random_state

import arfcexp.hcp
import arfcexp.prediction

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

# Number of outer train test/splits.
NUM_SPLITS = 20
NUM_INNER_SPLITS = 10

# Debug regularization strength
# ALPHAS = [0.01, 0.1, 1.0]

# Default regularization strengths From Yeo lab papers.
# ALPHAS = [
#     0.00001, 0.0001, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07,
#     0.1, 0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15, 20
# ]
# Truncated Yeo regularization strengths.
ALPHAS = [0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10]

# Minimum fraction of subjects with at least one successfully computed run.
MIN_VALID_SUB_FRACTION = 0.9


def main(
    spi: str = "cov_EmpiricalCovariance",
    parc_size: int = 200,
    pool: int = 3,
    target: str = "Cognition",
    seed: int = 2142,
    out_dir: str | None = None,
):
    params = {
        "spi": spi,
        "parc_size": parc_size,
        "pool": pool,
        "target": target,
        "seed": seed,
    }

    logging.info(
        "Testing HCP pyspi behavioral prediction:\n\t"
        + "\n\t".join(f"{k}={v}" for k, v in params.items())
    )

    project_root = Path(os.environ["PROJECT_ROOT"])
    random_state = check_random_state(seed)

    # Look up SPI and subject index IDs.
    spi_list_dir = project_root / "resources/spi_lists"
    spi_list = (spi_list_dir / "spi_list_all_284.txt").read_text().strip().split()
    spi_id_map = {spi: ii for ii, spi in enumerate(spi_list)}
    spi_id = spi_id_map[spi]

    if out_dir is None:
        out_dir = project_root / "results/hcp_1200_pyspi_behav_prediction"
    else:
        out_dir = Path(out_dir)
    out_dir: Path

    out_dir = (
        out_dir
        / f"parc-{parc_size}__pool-{pool}"
        / f"seed-{seed}"
        / f"{spi_id:03d}__spi-{spi}"
        / f"target-{target}"
    )

    output_summary_path = out_dir / "results.json"
    if output_summary_path.exists():
        logging.info("Output already exists; exiting.")
        return

    out_dir.mkdir(exist_ok=True, parents=True)

    logging.info(f"Loading matrices for {spi=}.")
    mats_dir = (
        project_root
        / "data/hcp_1200_rfmri_schaefer_pyspi"
        / f"parc-{parc_size}__pool-{pool}"
        / f"{spi_id:03d}__spi-{spi}"
    )
    if not mats_dir.exists():
        logging.warning(f"SPI matrices dir {mats_dir} does not exist; exiting.")
        return 1

    sub_list = arfcexp.hcp.load_hcp_subject_list()
    avg_mats_df = load_avg_mats(mats_dir, sub_list)

    # Matrices are averaged over session/run.
    # Count number of runs per subject.
    run_counts = avg_mats_df["Count"].value_counts().to_dict()
    valid_sub_fraction = (avg_mats_df["Count"] > 0).mean()
    logging.info(f"Run counts for {spi=}:\n{run_counts}")

    if valid_sub_fraction < MIN_VALID_SUB_FRACTION:
        logging.warning("Not enough subjects with data; exiting.")
        return 1

    # Behavioral targets, nuisance covariates, and family groups.
    behav = arfcexp.hcp.load_hcp_behav().loc[sub_list]
    covariates = arfcexp.hcp.load_hcp_covariates().loc[sub_list]
    groups = arfcexp.hcp.load_hcp_family_groups().loc[sub_list].values

    X = pd.concat([covariates, avg_mats_df], axis=1)
    y = behav

    # Group k-fold outer loop cross-validation.
    # Group splitting used to ensure no related individuals split across train and test.
    split_seed = random_state.randint(1000, 10000)
    splitter = GroupKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=split_seed)
    logging.info("Train/test split seed: %d", split_seed)

    for split, (train_ind, test_ind) in enumerate(splitter.split(X, groups=groups)):
        X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
        y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]
        groups_train = groups[train_ind]

        model = fit(
            X=X_train,
            y=y_train,
            groups=groups_train,
            target_name=target,
            n_splits=NUM_INNER_SPLITS,
            random_state=random_state,
        )

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        val_score = model.best_score_
        alpha = model.best_params_["regressor__alpha"]

        if alpha in {ALPHAS[0], ALPHAS[-1]}:
            logging.warning(f"Optimal alpha {alpha} on the grid boundary.")

        result = {
            **params,
            "split": split,
            "alpha": alpha,
            "train_score": train_score,
            "val_score": val_score,
            "test_score": test_score,
        }
        logging.info(f"Result (split={split}):\n{json.dumps(result)}")
        with output_summary_path.open("a") as f:
            print(json.dumps(result), file=f)

        state_path = out_dir / f"split-{split}__model.pkl"
        with state_path.open("wb") as f:
            pickle.dump(model, file=f)


def fit(
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
    target_name: str,
    n_splits: int = NUM_INNER_SPLITS,
    random_state: np.random.RandomState | None = None,
):
    random_state = check_random_state(random_state)

    model = arfcexp.prediction.HCPBehavRegressor(
        KernelRidge(kernel="cosine"),
        feature_name="Matrix",
        target_name=target_name,
    )

    # Fixed CV seed across all CV iterations, for reproducible splitting.
    cv_seed = random_state.randint(1000, 10000)

    model = GridSearchCV(
        model,
        param_grid={"regressor__alpha": ALPHAS},
        cv=GroupKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed),
        verbose=3,
    )

    model.fit(X, y, groups=groups)
    return model


def load_avg_mats(mats_dir: Path, sub_list: list[str]) -> pd.DataFrame:
    """Load average FC matrices from an FC matrix dataset for a list of subjects.

    Return array of average matrices and the run counts. Subjects with missing data are
    given all zero matrices.
    """
    mats_ds = pads.dataset(mats_dir, format="arrow")
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


if __name__ == "__main__":
    typer.run(main)
