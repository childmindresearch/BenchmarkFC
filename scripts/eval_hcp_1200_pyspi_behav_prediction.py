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
import yaml
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import _safe_split

import arfcexp.hcp
import arfcexp.prediction
import arfcexp.transforms

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

# Outer loop test size.
TEST_SIZE = 0.2
# Number of inner loop CV splits.
NUM_INNER_SPLITS = 5

# Debug regularization strength
# ALPHAS = [0.01, 0.1, 1.0]

# Default regularization strengths From Yeo lab papers.
# ALPHAS = [
#     0.00001, 0.0001, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07,
#     0.1, 0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15, 20
# ]

# Modified Yeo regularization strengths.
# Less low values, More high values.
ALPHAS = [
    0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5,
    4, 5, 10, 15, 20, 50, 100,
]  # fmt: skip

# Minimum fraction of subjects with at least one successfully computed run.
MIN_VALID_SUB_FRACTION = 0.9


def main(
    spi: str = "cov_EmpiricalCovariance",
    parc_size: int = 200,
    pool: int = 3,
    target: str = "Cognition",
    n_splits: int = 20,
    perm_test: bool = False,
    seed: int = 2142,
    out_dir: str | None = None,
):
    params = {
        "spi": spi,
        "parc_size": parc_size,
        "pool": pool,
        "target": target,
        "n_splits": n_splits,
        "perm_test": perm_test,
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
        / f"parc-{parc_size}__pool-{pool}__perm-{int(perm_test)}__seed-{seed}"
        / f"{spi_id:03d}__spi-{spi}"
        / f"target-{target}"
    )

    out_dir.mkdir(exist_ok=True, parents=True)

    # Save experiment params.
    with (out_dir / "params.yaml").open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

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

    X = np.stack(avg_mats_df["Matrix"].values)
    y = pd.concat([covariates, behav], axis=1)

    # Pre-compute NxN kernel matrix.
    X = compute_pearson_kernel(X)

    # Group k-fold outer loop cross-validation.
    # Group splitting used to ensure no related individuals split across train and test.
    split_seed = random_state.randint(1000, 10000)
    splitter = GroupShuffleSplit(
        n_splits=n_splits, test_size=TEST_SIZE, random_state=split_seed
    )
    logging.info("Train/test split seed: %d", split_seed)

    if perm_test:
        perm_seed = random_state.randint(1000, 10000)
        perm_random_state = check_random_state(perm_seed)

    output_summary_path = out_dir / "results.json"

    for split, (train_ind, test_ind) in enumerate(splitter.split(X, groups=groups)):
        state_path = out_dir / f"split-{split}__state.pkl"
        if state_path.exists():
            logging.info(f"State already exists for split={split}; skipping.")
            continue

        # Initialize model. Note we do this inside the loop rather than just once in order
        # to sample a new CV seed each time. No other reason really. Technically we
        # could probably just update the CV seed in place..
        cv_seed = random_state.randint(1000, 10000)
        model = GridSearchCV(
            arfcexp.prediction.TargetTransformEstimator(
                KernelRidge(kernel="precomputed"),
                arfcexp.transforms.HCPBehavTargetTransform(target_name=target),
                scoring="neg_mean_squared_error",
            ),
            param_grid={"estimator__alpha": ALPHAS},
            cv=GroupKFold(
                n_splits=NUM_INNER_SPLITS,
                shuffle=True,
                random_state=cv_seed,
            ),
            verbose=0,
        )

        # Permute targets for permutation test.
        if perm_test:
            perm_indices = perm_random_state.permutation(len(y))
            y_split = y.iloc[perm_indices]
        else:
            perm_indices = None
            y_split = y

        # Split data safely, respecting the precomputed kernel.
        X_train, y_train = _safe_split(model, X, y_split, train_ind)
        X_test, y_test = _safe_split(model, X, y_split, test_ind, train_ind)
        groups_train = groups[train_ind]

        model.fit(X_train, y_train, groups=groups_train)
        alpha = model.best_params_["estimator__alpha"]

        best_model = model.best_estimator_
        targets_train = best_model.target_transform_.transform(y_train)
        targets_test = best_model.target_transform_.transform(y_test)

        preds_train = best_model.predict(X_train)
        preds_test = best_model.predict(X_test)

        mse_train = mean_squared_error(targets_train, preds_train)
        mse_test = mean_squared_error(targets_test, preds_test)
        mse_val = -model.best_score_

        r2_train = r2_score(targets_train, preds_train)
        r2_test = r2_score(targets_test, preds_test)

        corr_train = arfcexp.prediction.corr_score(targets_train, preds_train)
        corr_test = arfcexp.prediction.corr_score(targets_test, preds_test)

        if alpha in {ALPHAS[0], ALPHAS[-1]}:
            logging.warning(f"Optimal alpha {alpha} on the grid boundary.")

        result = {
            **params,
            "split": split,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "alpha": alpha,
            "mse_train": mse_train,
            "mse_val": mse_val,
            "mse_test": mse_test,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "corr_train": corr_train,
            "corr_test": corr_test,
        }
        logging.info(f"Result (split={split}):\n{json.dumps(result)}")
        with output_summary_path.open("a") as f:
            print(json.dumps(result), file=f)

        state = {
            **result,
            "cv_seed": cv_seed,
            "targets_train": targets_train,
            "targets_test": targets_test,
            "preds_train": preds_train,
            "preds_test": preds_test,
            "cv_results": model.cv_results_,
        }
        with state_path.open("wb") as f:
            pickle.dump(state, file=f)


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
