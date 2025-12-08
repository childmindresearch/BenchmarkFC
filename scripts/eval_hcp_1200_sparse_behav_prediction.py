"""HCP behavioral prediction with sparsity-thresholded connectivity matrices.

Follows the methods from Yeo Lab papers, e.g. Kong et al., NeuroImage, 2023.
Applies configurable sparsity threshold to connectivity matrices before prediction.

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
import typer
import yaml
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import _safe_split

import arfcexp.hcp
import arfcexp.prediction
import arfcexp.transforms
from arfcexp.matrices import (
    compute_pearson_kernel,
    load_avg_mats_and_impose_sparsity,
    load_symmetry_lookup,
)

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

# Outer loop test size.
TEST_SIZE = 0.2
# Number of inner loop CV splits.
NUM_INNER_SPLITS = 5

# Default regularization strengths from Yeo lab papers + modified high values.
ALPHAS = [
    0.00001, 0.0001, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1,
    0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15, 20, 50, 100
]  # fmt: skip

# Minimum fraction of subjects with at least one successfully computed run.
MIN_VALID_SUB_FRACTION = 0.9


def load_method_func_id_map(project_root: Path) -> dict:
    """Load method and func_id mapping from CSV file."""
    csv_path = project_root / "output/mean_sparsity_by_function_all_thresholds.csv"
    df = pd.read_csv(csv_path)
    # Create mapping: (method, func) -> func_id
    id_map = {}
    for _, row in df.iterrows():
        key = (row["method"], row["func"])
        id_map[key] = int(row["func_id"])
    return id_map


def main(
    method: str = "pyspi",
    func: str = "cov_EmpiricalCovariance",
    data_path: str = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet",
    target: str = "Cognition",
    parc_size: int = 200,
    pool: int = 3,
    sparsity: float = 0.8,
    n_splits: int = 20,
    perm_test: bool = False,
    n_subjects: int | None = None,
    lag: int = 0,
    seed: int = 2142,
    out_dir: str | None = None,
):
    project_root = Path(os.environ["PROJECT_ROOT"])
    random_state = check_random_state(seed)

    # Look up method/func ID
    id_map = load_method_func_id_map(project_root)
    method_id = id_map.get((method, func))
    if method_id is None:
        logging.error(f"Method/func combination not found: {method}/{func}")
        return 1

    params = {
        "method": method,
        "func": func,
        "method_id": method_id,
        "target": target,
        "parc_size": parc_size,
        "pool": pool,
        "sparsity": sparsity,
        "n_splits": n_splits,
        "perm_test": perm_test,
        "n_subjects": n_subjects,
        "lag": lag,
        "seed": seed,
        "data_path": data_path,
    }

    logging.info(
        "Testing HCP sparse behavioral prediction:\n\t"
        + "\n\t".join(f"{k}={v}" for k, v in params.items())
    )

    if out_dir is None:
        out_dir = project_root / "results/hcp_1200_sparse_behav_prediction"
    else:
        out_dir = Path(out_dir)
    out_dir: Path

    # Add lag suffix to directory name for skarf methods only
    if method == "skarf":
        method_func_dir = f"{method_id:03d}__{method}__{func}__lag-{lag}"
    else:
        method_func_dir = f"{method_id:03d}__{method}__{func}"
    
    out_dir = (
        out_dir
        / f"sparsity-{sparsity:.2f}__parc-{parc_size}__pool-{pool}__n-{n_splits}__perm-{int(perm_test)}__seed-{seed}"
        / method_func_dir
        / f"target-{target}"
    )

    if out_dir.exists():
        logging.info("Output already exists; exiting.")
        return

    out_dir.mkdir(exist_ok=True, parents=True)

    # Save experiment params.
    with (out_dir / "params.yaml").open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    logging.info(f"Loading matrices for {method=} {func=} from parquet.")
    data_path = Path(data_path)
    if not data_path.exists():
        logging.warning(f"Data path {data_path} does not exist; exiting.")
        return 1

    sub_list = arfcexp.hcp.load_hcp_subject_list()
    
    # Optionally subset subjects for testing
    if n_subjects is not None:
        sub_list = sub_list[:n_subjects]
        logging.info(f"Using subset of {n_subjects} subjects for testing.")
    
    # Load symmetry lookup for efficient symmetric matrix handling
    logging.info("Loading symmetry lookup...")
    symmetry_lookup = load_symmetry_lookup(project_root)

    avg_mats_df = load_avg_mats_and_impose_sparsity(
        data_path, method, func, sub_list, 
        sparsity=sparsity, symmetry_lookup=symmetry_lookup, lag=lag
    )

    # Matrices are averaged over session/run.
    # Count number of runs per subject.
    run_counts = avg_mats_df["Count"].value_counts().to_dict()
    valid_sub_fraction = (avg_mats_df["Count"] > 0).mean()
    logging.info(f"Run counts for {method=} {func=}:\n{run_counts}")

    if valid_sub_fraction < MIN_VALID_SUB_FRACTION:
        logging.warning(
            f"Not enough subjects with data ({valid_sub_fraction:.2%} < {MIN_VALID_SUB_FRACTION:.0%}); exiting."
        )
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

        # Initialize model. Note we do this inside the loop rather than just once in order
        # to sample a new CV seed each time. No other reason really. Technically we
        # could probably just update the CV seed in place..
        cv_seed = random_state.randint(1000, 10000)
        model = GridSearchCV(
            arfcexp.prediction.TargetTransformEstimator(
                KernelRidge(kernel="precomputed"),
                arfcexp.transforms.HCPBehavTargetTransform(target_name=target),
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

        r2_train = r2_score(targets_train, preds_train)
        r2_test = r2_score(targets_test, preds_test)
        r2_val = model.best_score_

        mse_train = mean_squared_error(targets_train, preds_train)
        mse_test = mean_squared_error(targets_test, preds_test)

        corr_train = arfcexp.prediction.corr_score(targets_train, preds_train)
        corr_test = arfcexp.prediction.corr_score(targets_test, preds_test)

        result = {
            **params,
            "split": split,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "alpha": alpha,
            "r2_train": r2_train,
            "r2_val": r2_val,
            "r2_test": r2_test,
            "mse_train": mse_train,
            "mse_test": mse_test,
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


if __name__ == "__main__":
    typer.run(main)
