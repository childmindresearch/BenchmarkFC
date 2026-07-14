"""Ensemble FC - behavioral prediction (Cognition), the "phenotypic_non_sparse"
equivalent for method="ensemble" combos.

The existing eval_hcp_1200_{pyspi,skarf}_behav_prediction_factor_full.py scripts
read raw per-subject arrow directories keyed by a hardcoded spi_id/func_id list
and cannot be pointed at any parquet file. This script mirrors their exact
prediction algorithm (Pearson-kernel KRR, GroupShuffleSplit outer loop,
GroupKFold inner loop, same ALPHAS grid) but sources matrices from the small
ensemble parquet built by scripts/build_ensemble_matrices.py via
EfficientMatrixReader / load_avg_mats_from_parquet, keyed by method="ensemble".

Output directory naming follows the same "{id:03d}__{method}__{func}" unified
convention used elsewhere, so notebooks/analyze_hcp_1200_benchmark_scores_combined.ipynb's
parse_result_path() can extract method/func/lag once extended to recognize
"ensemble" (see Phase 5 changes to that notebook).

Usage:
    uv run python scripts/eval_hcp_1200_ensemble_behav_prediction.py --func top5_simple_average
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
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import _safe_split

import arfcexp.hcp
import arfcexp.prediction
import arfcexp.transforms
from arfcexp.matrices import EfficientMatrixReader, compute_pearson_kernel, load_avg_mats_from_parquet

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

# Outer loop test size.
TEST_SIZE = 0.2
# Number of inner loop CV splits.
NUM_INNER_SPLITS = 5

# Same modified-Yeo regularization grid as eval_hcp_1200_pyspi_behav_prediction.py.
ALPHAS = [
    0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5,
    4, 5, 10, 15, 20, 50, 100,
]  # fmt: skip

# Minimum fraction of subjects with at least one successfully computed run.
MIN_VALID_SUB_FRACTION = 0.9

DEFAULT_DATA_PATH = "data/ensemble_krakencoder/hcp_1200_rfmri_schaefer_ensemble.parquet"
DEFAULT_METHOD_LIST = "resources/ensemble_method_func_list.txt"


def main(
    func: str = "top5_simple_average",
    target: str = "Cognition",
    data_path: str | None = None,
    n_splits: int = 20,
    perm_test: bool = False,
    seed: int = 2142,
    out_dir: str | None = None,
):
    method = "ensemble"
    params = {
        "method": method,
        "func": func,
        "target": target,
        "n_splits": n_splits,
        "perm_test": perm_test,
        "seed": seed,
    }

    logging.info(
        "Testing HCP ensemble behavioral prediction:\n\t"
        + "\n\t".join(f"{k}={v}" for k, v in params.items())
    )

    project_root = Path(os.environ["PROJECT_ROOT"])
    random_state = check_random_state(seed)

    method_list_path = project_root / DEFAULT_METHOD_LIST
    method_func_list = method_list_path.read_text().strip().splitlines()
    func_list = [line.split("\t")[1] for line in method_func_list if line.strip()]
    if func not in func_list:
        raise ValueError(f"Unknown ensemble func {func!r}; expected one of {func_list}")
    func_id = func_list.index(func)

    data_path_p = Path(data_path) if data_path is not None else project_root / DEFAULT_DATA_PATH
    if not data_path_p.exists():
        raise FileNotFoundError(f"Ensemble parquet not found: {data_path_p}")

    if out_dir is None:
        out_dir_p = project_root / "results/hcp_1200_ensemble_behav_prediction"
    else:
        out_dir_p = Path(out_dir)

    out_dir_p = (
        out_dir_p
        / f"n-{n_splits}__perm-{int(perm_test)}__seed-{seed}"
        / f"{func_id:03d}__{method}__{func}"
        / f"target-{target}"
    )

    if out_dir_p.exists():
        logging.info("Output already exists; exiting.")
        return

    out_dir_p.mkdir(exist_ok=True, parents=True)

    with (out_dir_p / "params.yaml").open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    reader = EfficientMatrixReader(data_path_p)
    available_subs = sorted(
        reader.query(columns=[], method=method, func=func, success=True)["sub"].unique().to_list()
    )
    if not available_subs:
        logging.warning(f"No successful rows found for method={method} func={func}; exiting.")
        return 1
    logging.info(f"{len(available_subs)} subjects available for {method}/{func}.")

    avg_mats_df = load_avg_mats_from_parquet(
        data_path_p, method=method, func=func, sub_list=available_subs, lag=0, reader=reader
    )
    valid_sub_fraction = (avg_mats_df["Count"] > 0).mean()
    if valid_sub_fraction < MIN_VALID_SUB_FRACTION:
        logging.warning("Not enough subjects with data; exiting.")
        return 1

    # Behavioral targets, nuisance covariates, and family groups.
    behav = arfcexp.hcp.load_hcp_behav().loc[available_subs]
    covariates = arfcexp.hcp.load_hcp_covariates().loc[available_subs]
    groups = arfcexp.hcp.load_hcp_family_groups().loc[available_subs].values

    X = np.stack(avg_mats_df["Matrix"].values)
    y = pd.concat([covariates, behav], axis=1)

    # Pre-compute NxN kernel matrix.
    X = compute_pearson_kernel(X)

    split_seed = random_state.randint(1000, 10000)
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=TEST_SIZE, random_state=split_seed)
    logging.info("Train/test split seed: %d", split_seed)

    if perm_test:
        perm_seed = random_state.randint(1000, 10000)
        perm_random_state = check_random_state(perm_seed)

    output_summary_path = out_dir_p / "results.json"

    for split, (train_ind, test_ind) in enumerate(splitter.split(X, groups=groups)):
        state_path = out_dir_p / f"split-{split}__state.pkl"

        cv_seed = random_state.randint(1000, 10000)
        model = GridSearchCV(
            arfcexp.prediction.TargetTransformEstimator(
                KernelRidge(kernel="precomputed"),
                arfcexp.transforms.HCPBehavTargetTransform(target_name=target),
            ),
            param_grid={"estimator__alpha": ALPHAS},
            cv=GroupKFold(n_splits=NUM_INNER_SPLITS, shuffle=True, random_state=cv_seed),
            verbose=0,
        )

        if perm_test:
            perm_indices = perm_random_state.permutation(len(y))
            y_split = y.iloc[perm_indices]
        else:
            y_split = y

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
