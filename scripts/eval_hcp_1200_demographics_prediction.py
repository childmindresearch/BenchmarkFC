"""HCP demographics (gender, age) prediction from functional connectivity.

Predicts gender (binary, via KRR + threshold) and age (continuous, KRR) from
FC matrices, following He et al. 2020 (doi.org/10.1016/j.neuroimage.2019.116276)
methodology.

No confound regression is applied for demographics prediction, per He2020.

References:
    He et al. (2020). Deep Neural Networks and Kernel Regression Achieve
    Comparable Accuracies for Functional Connectivity Prediction of Behavior
    and Demographics. NeuroImage, 206, 116276.

    https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/predict_phenotypes/He2019_KRDNN
"""

import gc
import json
import logging
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import _safe_split
from threadpoolctl import threadpool_limits

import arfcexp.hcp
import arfcexp.prediction
from arfcexp.matrices import (
    EfficientMatrixReader,
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

# Number of inner loop CV splits (matching He2019).
NUM_INNER_SPLITS = 20

# Full Yeo lab regularization grid from He2019 (38 values).
# In sklearn KRR, alpha = lambda from He2019.
# Use small epsilon instead of 0 to avoid singular matrix.
ALPHAS = [
    1e-10, 0.00001, 0.0001, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1,
    0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15, 20, 30, 40, 50,
    60, 70, 80, 100, 150, 200, 300, 500, 700, 1000, 10000, 100000, 1000000,
]  # fmt: skip

# Minimum fraction of subjects with at least one successfully computed run.
MIN_VALID_SUB_FRACTION = 0.9

# Gender threshold for classification from continuous KRR output.
GENDER_THRESHOLD = 0.5


def load_method_func_id_map(project_root: Path) -> dict[tuple[str, str], int]:
    """Load method and func_id mapping from CSV file."""
    csv_path = project_root / "output/mean_sparsity_by_function_all_thresholds.csv"
    df = pd.read_csv(csv_path)
    id_map = {}
    for _, row in df.iterrows():
        key = (row["method"], row["func"])
        id_map[key] = int(row["func_id"])
    return id_map


def compute_gender_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = GENDER_THRESHOLD) -> float:
    """Compute classification accuracy from continuous predictions via thresholding."""
    y_class = (y_pred >= threshold).astype(int)
    return float(np.mean(y_class == y_true))


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Find the threshold that maximizes accuracy on training data."""
    thresholds = np.linspace(y_pred.min() - 0.01, y_pred.max() + 0.01, 200)
    best_acc = 0.0
    best_thresh = GENDER_THRESHOLD
    for threshold in thresholds:
        acc = float(np.mean(((y_pred >= threshold).astype(int)) == y_true))
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(threshold)
    return best_thresh, best_acc


def get_requested_tasks(task: str) -> tuple[str, ...]:
    """Return the task names to execute."""
    if task == "both":
        return ("gender", "age")
    if task in ("gender", "age"):
        return (task,)
    raise ValueError(f"Unknown task: {task}. Must be 'gender', 'age', or 'both'.")


def build_method_func_dir(method: str, func: str, method_id: int, lag: int) -> str:
    if method == "skarf":
        return f"{method_id:03d}__{method}__{func}__lag-{lag}"
    return f"{method_id:03d}__{method}__{func}"


def get_task_out_dir(task: str, params: dict, out_dir: Path) -> Path:
    return (
        out_dir
        / (
            f"task-{task}__sparsity-{params['sparsity']:.2f}"
            f"__parc-{params['parc_size']}__pool-{params['pool']}"
            f"__n-{params['n_splits']}__perm-{int(params['perm_test'])}"
            f"__seed-{params['seed']}"
        )
        / build_method_func_dir(
            params["method"],
            params["func"],
            params["method_id"],
            params["lag"],
        )
    )


def is_task_complete(task_out_dir: Path, n_splits: int) -> bool:
    if not task_out_dir.exists():
        return False
    return all(
        (task_out_dir / f"split-{split}__state.pkl").exists()
        for split in range(n_splits)
    )


def run_task_prediction(
    task: str,
    K: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict,
    out_dir: Path,
) -> dict:
    """Run nested CV for one task (gender or age) on a precomputed kernel."""
    seed = params["seed"]
    n_splits = params["n_splits"]
    n_inner_splits = params["n_inner_splits"]

    task_params = {**params, "task": task}
    task_out_dir = get_task_out_dir(task, params, out_dir)

    if is_task_complete(task_out_dir, n_splits):
        return {
            "task": task,
            "method": params["method"],
            "func": params["func"],
            "lag": params["lag"],
            "status": "skipped",
        }

    task_out_dir.mkdir(exist_ok=True, parents=True)

    with (task_out_dir / "params.yaml").open("w") as f:
        yaml.safe_dump(task_params, f, sort_keys=False)

    random_state = check_random_state(seed)
    split_seed = random_state.randint(1000, 10000)
    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=TEST_SIZE,
        random_state=split_seed,
    )

    perm_random_state = None
    if params["perm_test"]:
        perm_seed = random_state.randint(1000, 10000)
        perm_random_state = check_random_state(perm_seed)

    output_summary_path = task_out_dir / "results.json"
    did_run_split = False

    for split, (train_ind, test_ind) in enumerate(splitter.split(K, groups=groups)):
        state_path = task_out_dir / f"split-{split}__state.pkl"
        if state_path.exists():
            continue

        did_run_split = True
        cv_seed = random_state.randint(1000, 10000)
        model = GridSearchCV(
            KernelRidge(kernel="precomputed"),
            param_grid={"alpha": ALPHAS},
            cv=GroupKFold(
                n_splits=n_inner_splits,
                shuffle=True,
                random_state=cv_seed,
            ),
            scoring="neg_mean_squared_error",
            verbose=0,
        )

        if params["perm_test"]:
            assert perm_random_state is not None
            perm_indices = perm_random_state.permutation(len(y))
            y_split = y[perm_indices]
        else:
            y_split = y

        K_train, y_train = _safe_split(model, K, y_split, train_ind)
        K_test, y_test = _safe_split(model, K, y_split, test_ind, train_ind)
        groups_train = groups[train_ind]

        model.fit(K_train, y_train, groups=groups_train)
        alpha = model.best_params_["alpha"]

        preds_train = model.predict(K_train)
        preds_test = model.predict(K_test)

        result = {
            **task_params,
            "split": split,
            "n_train": len(K_train),
            "n_test": len(K_test),
            "alpha": alpha,
            "r2_train": r2_score(y_train, preds_train),
            "neg_mse_val": model.best_score_,
            "r2_test": r2_score(y_test, preds_test),
            "mse_train": mean_squared_error(y_train, preds_train),
            "mse_test": mean_squared_error(y_test, preds_test),
            "mae_train": mean_absolute_error(y_train, preds_train),
            "mae_test": mean_absolute_error(y_test, preds_test),
            "corr_train": float(arfcexp.prediction.corr_score(y_train, preds_train)),
            "corr_test": float(arfcexp.prediction.corr_score(y_test, preds_test)),
        }

        if task == "gender":
            acc_train = compute_gender_accuracy(y_train, preds_train)
            acc_test = compute_gender_accuracy(y_test, preds_test)
            opt_thresh, opt_acc_train = find_optimal_threshold(y_train, preds_train)
            opt_acc_test = compute_gender_accuracy(y_test, preds_test, threshold=opt_thresh)
            result.update({
                "acc_train": acc_train,
                "acc_test": acc_test,
                "opt_threshold": opt_thresh,
                "opt_acc_train": opt_acc_train,
                "opt_acc_test": opt_acc_test,
            })

        logging.info(f"Result (task={task}, split={split}):\n{json.dumps(result)}")
        with output_summary_path.open("a") as f:
            print(json.dumps(result), file=f)

        state = {
            **result,
            "cv_seed": cv_seed,
            "targets_train": y_train,
            "targets_test": y_test,
            "preds_train": preds_train,
            "preds_test": preds_test,
            "cv_results": model.cv_results_,
        }
        with state_path.open("wb") as f:
            pickle.dump(state, file=f)

    return {
        "task": task,
        "method": params["method"],
        "func": params["func"],
        "lag": params["lag"],
        "status": "success" if did_run_split else "skipped",
    }


def run_combination_prediction(
    task: str = "gender",
    method: str = "pyspi",
    func: str = "cov_EmpiricalCovariance",
    data_path: str = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet",
    parc_size: int = 200,
    pool: int = 3,
    sparsity: float = 0.0,
    n_splits: int = 20,
    n_inner_splits: int = NUM_INNER_SPLITS,
    perm_test: bool = False,
    n_subjects: int | None = None,
    lag: int = 0,
    seed: int = 2142,
    threads_per_worker: int = 1,
    out_dir: str | Path | None = None,
) -> tuple[list[dict], int]:
    """Run one method/func(/lag) combination end-to-end.

    If ``task='both'``, gender and age are evaluated in one process so the matrix
    load and kernel build happen only once.
    """
    project_root = Path(os.environ["PROJECT_ROOT"])
    requested_tasks = get_requested_tasks(task)

    if threads_per_worker < 1:
        raise ValueError(f"Expected threads_per_worker >= 1, got {threads_per_worker}.")

    data_path_path = Path(data_path)
    if not data_path_path.exists():
        logging.error("Data path %s does not exist.", data_path_path)
        return [{"method": method, "func": func, "lag": lag, "status": "missing_data_path"}], 1

    id_map = load_method_func_id_map(project_root)
    method_id = id_map.get((method, func))
    if method_id is None:
        logging.error("Method/func combination not found: %s/%s", method, func)
        return [{"method": method, "func": func, "lag": lag, "status": "unknown"}], 1

    if out_dir is None:
        out_dir_path = project_root / "results/hcp_1200_demographics_prediction"
    else:
        out_dir_path = Path(out_dir)
    out_dir_path.mkdir(exist_ok=True, parents=True)

    params = {
        "method": method,
        "func": func,
        "method_id": method_id,
        "parc_size": parc_size,
        "pool": pool,
        "sparsity": sparsity,
        "n_splits": n_splits,
        "n_inner_splits": n_inner_splits,
        "perm_test": perm_test,
        "n_subjects": n_subjects,
        "lag": lag,
        "seed": seed,
        "data_path": str(data_path_path),
        "threads_per_worker": threads_per_worker,
    }

    combo_label = f"{method} {func}" + (f" lag={lag}" if method == "skarf" else "")
    if all(
        is_task_complete(get_task_out_dir(task_name, params, out_dir_path), n_splits)
        for task_name in requested_tasks
    ):
        logging.info("[%s] Requested tasks already complete; skipping.", combo_label)
        return [
            {
                "task": task_name,
                "method": method,
                "func": func,
                "lag": lag,
                "status": "skipped",
            }
            for task_name in requested_tasks
        ], 0

    sub_list = arfcexp.hcp.load_hcp_subject_list()
    if n_subjects is not None:
        sub_list = sub_list[:n_subjects]
        logging.info("[%s] Using subset of %d subjects.", combo_label, n_subjects)

    task_targets = {}
    if "gender" in requested_tasks:
        task_targets["gender"] = arfcexp.hcp.load_hcp_gender().loc[sub_list].values.astype(float)
    if "age" in requested_tasks:
        task_targets["age"] = arfcexp.hcp.load_hcp_age().loc[sub_list].values.astype(float)
    groups = arfcexp.hcp.load_hcp_family_groups().loc[sub_list].values

    logging.info(
        "Testing HCP demographics prediction:\n\t"
        + "\n\t".join(f"{k}={v}" for k, v in {**params, "task": task}.items())
    )

    try:
        with threadpool_limits(limits=threads_per_worker):
            symmetry_lookup = load_symmetry_lookup(project_root)
            reader = EfficientMatrixReader(data_path_path)

            logging.info("[%s] Loading matrices from parquet.", combo_label)
            avg_mats_df = load_avg_mats_and_impose_sparsity(
                data_path_path,
                method,
                func,
                sub_list,
                sparsity=sparsity,
                symmetry_lookup=symmetry_lookup,
                lag=lag,
                reader=reader,
            )

            run_counts = avg_mats_df["Count"].value_counts().to_dict()
            valid_sub_fraction = (avg_mats_df["Count"] > 0).mean()
            logging.info("[%s] Run counts: %s", combo_label, run_counts)

            if valid_sub_fraction < MIN_VALID_SUB_FRACTION:
                logging.warning(
                    "[%s] Not enough subjects with data (%.2f%% < %.0f%%).",
                    combo_label,
                    valid_sub_fraction * 100,
                    MIN_VALID_SUB_FRACTION * 100,
                )
                del avg_mats_df
                gc.collect()
                return [{
                    "method": method,
                    "func": func,
                    "lag": lag,
                    "status": "insufficient_data",
                }], 1

            X = np.stack(avg_mats_df["Matrix"].values)
            K = compute_pearson_kernel(X)
            del avg_mats_df, X
            gc.collect()

            results = []
            exit_code = 0
            for task_name, y in task_targets.items():
                try:
                    result = run_task_prediction(task_name, K, y, groups, params, out_dir_path)
                    results.append(result)
                    logging.info("[%s] [%s] %s", combo_label, task_name, result["status"])
                except Exception:
                    logging.error(
                        "[%s] [%s] FAILED:\n%s",
                        combo_label,
                        task_name,
                        traceback.format_exc(),
                    )
                    results.append({
                        "task": task_name,
                        "method": method,
                        "func": func,
                        "lag": lag,
                        "status": "error",
                    })
                    exit_code = 1

            del K
            gc.collect()
            return results, exit_code
    except Exception:
        logging.error("[%s] Combination FAILED:\n%s", combo_label, traceback.format_exc())
        gc.collect()
        return [{"method": method, "func": func, "lag": lag, "status": "error"}], 1


def main(
    task: str = "gender",
    method: str = "pyspi",
    func: str = "cov_EmpiricalCovariance",
    data_path: str = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet",
    parc_size: int = 200,
    pool: int = 3,
    sparsity: float = 0.0,
    n_splits: int = 20,
    n_inner_splits: int = NUM_INNER_SPLITS,
    perm_test: bool = False,
    n_subjects: int | None = None,
    lag: int = 0,
    seed: int = 2142,
    threads_per_worker: int = 1,
    out_dir: str | None = None,
):
    results, exit_code = run_combination_prediction(
        task=task,
        method=method,
        func=func,
        data_path=data_path,
        parc_size=parc_size,
        pool=pool,
        sparsity=sparsity,
        n_splits=n_splits,
        n_inner_splits=n_inner_splits,
        perm_test=perm_test,
        n_subjects=n_subjects,
        lag=lag,
        seed=seed,
        threads_per_worker=threads_per_worker,
        out_dir=out_dir,
    )

    statuses = ", ".join(
        f"{result.get('task', 'load')}={result['status']}" for result in results
    )
    logging.info("Combination summary: %s", statuses)

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    typer.run(main)
