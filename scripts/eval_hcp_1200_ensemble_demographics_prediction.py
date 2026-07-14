"""Ensemble FC - demographics (gender + age) prediction.

Reuses eval_hcp_1200_demographics_prediction.run_combination_prediction()
directly (in-process, not via subprocess) for each of the 9 ensemble
method/func combos in resources/ensemble_method_func_list.txt, pointed at the
ensemble parquet built by scripts/build_ensemble_matrices.py.

The upstream script's method_id lookup (load_method_func_id_map) is sourced
from output/mean_sparsity_by_function_all_thresholds.csv, which has no
"ensemble" entries. Rather than mutate that shared file, this driver
monkey-patches the lookup with IDs derived from
resources/ensemble_method_func_list.txt (0-indexed row order) for the
duration of the run.

Usage:
    uv run python scripts/eval_hcp_1200_ensemble_demographics_prediction.py
"""

import importlib
import logging
import os
from pathlib import Path

import typer

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

DEFAULT_DATA_PATH = "data/ensemble_krakencoder/hcp_1200_rfmri_schaefer_ensemble.parquet"
DEFAULT_METHOD_LIST = "resources/ensemble_method_func_list.txt"
DEFAULT_OUT_DIR = "results/hcp_1200_demographics_prediction_ensemble"


def main(
    data_path: str | None = None,
    method_func_list: str | None = None,
    out_dir: str | None = None,
    n_splits: int = 20,
    n_inner_splits: int = 20,
    seed: int = 2142,
    threads_per_worker: int = 4,
):
    project_root = Path(os.environ["PROJECT_ROOT"])

    data_path_p = Path(data_path) if data_path is not None else project_root / DEFAULT_DATA_PATH
    method_list_path = (
        Path(method_func_list) if method_func_list is not None else project_root / DEFAULT_METHOD_LIST
    )
    out_dir_p = Path(out_dir) if out_dir is not None else project_root / DEFAULT_OUT_DIR
    out_dir_p.mkdir(parents=True, exist_ok=True)

    combos = []
    with method_list_path.open() as f:
        for line in f:
            method, func = line.strip().split("\t")
            combos.append((method, func))

    ensemble_id_map = {(method, func): idx for idx, (method, func) in enumerate(combos)}

    demo_module = importlib.import_module("eval_hcp_1200_demographics_prediction")
    original_load_map = demo_module.load_method_func_id_map

    def patched_load_method_func_id_map(project_root_arg):
        id_map = original_load_map(project_root_arg)
        id_map.update(ensemble_id_map)
        return id_map

    demo_module.load_method_func_id_map = patched_load_method_func_id_map

    all_results = []
    for method, func in combos:
        logging.info(f"Running demographics prediction for {method}/{func} ...")
        results, exit_code = demo_module.run_combination_prediction(
            task="both",
            method=method,
            func=func,
            data_path=str(data_path_p),
            n_splits=n_splits,
            n_inner_splits=n_inner_splits,
            lag=0,
            seed=seed,
            threads_per_worker=threads_per_worker,
            out_dir=str(out_dir_p),
        )
        for result in results:
            logging.info(f"[{method}/{func}] {result}")
        all_results.extend(results)
        if exit_code != 0:
            logging.error(f"Combination {method}/{func} exited with code {exit_code}")

    demo_module.load_method_func_id_map = original_load_map

    n_success = sum(1 for r in all_results if r.get("status") in ("success", "skipped"))
    logging.info(f"Done. {n_success}/{len(all_results)} task results succeeded/skipped.")


if __name__ == "__main__":
    typer.run(main)
