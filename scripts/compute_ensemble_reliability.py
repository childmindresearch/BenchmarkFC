"""Compute reliability intrinsic benchmark metrics for ensemble combos.

Mirrors notebooks/compute_reliability_benchmarks.ipynb's per-combo reliability
computation (ICC, gradient similarity, identifiability, discriminability) but
scoped to the ensemble parquet, and aggregates into a summary CSV with the
SAME column schema as output/reliability_summary.csv, so
notebooks/analyze_hcp_1200_benchmark_scores_combined.ipynb can concatenate
core + ensemble reliability blocks identically to how it already does for
info_density.

This only works now that scripts/build_ensemble_matrices.py fuses PER RUN
(preserving [sub, ses, run] structure) instead of on a single subject-averaged
matrix - see that script's docstring. Writes to
output/reliability_summary_ensemble.csv (separate file, does not touch the
existing output/reliability_summary.csv).

Usage:
    uv run python scripts/compute_ensemble_reliability.py
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from arfcexp.matrices import EfficientMatrixReader
from arfcexp.reliability import (
    compute_discriminability,
    compute_gradient_reliability,
    compute_icc,
    compute_identifiability,
)

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

DEFAULT_DATA_PATH = "data/ensemble_krakencoder/hcp_1200_rfmri_schaefer_ensemble.parquet"
DEFAULT_METHOD_LIST = "resources/ensemble_method_func_list.txt"
DEFAULT_OUT_DIR = "output"


def _empty_result(method: str, func: str) -> dict:
    return {
        "method": method,
        "func": func,
        "lag": 0,
        "combo_key": f"{method}__{func}",
        "is_directed": False,
        "mean_icc2": np.nan,
        "mean_icc3": np.nan,
        "gradient_similarity": np.nan,
        "I_diff": np.nan,
        "success_rate": np.nan,
        "discriminability": np.nan,
    }


def main(
    data_path: str | None = None,
    method_func_list: str | None = None,
    out_dir: str | None = None,
    n_jobs: int = 4,
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

    def _process(method: str, func: str) -> dict:
        with threadpool_limits(limits=1):
            reader = EfficientMatrixReader(data_path_p)
            run_df = reader.query(
                columns=["sub", "ses", "run", "mat"], method=method, func=func, success=True
            ).to_pandas()

            if run_df.empty:
                logging.warning(f"{method}/{func}: no rows, skipping.")
                return _empty_result(method, func)

            n_subs = run_df["sub"].nunique()
            logging.info(
                f"{method}/{func}: computing reliability metrics for "
                f"{n_subs} subjects, {len(run_df)} runs."
            )

            result = _empty_result(method, func)

            try:
                icc_result = compute_icc(run_df, is_symmetric=True)
                result["mean_icc2"] = float(np.nanmean(icc_result.icc2))
                result["mean_icc3"] = float(np.nanmean(icc_result.icc3))
            except Exception as exc:
                logging.warning(f"{method}/{func}: ICC failed: {exc}")

            try:
                grad_df = compute_gradient_reliability(run_df)
                result["gradient_similarity"] = float(np.nanmean(grad_df["gradient_similarity"]))
            except Exception as exc:
                logging.warning(f"{method}/{func}: gradient reliability failed: {exc}")

            try:
                ident = compute_identifiability(run_df, is_symmetric=True)
                result["I_diff"] = float(ident["I_diff"])
                result["success_rate"] = float(ident["success_rate"])
            except Exception as exc:
                logging.warning(f"{method}/{func}: identifiability failed: {exc}")

            try:
                result["discriminability"] = float(
                    compute_discriminability(run_df, is_symmetric=True)
                )
            except Exception as exc:
                logging.warning(f"{method}/{func}: discriminability failed: {exc}")

            return result

    logging.info(f"Computing reliability for {len(combos)} ensemble combos ({n_jobs} parallel workers)...")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_process)(method, func) for method, func in combos
    )

    summary_df = pd.DataFrame(results)
    out_path = out_dir_p / "reliability_summary_ensemble.csv"
    summary_df.to_csv(out_path, index=False)
    logging.info(f"Wrote summary ({len(summary_df)} combos) to {out_path}")


if __name__ == "__main__":
    typer.run(main)
