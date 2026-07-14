"""Compute information-density intrinsic benchmark metrics for ensemble combos.

Mirrors notebooks/compute_infodensity_benchmarks.ipynb's per-subject metric
computation (arfcexp.info_density.compute_all_intrinsic) but scoped to the
ensemble parquet, and additionally aggregates (mean across subjects) into a
summary CSV with the same column schema as output/info_density_summary.csv,
since that notebook only produces raw per-subject output (info_density.parquet)
and the aggregation step that produces the tracked summary CSV was not found
in this repository (likely run out-of-repo). Writes to
output/info_density_summary_ensemble.csv (separate file, does not touch the
existing output/info_density_summary.csv).

Usage:
    uv run python scripts/compute_ensemble_info_density.py
"""

import logging
import os
from pathlib import Path

import pandas as pd
import typer
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from arfcexp.info_density import compute_all_intrinsic
from arfcexp.matrices import EfficientMatrixReader, load_avg_mats_from_parquet

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

DEFAULT_DATA_PATH = "data/ensemble_krakencoder/hcp_1200_rfmri_schaefer_ensemble.parquet"
DEFAULT_METHOD_LIST = "resources/ensemble_method_func_list.txt"
DEFAULT_OUT_DIR = "output"
TSP_NODES = list(range(20))
SMALL_WORLD_KWARGS = dict(seed=70, nrand=1, nswap=2000)
RICH_CLUB_KWARGS = {"normalized": False}


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

    reader = EfficientMatrixReader(data_path_p)
    sub_list = sorted(
        reader.query(columns=[], method="ensemble", func=combos[0][1], success=True)["sub"]
        .unique()
        .to_list()
    )
    logging.info(f"Using {len(sub_list)} subjects (from first combo) for all ensemble combos.")

    def _process(method: str, func: str) -> pd.DataFrame:
        with threadpool_limits(limits=1):
            local_reader = EfficientMatrixReader(data_path_p)
            avg_df = load_avg_mats_from_parquet(
                data_path_p, method=method, func=func, sub_list=sub_list, lag=0, reader=local_reader
            )
            valid = avg_df[avg_df["Matrix"].notna() & (avg_df["Count"] > 0)]
            if valid.empty:
                logging.warning(f"{method}/{func}: no valid subjects, skipping.")
                return pd.DataFrame()

            logging.info(f"{method}/{func}: computing intrinsic metrics for {len(valid)} subjects.")
            rows = []
            for sub, row_data in valid.iterrows():
                try:
                    metrics = compute_all_intrinsic(
                        row_data["Matrix"],
                        is_symmetric=True,  # all ensemble matrices are symmetric (tri2square construction)
                        tsp_nodes=TSP_NODES,
                        small_world_kwargs=SMALL_WORLD_KWARGS,
                        rich_club_kwargs=RICH_CLUB_KWARGS,
                    )
                except Exception as exc:
                    logging.warning(f"{method}/{func} sub={sub}: {exc}")
                    metrics = {}
                rows.append({"sub": sub, "method": method, "func": func, "lag": 0, **metrics})
            return pd.DataFrame(rows)

    logging.info(f"Computing info density for {len(combos)} ensemble combos ({n_jobs} parallel workers)...")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_process)(method, func) for method, func in combos
    )
    all_dfs = [df for df in results if df is not None and not df.empty]
    if not all_dfs:
        raise RuntimeError("No info density results computed for any ensemble combo.")

    raw_df = pd.concat(all_dfs, ignore_index=True)
    raw_out_path = out_dir_p / "info_density_ensemble.parquet"
    raw_df.to_parquet(raw_out_path, index=False)
    logging.info(f"Wrote raw per-subject metrics ({len(raw_df)} rows) to {raw_out_path}")

    metric_cols = [c for c in raw_df.columns if c not in ("sub", "method", "func", "lag")]
    summary_df = raw_df.groupby(["method", "func", "lag"], as_index=False)[metric_cols].mean()
    summary_df["combo_key"] = summary_df.apply(lambda r: f"{r['method']}__{r['func']}", axis=1)
    summary_df["is_directed"] = False

    summary_out_path = out_dir_p / "info_density_summary_ensemble.csv"
    summary_df.to_csv(summary_out_path, index=False)
    logging.info(f"Wrote summary ({len(summary_df)} combos) to {summary_out_path}")


if __name__ == "__main__":
    typer.run(main)
