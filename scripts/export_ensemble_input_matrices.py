"""Export per-subject averaged FC matrices for the top-N ensemble methods.

Reads resources/ensemble_method_lists/top{N}_methods.csv (produced by
scripts/select_ensemble_methods.py), pulls each method's per-subject averaged
matrix from the aggregated parquet via ``EfficientMatrixReader`` /
``load_avg_mats_from_parquet``, and writes stacked ``.npy`` arrays (one per
combo) plus a ``subjects.txt`` ordering file, so that later krakencoder
bridging/training scripts have simple numpy inputs to work with.

Output layout:
    data/ensemble_krakencoder/inputs/top{N}/subjects.txt
    data/ensemble_krakencoder/inputs/top{N}/{combo_key}.npy   # (n_subjects, parc, parc)

Usage:
    uv run python scripts/export_ensemble_input_matrices.py
    uv run python scripts/export_ensemble_input_matrices.py --top-n 5 --top-n 10
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from arfcexp.matrices import EfficientMatrixReader, import_matrix, load_avg_mats_from_parquet

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

DEFAULT_DATA_PATH = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet"
DEFAULT_METHOD_LIST_DIR = "resources/ensemble_method_lists"
DEFAULT_SUBJECT_LIST = "resources/subject_lists/hcp_complete_data_867_subject_list.txt"
DEFAULT_OUT_DIR = "data/ensemble_krakencoder/inputs"
DEFAULT_TOP_N = [5, 10, 15]


def main(
    data_path: str = DEFAULT_DATA_PATH,
    method_list_dir: str | None = None,
    subject_list: str | None = None,
    out_dir: str | None = None,
    top_n: list[int] = DEFAULT_TOP_N,
):
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))

    method_list_dir_path = (
        Path(method_list_dir) if method_list_dir is not None else project_root / DEFAULT_METHOD_LIST_DIR
    )
    subject_list_path = (
        Path(subject_list) if subject_list is not None else project_root / DEFAULT_SUBJECT_LIST
    )
    out_dir_path = Path(out_dir) if out_dir is not None else project_root / DEFAULT_OUT_DIR
    out_dir_path.mkdir(parents=True, exist_ok=True)

    data_path_p = Path(data_path)
    if not data_path_p.exists():
        raise FileNotFoundError(f"Aggregated parquet not found: {data_path_p}")

    with open(subject_list_path) as f:
        sub_list = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(sub_list)} subjects from {subject_list_path}")

    reader = EfficientMatrixReader(data_path_p)

    # Collect the union of (method, func, lag) combos needed across all requested N,
    # so each combo's matrices are only loaded once even if it appears in multiple lists.
    combo_cache: dict[tuple[str, str, int], np.ndarray] = {}

    for n in sorted(top_n):
        list_path = method_list_dir_path / f"top{n}_methods.csv"
        if not list_path.exists():
            raise FileNotFoundError(
                f"Method list not found: {list_path}. "
                "Run scripts/select_ensemble_methods.py first."
            )
        methods_df = pd.read_csv(list_path)

        n_out_dir = out_dir_path / f"top{n}"
        n_out_dir.mkdir(parents=True, exist_ok=True)
        with open(n_out_dir / "subjects.txt", "w") as f:
            f.write("\n".join(sub_list) + "\n")

        for row in methods_df.itertuples(index=False):
            method = row.method
            func = row.func
            lag = int(row.lag) if not pd.isna(row.lag) else 0
            combo_key = row.combo_key

            cache_key = (method, func, lag)
            if cache_key not in combo_cache:
                logging.info(f"Loading matrices for {combo_key} ...")
                avg_df = load_avg_mats_from_parquet(
                    data_path_p,
                    method=method,
                    func=func,
                    sub_list=sub_list,
                    lag=lag,
                    reader=reader,
                )
                n_valid = int((avg_df["Count"] > 0).sum())
                if n_valid < len(sub_list):
                    logging.warning(
                        f"{combo_key}: only {n_valid}/{len(sub_list)} subjects have data; "
                        "missing subjects filled with zero matrices."
                    )
                stacked = np.stack(
                    [import_matrix(mat) for mat in avg_df["Matrix"].to_numpy()]
                )
                combo_cache[cache_key] = stacked

            stacked = combo_cache[cache_key]
            out_path = n_out_dir / f"{combo_key}.npy"
            np.save(out_path, stacked)
            logging.info(f"Wrote {out_path} (shape={stacked.shape})")

    logging.info(f"Done. Exported matrices for {len(combo_cache)} unique method/func/lag combos.")


if __name__ == "__main__":
    typer.run(main)
