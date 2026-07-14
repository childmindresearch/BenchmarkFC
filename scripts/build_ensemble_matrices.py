"""Build ensemble FC matrices via krakencoder, PER RUN, and write them to a small parquet.

For each top-N method list and each of the 3 reconstruction rules, loads each
constituent method's RAW per-run matrices (not subject-averaged) directly from
the main aggregated parquet, applies the already-trained per-flavor bridged
krakencoder encoder/decoder (scripts/train_ensemble_krakencoder_bridge.py) to
every (sub, ses, run) where ALL constituent methods succeeded, and writes one
row per (sub, ses, run) to a new small parquet file using the SAME column
schema as the main aggregated parquet (method, func, func_id, sub, sub_id,
ses, run, success, run_time, mat, lag), so it can be read unmodified via
EfficientMatrixReader / load_avg_mats_from_parquet.

Fusing per-run (instead of on a single subject-averaged matrix) preserves the
session/run structure needed to compute reliability metrics (ICC, gradient
similarity, discriminability) for ensemble combos - see
scripts/compute_ensemble_reliability.py. No retraining of the bridge models is
needed: encode_matrix/decode_latent/apply_ensemble_rule already operate on one
matrix at a time, so they can be called once per (sub, ses, run) instead of
once per subject.

method="ensemble", func=f"top{N}_{rule}", lag=0.

Usage:
    uv run python scripts/build_ensemble_matrices.py
    uv run python scripts/build_ensemble_matrices.py --top-n 5 --rule simple_average
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from scipy.io import loadmat

from arfcexp.ensemble_utils import (
    AVAILABLE_ENSEMBLE_RULES,
    apply_ensemble_rule,
    load_ensemble_manifest,
    load_flavor_models,
)
from arfcexp.matrices import EfficientMatrixReader, import_matrix

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

DEFAULT_DATA_PATH = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet"
DEFAULT_METHOD_LIST_DIR = "resources/ensemble_method_lists"
DEFAULT_BRIDGE_DIR = "data/ensemble_krakencoder/bridge"
DEFAULT_MANIFEST = "data/ensemble_krakencoder/checkpoints/manifest.json"
DEFAULT_OUT_PATH = "data/ensemble_krakencoder/hcp_1200_rfmri_schaefer_ensemble.parquet"
DEFAULT_TOP_N = [5, 10, 15]
PARC_SIZE = 200


def _load_combo_run_df(
    raw_df: pl.DataFrame,
    overlap_subjects: set[str],
) -> pl.DataFrame:
    """Restrict a raw per-run [sub, ses, run, mat] fetch to the bridge overlap cohort.

    ``EfficientMatrixReader.batch_query`` already compacts the ``mat`` column
    to float32 numpy arrays during the row-group read (compact_mat=True), so
    no further conversion is needed here.
    """
    return raw_df.filter(pl.col("sub").is_in(overlap_subjects)).select(["sub", "ses", "run", "mat"])


def main(
    data_path: str | None = None,
    method_list_dir: str | None = None,
    bridge_dir: str | None = None,
    manifest_path: str | None = None,
    out_path: str | None = None,
    top_n: list[int] = DEFAULT_TOP_N,
    rule: list[str] = AVAILABLE_ENSEMBLE_RULES,
):
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))

    data_path_p = Path(data_path) if data_path is not None else Path(DEFAULT_DATA_PATH)
    method_list_dir_path = (
        Path(method_list_dir) if method_list_dir is not None
        else project_root / DEFAULT_METHOD_LIST_DIR
    )
    bridge_dir_path = Path(bridge_dir) if bridge_dir is not None else project_root / DEFAULT_BRIDGE_DIR
    manifest_full_path = (
        Path(manifest_path) if manifest_path is not None else project_root / DEFAULT_MANIFEST
    )
    out_path_full = Path(out_path) if out_path is not None else project_root / DEFAULT_OUT_PATH
    out_path_full.parent.mkdir(parents=True, exist_ok=True)

    if not data_path_p.exists():
        raise FileNotFoundError(f"Aggregated parquet not found: {data_path_p}")

    manifest = load_ensemble_manifest(manifest_full_path)

    unknown_rules = set(rule) - set(AVAILABLE_ENSEMBLE_RULES)
    if unknown_rules:
        raise ValueError(f"Unknown rules: {unknown_rules}. Available: {AVAILABLE_ENSEMBLE_RULES}")

    reader = EfficientMatrixReader(data_path_p)

    # --- Pass 1: gather per-group metadata + the union of constituent combos ---
    # (methods_df, combo_keys, weights, reference_key, overlap_subjects, sub_to_idx)
    group_info: dict[int, dict] = {}
    combo_filters: dict[str, dict] = {}  # combo_key -> (method, func, lag)
    for n in sorted(top_n):
        list_path = method_list_dir_path / f"top{n}_methods.csv"
        if not list_path.exists():
            raise FileNotFoundError(f"{list_path} not found. Run scripts/select_ensemble_methods.py first.")
        methods_df = pd.read_csv(list_path)
        combo_keys = methods_df["combo_key"].tolist()
        weights = dict(zip(methods_df["combo_key"], methods_df["maxnorm_rank_sum"]))
        reference_key = methods_df.loc[methods_df["rank"] == 1, "combo_key"].iloc[0]

        n_bridge_dir = bridge_dir_path / f"top{n}"
        split_mat = loadmat(n_bridge_dir / "bridge_subject_split.mat", simplify_cells=True)
        overlap_subjects = [str(s).strip() for s in split_mat["subjects"]]

        group_info[n] = {
            "combo_keys": combo_keys,
            "weights": weights,
            "reference_key": reference_key,
            "overlap_subject_set": set(overlap_subjects),
            "sub_to_idx": {sub: idx for idx, sub in enumerate(overlap_subjects)},
        }

        for row in methods_df.itertuples(index=False):
            if row.combo_key not in combo_filters:
                lag_val = 0 if pd.isna(row.lag) else int(row.lag)
                combo_filters[row.combo_key] = {"method": row.method, "func": row.func, "lag": lag_val}

    # --- Pass 2: fetch ALL constituent combos' per-run rows in ONE parquet pass ---
    # (batch_query reads each row group at most once across all combos; calling
    # EfficientMatrixReader.query() once per combo instead would re-read most
    # row groups from scratch for every combo since rows aren't contiguous by
    # method/func in the parquet - this was the main source of the earlier slowdown.)
    combo_keys_ordered = list(combo_filters.keys())
    queries = []
    for key in combo_keys_ordered:
        spec = combo_filters[key]
        query_filters = {"method": spec["method"], "func": spec["func"], "success": True}
        if spec["method"] == "skarf":
            query_filters["lag"] = spec["lag"]
        queries.append(query_filters)

    logging.info(f"Fetching per-run matrices for {len(queries)} unique constituent combos in one pass...")
    raw_dfs = reader.batch_query(queries, columns=["sub", "ses", "run", "mat"])
    combo_raw_cache = dict(zip(combo_keys_ordered, raw_dfs))

    # --- Pass 3: per top-N group, load models, restrict to overlap cohort, ---
    # --- inner-join constituents on (sub, ses, run), apply ensemble rules   ---
    # Rows are written incrementally (one rule's worth at a time) via a
    # streaming ParquetWriter, rather than accumulated in a single Python list
    # for the whole run - each output matrix is flattened to a 40000-element
    # list, and holding all combos x rules x subject-runs in memory at once
    # (tens of thousands of rows) was a major standing memory cost.
    func_id = 0
    model_cache: dict[str, tuple] = {}
    parquet_writer: pq.ParquetWriter | None = None
    total_rows_written = 0

    for n in sorted(top_n):
        info = group_info[n]
        combo_keys = info["combo_keys"]
        weights = info["weights"]
        reference_key = info["reference_key"]
        overlap_subject_set = info["overlap_subject_set"]
        sub_to_idx = info["sub_to_idx"]

        for key in combo_keys:
            if key not in model_cache:
                model_cache[key] = load_flavor_models([key], manifest)[key]
        models = {key: model_cache[key] for key in combo_keys}

        # Build a {(sub, ses, run): mat} lookup per combo instead of joining
        # DataFrames on (sub, ses, run) - polars .join() on Object-dtype "mat"
        # columns copies the underlying data at every sequential join step
        # (5-15 joins compounding), which was the single biggest memory cost
        # in this pipeline. Plain dict lookups just carry references to the
        # same numpy arrays already in combo_raw_cache - no copying.
        combo_lookup: dict[str, dict[tuple, np.ndarray]] = {}
        combo_keysets: dict[str, set[tuple]] = {}
        for key in combo_keys:
            df = _load_combo_run_df(combo_raw_cache[key], overlap_subject_set)
            lookup = dict(zip(zip(df["sub"], df["ses"], df["run"]), df["mat"]))
            combo_lookup[key] = lookup
            combo_keysets[key] = set(lookup.keys())

        # Only (sub, ses, run) triples where EVERY constituent method succeeded
        # are ensembled.
        common_keys = sorted(set.intersection(*combo_keysets.values())) if combo_keysets else []
        n_runs_available = len(common_keys)
        logging.info(
            f"top{n}: {n_runs_available} (sub, ses, run) rows with all "
            f"{len(combo_keys)} constituent methods present."
        )

        for rule_name in rule:
            func_name = f"top{n}_{rule_name}"
            logging.info(
                f"Building ensemble matrices for method=ensemble func={func_name} "
                f"({n_runs_available} subject-runs, {len(combo_keys)} constituent methods)"
            )
            rule_rows = []
            for sub, ses, run in common_keys:
                combo_matrices = {
                    key: import_matrix(combo_lookup[key][(sub, ses, run)]) for key in combo_keys
                }

                t0 = time.time()
                success = True
                try:
                    ensemble_mat = apply_ensemble_rule(
                        rule_name,
                        combo_matrices,
                        models,
                        parc=PARC_SIZE,
                        weights=weights,
                        reference_key=reference_key,
                    )
                except Exception as exc:
                    logging.error(
                        f"Failed to build {func_name} for sub={sub} ses={ses} run={run}: {exc}"
                    )
                    ensemble_mat = np.zeros((PARC_SIZE, PARC_SIZE), dtype=np.float32)
                    success = False
                run_time = time.time() - t0

                rule_rows.append(
                    {
                        "method": "ensemble",
                        "func": func_name,
                        "func_id": func_id,
                        "sub": sub,
                        "sub_id": sub_to_idx[sub],
                        "ses": int(ses),
                        "run": int(run),
                        "success": success,
                        "run_time": float(run_time),
                        "mat": ensemble_mat.astype(np.float32).flatten().tolist(),
                        "lag": 0,
                    }
                )
            func_id += 1

            rule_table = pa.Table.from_pandas(pd.DataFrame(rule_rows), preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(out_path_full, rule_table.schema)
            parquet_writer.write_table(rule_table)
            total_rows_written += len(rule_rows)
            del rule_rows, rule_table

    if parquet_writer is not None:
        parquet_writer.close()
    logging.info(f"Wrote {total_rows_written} rows to {out_path_full}")



if __name__ == "__main__":
    typer.run(main)
