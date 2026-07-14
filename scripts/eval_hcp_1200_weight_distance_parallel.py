#!/usr/bin/env python
"""Run weight-distance evaluation in parallel across method/function combos."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer
import yaml
from joblib import Parallel, delayed

from arfcexp.benchmark_utils import load_combinations
from arfcexp.schaefer_metadata import find_schaefer_dlabel
from arfcexp.weight_distance import prepare_weight_distance_cache


_spec = importlib.util.spec_from_file_location(
    "weight_distance_eval",
    Path(__file__).with_name("eval_hcp_1200_weight_distance.py"),
)
if _spec is None or _spec.loader is None:
    raise ImportError("Could not load eval_hcp_1200_weight_distance.py")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_output_dir = _mod.build_output_dir
compute_aggregate_stats = _mod.compute_aggregate_stats


DEFAULT_DATA_PATH = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet"


def run_eval(
    method: str,
    func: str,
    lag: int,
    *,
    data_path: str,
    parc_size: int,
    include_skarf_lag1: bool,
    min_valid_sub_fraction: float,
    n_subjects: int | None,
    seed: int,
    n_perm: int,
    density: str,
    distance_surface: str,
    out_dir: str,
    cache_dir: str,
    threads_per_worker: int,
    overwrite: bool,
) -> dict:
    """Run one method/function/lag evaluation subprocess."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads_per_worker)
    env["MKL_NUM_THREADS"] = str(threads_per_worker)
    env["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    env["BLIS_NUM_THREADS"] = str(threads_per_worker)

    script_path = Path(__file__).with_name("eval_hcp_1200_weight_distance.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--method",
        method,
        "--func",
        func,
        "--lag",
        str(lag),
        "--data-path",
        data_path,
        "--parc-size",
        str(parc_size),
        "--min-valid-sub-fraction",
        str(min_valid_sub_fraction),
        "--seed",
        str(seed),
        "--n-perm",
        str(n_perm),
        "--density",
        density,
        "--distance-surface",
        distance_surface,
        "--cache-dir",
        cache_dir,
        "--out-dir",
        out_dir,
    ]

    if include_skarf_lag1:
        cmd.append("--include-skarf-lag1")
    else:
        cmd.append("--no-include-skarf-lag1")
    if n_subjects is not None:
        cmd.extend(["--n-subjects", str(n_subjects)])
    if overwrite:
        cmd.append("--overwrite")

    print(f"[START] {method} {func} lag={lag}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        print(f"[SUCCESS] {method} {func} lag={lag}")
    else:
        print(f"[FAILED] {method} {func} lag={lag} (exit code: {result.returncode})")
        if result.stderr:
            print(f"stderr: {result.stderr[-1500:]}")
        if result.stdout:
            print(f"stdout: {result.stdout[-1500:]}")

    return {
        "method": method,
        "func": func,
        "lag": lag,
        "returncode": result.returncode,
    }


def child_output_dir(
    out_dir: Path,
    *,
    parc_size: int,
    include_skarf_lag1: bool,
    n_subjects: int | None,
    seed: int,
    n_perm: int,
    method: str,
    func: str,
    lag: int,
) -> Path:
    """Return the output directory used by the single-run child process."""
    return build_output_dir(
        out_dir,
        parc_size=parc_size,
        include_skarf_lag1=include_skarf_lag1,
        n_subjects=n_subjects,
        max_combos=None,
        seed=seed,
        n_perm=n_perm,
        method=method,
        func=func,
        lag=lag,
    )


def merge_parallel_outputs(
    out_dir: Path,
    child_dirs: list[Path],
    *,
    cache_dir: Path,
    seed: int,
    n_perm: int,
) -> dict:
    """Merge per-combination child outputs into the parallel root bundle."""
    combo_frames = []
    excluded_frames = []
    score_frames = []
    summary_frames = []
    group_frames = []

    for run_dir in child_dirs:
        if (run_dir / "combination_status.csv").exists():
            combo_frames.append(pd.read_csv(run_dir / "combination_status.csv"))
        if (run_dir / "excluded_combinations.csv").exists():
            excluded_frames.append(pd.read_csv(run_dir / "excluded_combinations.csv"))
        if (run_dir / "subject_weight_distance_scores.parquet").exists():
            score_frames.append(pd.read_parquet(run_dir / "subject_weight_distance_scores.parquet"))
        if (run_dir / "method_weight_distance_summary.csv").exists():
            summary_frames.append(pd.read_csv(run_dir / "method_weight_distance_summary.csv"))
        if (run_dir / "group_weight_distance_scores.csv").exists():
            group_frames.append(pd.read_csv(run_dir / "group_weight_distance_scores.csv"))

    combo_df = pd.concat(combo_frames, ignore_index=True) if combo_frames else pd.DataFrame()
    excluded_df = pd.concat(excluded_frames, ignore_index=True).drop_duplicates() if excluded_frames else pd.DataFrame()
    scores_df = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    group_df = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()

    if len(combo_df) > 0:
        combo_df.sort_values(["method", "func", "lag"], inplace=True)
    if len(scores_df) > 0:
        scores_df.sort_values(["method", "func", "lag", "sub"], inplace=True)
    if len(summary_df) > 0:
        summary_df.sort_values(["method", "func", "lag"], inplace=True)
    if len(group_df) > 0:
        group_df.sort_values(["method", "func", "lag"], inplace=True)

    combo_df.to_csv(out_dir / "combination_status.csv", index=False)
    excluded_df.to_csv(out_dir / "excluded_combinations.csv", index=False)
    scores_df.to_parquet(out_dir / "subject_weight_distance_scores.parquet", index=False)
    summary_df.to_csv(out_dir / "method_weight_distance_summary.csv", index=False)
    group_df.to_csv(out_dir / "group_weight_distance_scores.csv", index=False)

    for name in ("parcel_centroids.csv", "parcel_distance_matrix.npy"):
        src = cache_dir / name
        if src.exists():
            shutil.copyfile(src, out_dir / name)
    spin_path = cache_dir / f"spin_indices_seed-{seed}_nperm-{n_perm}.npy"
    if spin_path.exists():
        shutil.copyfile(spin_path, out_dir / "spin_indices.npy")

    aggregate_stats = compute_aggregate_stats(scores_df, summary_df)
    with (out_dir / "aggregate_stats.json").open("w") as f:
        json.dump(aggregate_stats, f, indent=2)
    return aggregate_stats


def main(
    data_path: str = DEFAULT_DATA_PATH,
    parc_size: int = 200,
    include_skarf_lag1: bool = True,
    min_valid_sub_fraction: float = 0.9,
    n_subjects: int | None = None,
    max_combos: int | None = None,
    seed: int = 2142,
    n_perm: int = 1000,
    density: str = "32k",
    distance_surface: str = "midthickness",
    n_jobs_parallel: int = 8,
    threads_per_worker: int = 2,
    method_func_list: str | None = None,
    degenerate_lookup_path: str | None = None,
    out_dir: str | None = None,
    cache_dir: str | None = None,
    overwrite: bool = False,
):
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("ERROR: PROJECT_ROOT not set. Please source .env file.")
        raise SystemExit(1)
    if n_jobs_parallel < 1:
        raise ValueError(f"Expected n_jobs_parallel >= 1, got {n_jobs_parallel}.")
    if threads_per_worker < 1:
        raise ValueError(f"Expected threads_per_worker >= 1, got {threads_per_worker}.")

    project_root_path = Path(project_root)
    method_func_list_path = (
        Path(method_func_list)
        if method_func_list is not None
        else project_root_path / "resources/sparse_prediction_method_func_list.txt"
    )
    degenerate_lookup = (
        Path(degenerate_lookup_path)
        if degenerate_lookup_path is not None
        else project_root_path / "resources/matrix_degenerate_lookup.json"
    )
    out_dir_path = (
        Path(out_dir)
        if out_dir is not None
        else project_root_path / "results/hcp_1200_weight_distance_parallel"
    )
    out_dir_path.mkdir(exist_ok=True, parents=True)
    cache_dir_path = Path(cache_dir) if cache_dir is not None else out_dir_path / "weight_distance_cache"

    combos, excluded = load_combinations(
        method_func_list_path,
        degenerate_lookup,
        include_skarf_lag1=include_skarf_lag1,
        max_combos=max_combos,
    )
    if len(combos) == 0:
        raise ValueError("No method/function combinations selected.")

    dlabel_path = find_schaefer_dlabel(project_root_path, parc_size=parc_size)
    prepare_weight_distance_cache(
        dlabel_path,
        cache_dir_path,
        n_perm=n_perm,
        seed=seed,
        density=density,
        distance_surface=distance_surface,
    )

    params = {
        "data_path": data_path,
        "parc_size": parc_size,
        "include_skarf_lag1": include_skarf_lag1,
        "min_valid_sub_fraction": min_valid_sub_fraction,
        "n_subjects": n_subjects,
        "max_combos": max_combos,
        "seed": seed,
        "n_perm": n_perm,
        "density": density,
        "distance_surface": distance_surface,
        "n_jobs_parallel": n_jobs_parallel,
        "threads_per_worker": threads_per_worker,
        "method_func_list": str(method_func_list_path),
        "degenerate_lookup_path": str(degenerate_lookup),
        "dlabel_path": str(dlabel_path),
        "cache_dir": str(cache_dir_path),
        "metric": "raw_signed_spearman_distance_weight_correlation",
        "null_model": "alexander_bloch_parcel_spin_permuted_distances",
        "n_combinations": len(combos),
        "excluded_count": len(excluded),
    }
    with (out_dir_path / "params.yaml").open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    print(f"Total combinations to run: {len(combos)}")
    print(f"Excluded degenerate PySPI combinations: {len(excluded)}")
    print("  - All valid methods at lag=0")
    if include_skarf_lag1:
        print("  - skarf methods also at lag=1")
    print(
        f"Strategy: joblib subprocess wrapper ({n_jobs_parallel} workers, "
        f"thread cap={threads_per_worker})"
    )
    print(f"Shared cache: {cache_dir_path}")
    print("-" * 80)

    results = Parallel(n_jobs=n_jobs_parallel, verbose=10)(
        delayed(run_eval)(
            combo["method"],
            combo["func"],
            combo["lag"],
            data_path=data_path,
            parc_size=parc_size,
            include_skarf_lag1=include_skarf_lag1,
            min_valid_sub_fraction=min_valid_sub_fraction,
            n_subjects=n_subjects,
            seed=seed,
            n_perm=n_perm,
            density=density,
            distance_surface=distance_surface,
            out_dir=str(out_dir_path),
            cache_dir=str(cache_dir_path),
            threads_per_worker=threads_per_worker,
            overwrite=overwrite,
        )
        for combo in combos
    )

    print("-" * 80)
    print("SUMMARY:")
    successful = sum(1 for result in results if result["returncode"] == 0)
    failed = len(results) - successful
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    if failed > 0:
        print("\nFailed combinations:")
        for result in results:
            if result["returncode"] != 0:
                print(f"  - {result['method']} {result['func']} lag={result['lag']}")

    child_dirs = [
        child_output_dir(
            out_dir_path,
            parc_size=parc_size,
            include_skarf_lag1=include_skarf_lag1,
            n_subjects=n_subjects,
            seed=seed,
            n_perm=n_perm,
            method=combo["method"],
            func=combo["func"],
            lag=combo["lag"],
        )
        for combo in combos
    ]
    aggregate_stats = merge_parallel_outputs(
        out_dir_path,
        child_dirs,
        cache_dir=cache_dir_path,
        seed=seed,
        n_perm=n_perm,
    )

    summary = {
        "data_path": data_path,
        "parc_size": parc_size,
        "include_skarf_lag1": include_skarf_lag1,
        "min_valid_sub_fraction": min_valid_sub_fraction,
        "n_subjects": n_subjects,
        "max_combos": max_combos,
        "seed": seed,
        "n_perm": n_perm,
        "density": density,
        "distance_surface": distance_surface,
        "n_jobs_parallel": n_jobs_parallel,
        "threads_per_worker": threads_per_worker,
        "excluded_count": len(excluded),
        "successful": successful,
        "failed": failed,
        "aggregate_stats": aggregate_stats,
        "results": results,
    }
    with (out_dir_path / "parallel_run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    typer.run(main)