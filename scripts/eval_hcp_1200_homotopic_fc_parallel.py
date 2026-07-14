#!/usr/bin/env python
"""Run network-block homotopic FC evaluation in parallel across combinations."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import typer
from joblib import Parallel, delayed


_spec = importlib.util.spec_from_file_location(
    "homotopic_fc",
    Path(__file__).with_name("eval_hcp_1200_homotopic_fc.py"),
)
if _spec is None or _spec.loader is None:
    raise ImportError("Could not load eval_hcp_1200_homotopic_fc.py")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_combinations = _mod.load_combinations


def run_eval(
    method: str,
    func: str,
    lag: int,
    *,
    data_path: str,
    parc_size: int,
    reducer: str,
    use_abs: bool,
    include_skarf_lag1: bool,
    min_valid_sub_fraction: float,
    n_subjects: int | None,
    out_dir: str,
    n_jobs_per_task: int = 4,
) -> dict:
    env = os.environ.copy()

    env["OMP_NUM_THREADS"] = str(n_jobs_per_task)
    env["MKL_NUM_THREADS"] = str(n_jobs_per_task)
    env["OPENBLAS_NUM_THREADS"] = str(n_jobs_per_task)
    env["BLIS_NUM_THREADS"] = str(n_jobs_per_task)

    script_path = Path(__file__).with_name("eval_hcp_1200_homotopic_fc.py")
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
        "--reducer",
        reducer,
        "--min-valid-sub-fraction",
        str(min_valid_sub_fraction),
        "--out-dir",
        out_dir,
    ]

    if use_abs:
        cmd.append("--use-abs")
    else:
        cmd.append("--no-use-abs")

    if include_skarf_lag1:
        cmd.append("--include-skarf-lag1")
    else:
        cmd.append("--no-include-skarf-lag1")

    if n_subjects is not None:
        cmd.extend(["--n-subjects", str(n_subjects)])

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


def main(
    data_path: str = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet",
    parc_size: int = 200,
    reducer: str = "mean",
    use_abs: bool = True,
    include_skarf_lag1: bool = True,
    min_valid_sub_fraction: float = 0.9,
    n_subjects: int | None = None,
    max_combos: int | None = None,
    n_jobs_parallel: int = 5,
    n_jobs_per_task: int = 3,
    method_func_list: str | None = None,
    degenerate_lookup_path: str | None = None,
    out_dir: str | None = None,
):
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("ERROR: PROJECT_ROOT not set. Please source .env file.")
        sys.exit(1)

    project_root_path = Path(project_root)
    method_func_list_path = (
        project_root_path / "resources/sparse_prediction_method_func_list.txt"
        if method_func_list is None
        else Path(method_func_list)
    )
    degenerate_lookup = (
        project_root_path / "resources/matrix_degenerate_lookup.json"
        if degenerate_lookup_path is None
        else Path(degenerate_lookup_path)
    )
    out_dir_path = (
        project_root_path / "results/hcp_1200_homotopic_fc_parallel"
        if out_dir is None
        else Path(out_dir)
    )
    out_dir_path.mkdir(exist_ok=True, parents=True)

    combos, excluded = load_combinations(
        method_func_list_path,
        degenerate_lookup,
        include_skarf_lag1=include_skarf_lag1,
        max_combos=max_combos,
    )

    print(f"Total combinations to run: {len(combos)}")
    print(f"Excluded degenerate PySPI combinations: {len(excluded)}")
    print("  - All valid methods at lag=0")
    if include_skarf_lag1:
        print("  - skarf methods also at lag=1")
    print(
        "Strategy: deterministic network-block homotopy "
        f"({n_jobs_parallel} workers, {n_jobs_per_task} threads each)"
    )
    print("-" * 80)

    results = Parallel(n_jobs=n_jobs_parallel, verbose=10)(
        delayed(run_eval)(
            combo["method"],
            combo["func"],
            combo["lag"],
            data_path=data_path,
            parc_size=parc_size,
            reducer=reducer,
            use_abs=use_abs,
            include_skarf_lag1=include_skarf_lag1,
            min_valid_sub_fraction=min_valid_sub_fraction,
            n_subjects=n_subjects,
            out_dir=str(out_dir_path),
            n_jobs_per_task=n_jobs_per_task,
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

    summary = {
        "data_path": data_path,
        "parc_size": parc_size,
        "reducer": reducer,
        "use_abs": use_abs,
        "include_skarf_lag1": include_skarf_lag1,
        "min_valid_sub_fraction": min_valid_sub_fraction,
        "n_subjects": n_subjects,
        "max_combos": max_combos,
        "n_jobs_parallel": n_jobs_parallel,
        "n_jobs_per_task": n_jobs_per_task,
        "reference_type": "between_network_crosshemisphere",
        "analysis_tag": "networkblock-v2",
        "excluded_count": len(excluded),
        "successful": successful,
        "failed": failed,
        "results": results,
    }
    with (out_dir_path / "parallel_run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
