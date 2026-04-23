#!/usr/bin/env python
"""Run homotopic FC evaluation in parallel using joblib.

This wrapper delegates the actual evaluation to
scripts/eval_hcp_1200_homotopic_fc.py and only handles outer parallel
orchestration across combinations.
"""

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
    perm_test: bool,
    seed: int,
    n_perm: int,
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
        "--seed",
        str(seed),
        "--n-perm",
        str(n_perm),
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

    if perm_test:
        cmd.append("--perm-test")
    else:
        cmd.append("--no-perm-test")

    print(f"[START] {method} {func} lag={lag} seed={seed}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode == 0:
        print(f"[SUCCESS] {method} {func} lag={lag} seed={seed}")
    else:
        print(
            f"[FAILED] {method} {func} lag={lag} seed={seed} "
            f"(exit code: {result.returncode})"
        )
        print(f"stderr: {result.stderr[:500]}")

    return {
        "method": method,
        "func": func,
        "lag": lag,
        "seed": seed,
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
    perm_test: bool = False,
    seed: int = 2142,
    n_perm: int = 200,
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

    project_root = Path(project_root)

    if method_func_list is None:
        method_func_list_path = project_root / "resources/sparse_prediction_method_func_list.txt"
    else:
        method_func_list_path = Path(method_func_list)

    if degenerate_lookup_path is None:
        degenerate_lookup = project_root / "resources/matrix_degenerate_lookup.json"
    else:
        degenerate_lookup = Path(degenerate_lookup_path)

    if out_dir is None:
        out_dir_path = project_root / "results/hcp_1200_homotopic_fc_parallel"
    else:
        out_dir_path = Path(out_dir)
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
    print(f"Will use {n_jobs_parallel} parallel jobs ({n_jobs_per_task} cores each)")
    print("-" * 80)

    combo_with_seed = [
        (combo["method"], combo["func"], combo["lag"], seed + idx)
        for idx, combo in enumerate(combos)
    ]

    results = Parallel(n_jobs=n_jobs_parallel, verbose=10)(
        delayed(run_eval)(
            method,
            func,
            lag,
            data_path=data_path,
            parc_size=parc_size,
            reducer=reducer,
            use_abs=use_abs,
            include_skarf_lag1=include_skarf_lag1,
            min_valid_sub_fraction=min_valid_sub_fraction,
            n_subjects=n_subjects,
            perm_test=perm_test,
            seed=combo_seed,
            n_perm=n_perm,
            out_dir=str(out_dir_path),
            n_jobs_per_task=n_jobs_per_task,
        )
        for method, func, lag, combo_seed in combo_with_seed
    )

    print("-" * 80)
    print("SUMMARY:")
    successful = sum(1 for r in results if r["returncode"] == 0)
    failed = len(results) - successful
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed combinations:")
        for r in results:
            if r["returncode"] != 0:
                print(f"  - {r['method']} {r['func']} lag={r['lag']}")

    summary = {
        "data_path": data_path,
        "parc_size": parc_size,
        "reducer": reducer,
        "use_abs": use_abs,
        "include_skarf_lag1": include_skarf_lag1,
        "min_valid_sub_fraction": min_valid_sub_fraction,
        "n_subjects": n_subjects,
        "max_combos": max_combos,
        "perm_test": perm_test,
        "seed": seed,
        "n_perm": n_perm,
        "n_jobs_parallel": n_jobs_parallel,
        "n_jobs_per_task": n_jobs_per_task,
        "excluded_count": len(excluded),
        "successful": successful,
        "failed": failed,
        "results": results,
    }

    with (out_dir_path / "parallel_run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
