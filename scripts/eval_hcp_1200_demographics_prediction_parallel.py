"""Run demographics prediction in parallel across method/func combinations.

This wrapper delegates the actual combo execution to
scripts/eval_hcp_1200_demographics_prediction.py and only handles outer parallel
orchestration across combinations.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import typer
from joblib import Parallel, delayed

DEFAULT_SEED = 2142
DEFAULT_N_SPLITS = 20
DEFAULT_N_INNER_SPLITS = 20
DEFAULT_SPARSITY = 0.0
DEFAULT_PARC_SIZE = 200
DEFAULT_POOL = 3
DEFAULT_PERM_TEST = False
DEFAULT_N_JOBS_PARALLEL = 8
DEFAULT_THREADS_PER_WORKER = 2
DEFAULT_DATA_PATH = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet"


def run_prediction(
    method: str,
    func: str,
    lag: int,
    *,
    data_path: str,
    parc_size: int,
    pool: int,
    sparsity: float,
    n_splits: int,
    n_inner_splits: int,
    perm_test: bool,
    n_subjects: int | None,
    seed: int,
    threads_per_worker: int,
    out_dir: str,
) -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads_per_worker)
    env["MKL_NUM_THREADS"] = str(threads_per_worker)
    env["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    env["BLIS_NUM_THREADS"] = str(threads_per_worker)

    script_path = Path(__file__).with_name("eval_hcp_1200_demographics_prediction.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--task",
        "both",
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
        "--pool",
        str(pool),
        "--sparsity",
        str(sparsity),
        "--n-splits",
        str(n_splits),
        "--n-inner-splits",
        str(n_inner_splits),
        "--seed",
        str(seed),
        "--threads-per-worker",
        str(threads_per_worker),
        "--out-dir",
        out_dir,
    ]

    if perm_test:
        cmd.append("--perm-test")
    if n_subjects is not None:
        cmd.extend(["--n-subjects", str(n_subjects)])

    print(f"[START] {method} {func} lag={lag}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode == 0:
        print(f"[SUCCESS] {method} {func} lag={lag}")
    else:
        print(f"[FAILED] {method} {func} lag={lag} (exit code: {result.returncode})")
        if result.stderr:
            print(f"stderr: {result.stderr[-1000:]}")
        if result.stdout:
            print(f"stdout: {result.stdout[-1000:]}")

    return {
        "method": method,
        "func": func,
        "lag": lag,
        "returncode": result.returncode,
    }


def main(
    data_path: str = DEFAULT_DATA_PATH,
    n_jobs_parallel: int = DEFAULT_N_JOBS_PARALLEL,
    threads_per_worker: int = DEFAULT_THREADS_PER_WORKER,
    pool: int = DEFAULT_POOL,
    parc_size: int = DEFAULT_PARC_SIZE,
    sparsity: float = DEFAULT_SPARSITY,
    n_splits: int = DEFAULT_N_SPLITS,
    n_inner_splits: int = DEFAULT_N_INNER_SPLITS,
    perm_test: bool = DEFAULT_PERM_TEST,
    n_subjects: int | None = None,
    seed: int = DEFAULT_SEED,
    max_combos: int | None = None,
    method_func_list: str | None = None,
    out_dir: str | None = None,
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
        else project_root_path / "resources" / "sparse_prediction_method_func_list.txt"
    )
    out_dir_path = (
        Path(out_dir)
        if out_dir is not None
        else project_root_path / "results/hcp_1200_demographics_prediction"
    )
    out_dir_path.mkdir(exist_ok=True, parents=True)

    combinations = []
    with method_func_list_path.open() as f:
        for line in f:
            method, func = line.strip().split("\t")
            combinations.append((method, func, 0))
            if method == "skarf":
                combinations.append((method, func, 1))

    if max_combos is not None:
        combinations = combinations[:max_combos]

    print(f"Total method/func combinations: {len(combinations)}")
    print("Tasks per combination: gender, age")
    print(f"Total prediction runs: {len(combinations) * 2}")
    print(
        f"Strategy: joblib subprocess wrapper around eval_hcp_1200_demographics_prediction.py "
        f"({n_jobs_parallel} workers, thread cap={threads_per_worker})"
    )
    print("-" * 80)

    results = Parallel(n_jobs=n_jobs_parallel, verbose=10)(
        delayed(run_prediction)(
            method,
            func,
            lag,
            data_path=data_path,
            parc_size=parc_size,
            pool=pool,
            sparsity=sparsity,
            n_splits=n_splits,
            n_inner_splits=n_inner_splits,
            perm_test=perm_test,
            n_subjects=n_subjects,
            seed=seed,
            threads_per_worker=threads_per_worker,
            out_dir=str(out_dir_path),
        )
        for method, func, lag in combinations
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
        "n_jobs_parallel": n_jobs_parallel,
        "threads_per_worker": threads_per_worker,
        "pool": pool,
        "parc_size": parc_size,
        "sparsity": sparsity,
        "n_splits": n_splits,
        "n_inner_splits": n_inner_splits,
        "perm_test": perm_test,
        "n_subjects": n_subjects,
        "seed": seed,
        "max_combos": max_combos,
        "results": results,
    }
    with (out_dir_path / "parallel_run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
