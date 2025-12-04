#!/usr/bin/env python
"""
Run sparse behavioral prediction for all method/func combinations in parallel using joblib.

This script uses joblib.Parallel to run multiple predictions simultaneously on a single machine,
utilizing all available CPU cores efficiently.
"""

import os
import subprocess
import sys
from pathlib import Path

from joblib import Parallel, delayed


def run_prediction(method: str, func: str, data_path: str, n_jobs_per_task: int = 4) -> dict:
    """
    Run prediction for a single method/func combination.

    Args:
        method: Method name (pyspi or skarf)
        func: Function name
        data_path: Path to parquet data file
        n_jobs_per_task: Number of jobs to allocate per prediction task

    Returns:
        dict: Result status with method, func, and return code
    """
    # Set thread limits for this subprocess (if needed)
    # scikit-learn will use these to limit its internal parallelism
    env = os.environ.copy()
    # env["OMP_NUM_THREADS"] = str(n_jobs_per_task)
    # env["MKL_NUM_THREADS"] = str(n_jobs_per_task)
    # env["OPENBLAS_NUM_THREADS"] = str(n_jobs_per_task)
    # env["BLIS_NUM_THREADS"] = str(n_jobs_per_task)
    
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/eval_hcp_1200_sparse_behav_prediction.py",
        "--method",
        method,
        "--func",
        func,
        "--data-path",
        data_path,
        "--target",
        "Cognition",
        "--sparsity",
        "0.8",
        "--seed",
        "2142",
    ]

    print(f"[START] {method} {func}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode == 0:
        print(f"[SUCCESS] {method} {func}")
    else:
        print(f"[FAILED] {method} {func} (exit code: {result.returncode})")
        print(f"stderr: {result.stderr[:500]}")

    return {"method": method, "func": func, "returncode": result.returncode}


def main():
    # Load environment variables
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("ERROR: PROJECT_ROOT not set. Please source .env file.")
        sys.exit(1)

    # Paths
    method_func_list = Path(project_root) / "resources" / "sparse_prediction_method_func_list.txt"
    data_path = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet"

    # Load method/func combinations
    combinations = []
    with open(method_func_list) as f:
        for line in f:
            method, func = line.strip().split("\t")
            combinations.append((method, func))

    print(f"Total combinations to run: {len(combinations)}")
    print(f"Will use 6 parallel jobs (4 cores each = 24 cores total)")
    print("-" * 80)

    # Run in parallel with joblib
    # n_jobs=6 means 6 predictions running simultaneously
    # Each prediction gets 4 cores (via sklearn's internal parallelization)
    # Total: 6 × 4 = 24 cores utilized
    results = Parallel(n_jobs=6, verbose=10)(
        delayed(run_prediction)(method, func, data_path, n_jobs_per_task=4)
        for method, func in combinations
    )

    # Summary
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
                print(f"  - {r['method']} {r['func']}")


if __name__ == "__main__":
    main()
