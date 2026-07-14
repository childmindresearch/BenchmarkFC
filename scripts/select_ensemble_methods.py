"""Select the top-N ranked FC methods to feed into the krakencoder ensembling module.

Reads the combined benchmark leaderboard produced by
notebooks/analyze_hcp_1200_benchmark_scores_combined.ipynb and picks the top N
(method, func, lag) combinations by ``maxnorm_rank_sum`` for each requested N.

Writes one CSV per N to resources/ensemble_method_lists/top{N}_methods.csv with
columns: rank, method, func, lag, combo_key, maxnorm_rank_sum, maxnorm_rank_order.

Usage:
    uv run python scripts/select_ensemble_methods.py
    uv run python scripts/select_ensemble_methods.py --top-n 5 --top-n 10 --top-n 15
"""

import logging
import os
from pathlib import Path

import pandas as pd
import typer

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

DEFAULT_RANKED_CSV = (
    "results/hcp_1200_benchmark_scores_combined_analysis/"
    "combined_benchmark_scores_ranked.csv"
)
DEFAULT_OUT_DIR = "resources/ensemble_method_lists"
DEFAULT_TOP_N = [5, 10, 15]
RANK_COLUMN = "maxnorm_rank_sum"
ORDER_COLUMN = "maxnorm_rank_order"


def main(
    ranked_csv: str | None = None,
    out_dir: str | None = None,
    top_n: list[int] = DEFAULT_TOP_N,
    rank_column: str = RANK_COLUMN,
    exclude_method: list[str] = ["ensemble"],
):
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))

    ranked_csv_path = (
        Path(ranked_csv) if ranked_csv is not None else project_root / DEFAULT_RANKED_CSV
    )
    out_dir_path = Path(out_dir) if out_dir is not None else project_root / DEFAULT_OUT_DIR
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if not ranked_csv_path.exists():
        raise FileNotFoundError(
            f"Ranked benchmark CSV not found: {ranked_csv_path}. "
            "Run notebooks/analyze_hcp_1200_benchmark_scores_combined.ipynb first."
        )

    logging.info(f"Loading ranked benchmark scores from {ranked_csv_path}")
    ranked = pd.read_csv(ranked_csv_path)

    required_cols = {"method", "func", "lag", "component", "combo_key", rank_column}
    missing = required_cols - set(ranked.columns)
    if missing:
        raise KeyError(f"Ranked CSV is missing expected columns: {sorted(missing)}")

    # Restrict to "combined" rows (full-matrix scores, not directional upper/lower),
    # and exclude any pre-existing ensemble rows so re-running this script after
    # ensembles have been benchmarked doesn't feed ensembles back into themselves.
    pool = ranked.loc[ranked["component"] == "combined"].copy()
    pool = pool.loc[~pool["method"].isin(exclude_method)].copy()
    pool = pool.dropna(subset=[rank_column])

    logging.info(f"Candidate pool size (combined, non-ensemble, non-null score): {len(pool)}")

    max_n = max(top_n)
    if len(pool) < max_n:
        logging.warning(
            f"Requested top-{max_n} but only {len(pool)} candidates available; "
            "will select as many as exist."
        )

    keep_cols = [
        "method", "func", "lag", "combo_key", "is_directed", "family",
        "display_name", rank_column,
    ]
    if ORDER_COLUMN in pool.columns:
        keep_cols.append(ORDER_COLUMN)
    keep_cols = [c for c in keep_cols if c in pool.columns]

    for n in sorted(top_n):
        top = pool.nlargest(n, rank_column).sort_values(rank_column, ascending=False)
        top = top[keep_cols].reset_index(drop=True)
        top.insert(0, "rank", range(1, len(top) + 1))

        out_path = out_dir_path / f"top{n}_methods.csv"
        top.to_csv(out_path, index=False)
        logging.info(f"Wrote top-{n} methods ({len(top)} rows) to {out_path}")
        logging.info(f"\n{top[['rank', 'method', 'func', 'lag', rank_column]].to_string(index=False)}")


if __name__ == "__main__":
    typer.run(main)
