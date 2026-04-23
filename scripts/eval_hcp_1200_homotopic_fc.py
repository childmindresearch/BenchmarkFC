"""Evaluate homotopic FC across PySPI and skarf methods on HCP-1200.

Compares mean/median homotopic FC strength using Schaefer parcellation ordering.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import typer
import yaml
from sklearn.utils import check_random_state

from arfcexp.homotopy import (
    summarize_heterotopic_null_fc_ranked,
    summarize_homotopic_fc_ranked,
    validate_schaefer_homotopic_pairs,
)
from arfcexp.matrices import (
    EfficientMatrixReader,
    load_avg_mats_and_impose_sparsity,
    load_symmetry_lookup,
)

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

MIN_VALID_SUB_FRACTION = 0.9


def infer_combo_is_directed(
    *,
    method: str,
    func: str,
    symmetry_lookup: dict[str, bool],
) -> bool:
    """Infer whether a method/function pair should be treated as directed.

    Directionality is derived strictly from matrix_symmetry_lookup.json.
    """
    key = f"{method}__{func}"
    if key not in symmetry_lookup:
        raise KeyError(
            f"Missing symmetry lookup key: {key}. "
            "Please update resources/matrix_symmetry_lookup.json before running."
        )
    return not bool(symmetry_lookup[key])


def load_method_func_pairs(method_func_path: Path) -> list[tuple[str, str]]:
    """Load method/function pairs from the standard tab-separated resource file."""
    base_pairs: list[tuple[str, str]] = []
    with method_func_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid line in method/function list: {line!r}")
            method_i, func_i = parts
            base_pairs.append((method_i, func_i))
    return base_pairs


def load_combinations(
    method_func_path: Path,
    degenerate_lookup_path: Path,
    *,
    include_skarf_lag1: bool,
    method: str | None = None,
    func: str | None = None,
    lag: int = 0,
    max_combos: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load, expand, and optionally filter benchmark combinations."""
    base_pairs = load_method_func_pairs(method_func_path)

    with degenerate_lookup_path.open() as f:
        degenerate_lookup = json.load(f)

    combos, excluded = build_combinations(
        base_pairs,
        degenerate_lookup,
        include_skarf_lag1=include_skarf_lag1,
    )
    combos = select_combinations(combos, method=method, func=func, lag=lag)

    if max_combos is not None:
        combos = combos[:max_combos]

    return combos, excluded


def build_output_dir(
    out_dir: Path,
    *,
    parc_size: int,
    reducer: str,
    use_abs: bool,
    include_skarf_lag1: bool,
    n_subjects: int | None,
    max_combos: int | None,
    perm_test: bool,
    seed: int,
    n_perm: int,
    method: str | None,
    func: str | None,
    lag: int,
) -> Path:
    """Build the canonical output directory path for a homotopic run."""
    sub_tag = f"subs-{n_subjects}" if n_subjects is not None else "subs-all"
    combo_tag = f"max-{max_combos}" if max_combos is not None else "max-all"
    selector_tag = "selector-all"
    if method is not None and func is not None:
        selector_lag = lag if method == "skarf" else 0
        selector_tag = f"selector-{method}__{func}__lag-{selector_lag}"

    return out_dir / (
        f"parc-{parc_size}__reducer-{reducer}__rankabs-{int(use_abs)}"
        f"__skarf-lag1-{int(include_skarf_lag1)}__{sub_tag}__{combo_tag}"
        f"__perm-{int(perm_test)}__seed-{seed}__nperm-{n_perm}"
        f"__{selector_tag}"
    )


def build_empty_homotopic_summary() -> dict[str, float]:
    return {
        "homotopic_score": np.nan,
        "homotopic_score_upper": np.nan,
        "homotopic_score_lower": np.nan,
    }


def build_empty_null_summary() -> dict[str, float]:
    return {
        "heterotopic_null_mean": np.nan,
        "heterotopic_null_std": np.nan,
        "heterotopic_null_q25": np.nan,
        "heterotopic_null_q75": np.nan,
        "heterotopic_null_mean_upper": np.nan,
        "heterotopic_null_mean_lower": np.nan,
        "heterotopic_null_q25_upper": np.nan,
        "heterotopic_null_q75_upper": np.nan,
        "heterotopic_null_q25_lower": np.nan,
        "heterotopic_null_q75_lower": np.nan,
    }


def evaluate_combo(
    combo: dict,
    *,
    data_path: Path,
    sub_list: list[str],
    reader: EfficientMatrixReader,
    symmetry_lookup: dict[str, bool],
    parc_size: int,
    reducer: str,
    use_abs: bool,
    min_valid_sub_fraction: float,
    perm_test: bool,
    n_perm: int,
    random_state: np.random.RandomState,
) -> tuple[dict, list[dict], dict | None]:
    """Evaluate one method/function(/lag) combination."""
    method = combo["method"]
    func = combo["func"]
    lag = combo["lag"]

    avg_mats_df = load_avg_mats_and_impose_sparsity(
        data_path,
        method,
        func,
        sub_list,
        sparsity=None,
        symmetry_lookup=symmetry_lookup,
        lag=lag,
        fill_missing=False,
        reader=reader,
    )

    counts = avg_mats_df["Count"].to_numpy()
    valid_mask = counts > 0
    valid_fraction = float(np.mean(valid_mask))

    combo_key = make_combo_key(method, func, lag)
    combo_status = "ok" if valid_fraction >= min_valid_sub_fraction else "insufficient_data"
    combo_record = {
        "method": method,
        "func": func,
        "lag": lag,
        "combo_key": combo_key,
        "status": combo_status,
        "valid_fraction": valid_fraction,
        "n_subjects": len(sub_list),
        "n_valid_subjects": int(valid_mask.sum()),
    }

    if combo_status != "ok":
        return combo_record, [], None

    is_directed = infer_combo_is_directed(method=method, func=func, symmetry_lookup=symmetry_lookup)
    combo_perm_seed = int(random_state.randint(1000, 1000000))

    values = []
    values_upper = []
    values_lower = []
    null_values = []
    null_values_upper = []
    null_values_lower = []
    score_records = []

    for sub, row in avg_mats_df.iterrows():
        count = int(row["Count"])
        mat = row["Matrix"]
        if count <= 0 or mat is None:
            summary = build_empty_homotopic_summary()
            null_summary = build_empty_null_summary()
        else:
            summary = summarize_homotopic_fc_ranked(
                mat,
                parc_size=parc_size,
                reducer=reducer,
                use_abs_rank=use_abs,
                is_directed=is_directed,
            )
            if perm_test:
                null_summary = summarize_heterotopic_null_fc_ranked(
                    mat,
                    parc_size=parc_size,
                    reducer=reducer,
                    use_abs_rank=use_abs,
                    is_directed=is_directed,
                    n_perm=n_perm,
                    seed=combo_perm_seed,
                )
            else:
                null_summary = build_empty_null_summary()

        score = float(summary["homotopic_score"])
        score_upper = float(summary["homotopic_score_upper"])
        score_lower = float(summary["homotopic_score_lower"])
        null_mean = float(null_summary["heterotopic_null_mean"])
        null_std = float(null_summary["heterotopic_null_std"])
        null_q25 = float(null_summary["heterotopic_null_q25"])
        null_q75 = float(null_summary["heterotopic_null_q75"])
        null_mean_upper = float(null_summary["heterotopic_null_mean_upper"])
        null_mean_lower = float(null_summary["heterotopic_null_mean_lower"])

        homotopic_minus_null = (
            float(score - null_mean)
            if np.isfinite(score) and np.isfinite(null_mean)
            else np.nan
        )
        homotopic_vs_null_z = (
            float((score - null_mean) / null_std)
            if np.isfinite(score) and np.isfinite(null_mean) and np.isfinite(null_std) and null_std > 0
            else np.nan
        )

        values.append(score)
        values_upper.append(score_upper)
        values_lower.append(score_lower)
        null_values.append(null_mean)
        null_values_upper.append(null_mean_upper)
        null_values_lower.append(null_mean_lower)
        score_records.append(
            {
                "sub": sub,
                "method": method,
                "func": func,
                "lag": lag,
                "combo_key": combo_key,
                "run_count": count,
                "homotopic_score": score,
                "homotopic_score_upper": score_upper,
                "homotopic_score_lower": score_lower,
                "heterotopic_null_mean": null_mean,
                "heterotopic_null_std": null_std,
                "heterotopic_null_q25": null_q25,
                "heterotopic_null_q75": null_q75,
                "heterotopic_null_mean_upper": null_mean_upper,
                "heterotopic_null_mean_lower": null_mean_lower,
                "heterotopic_null_q25_upper": float(null_summary["heterotopic_null_q25_upper"]),
                "heterotopic_null_q75_upper": float(null_summary["heterotopic_null_q75_upper"]),
                "heterotopic_null_q25_lower": float(null_summary["heterotopic_null_q25_lower"]),
                "heterotopic_null_q75_lower": float(null_summary["heterotopic_null_q75_lower"]),
                "homotopic_minus_null": homotopic_minus_null,
                "homotopic_vs_null_z": homotopic_vs_null_z,
                "perm_test": perm_test,
                "perm_seed": combo_perm_seed,
                "n_perm": n_perm,
                "is_directed": is_directed,
            }
        )

    values = np.asarray(values, dtype=np.float64)
    values_upper = np.asarray(values_upper, dtype=np.float64)
    values_lower = np.asarray(values_lower, dtype=np.float64)
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values_upper = np.asarray(null_values_upper, dtype=np.float64)
    null_values_lower = np.asarray(null_values_lower, dtype=np.float64)
    finite_mask = np.isfinite(values)
    finite_upper = np.isfinite(values_upper)
    finite_lower = np.isfinite(values_lower)
    finite_null = np.isfinite(null_values)
    finite_null_upper = np.isfinite(null_values_upper)
    finite_null_lower = np.isfinite(null_values_lower)

    method_summary_record = {
        "method": method,
        "func": func,
        "lag": lag,
        "combo_key": combo_key,
        "is_directed": is_directed,
        "n_finite": int(finite_mask.sum()),
        "mean": float(np.nanmean(values)) if np.any(finite_mask) else np.nan,
        "median": float(np.nanmedian(values)) if np.any(finite_mask) else np.nan,
        "std": float(np.nanstd(values, ddof=1)) if finite_mask.sum() > 1 else np.nan,
        "mean_upper": float(np.nanmean(values_upper)) if np.any(finite_upper) else np.nan,
        "median_upper": float(np.nanmedian(values_upper)) if np.any(finite_upper) else np.nan,
        "std_upper": float(np.nanstd(values_upper, ddof=1)) if finite_upper.sum() > 1 else np.nan,
        "mean_lower": float(np.nanmean(values_lower)) if np.any(finite_lower) else np.nan,
        "median_lower": float(np.nanmedian(values_lower)) if np.any(finite_lower) else np.nan,
        "std_lower": float(np.nanstd(values_lower, ddof=1)) if finite_lower.sum() > 1 else np.nan,
        "null_mean_across_subjects": float(np.nanmean(null_values)) if np.any(finite_null) else np.nan,
        "null_std_across_subjects": float(np.nanstd(null_values, ddof=1)) if finite_null.sum() > 1 else np.nan,
        "null_q25_across_subjects": float(np.nanpercentile(null_values, 25)) if np.any(finite_null) else np.nan,
        "null_q75_across_subjects": float(np.nanpercentile(null_values, 75)) if np.any(finite_null) else np.nan,
        "null_mean_upper_across_subjects": (
            float(np.nanmean(null_values_upper)) if np.any(finite_null_upper) else np.nan
        ),
        "null_mean_lower_across_subjects": (
            float(np.nanmean(null_values_lower)) if np.any(finite_null_lower) else np.nan
        ),
        "perm_test": perm_test,
        "perm_seed": combo_perm_seed,
        "n_perm": n_perm,
        "valid_fraction": valid_fraction,
    }
    return combo_record, score_records, method_summary_record


def run_homotopic_benchmark(
    data_path: str = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet",
    parc_size: int = 200,
    reducer: str = "mean",
    use_abs: bool = True,
    include_skarf_lag1: bool = True,
    method: str | None = None,
    func: str | None = None,
    lag: int = 0,
    min_valid_sub_fraction: float = MIN_VALID_SUB_FRACTION,
    n_subjects: int | None = None,
    max_combos: int | None = None,
    perm_test: bool = False,
    seed: int = 2142,
    n_perm: int = 200,
    out_dir: str | None = None,
) -> Path:
    import arfcexp.hcp

    if not validate_schaefer_homotopic_pairs(parc_size=parc_size):
        raise RuntimeError("Homotopic pair validation failed.")

    project_root = Path(os.environ["PROJECT_ROOT"])
    data_path_p = Path(data_path)
    if not data_path_p.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path_p}")

    method_func_path = project_root / "resources/sparse_prediction_method_func_list.txt"
    degenerate_lookup_path = project_root / "resources/matrix_degenerate_lookup.json"

    if out_dir is None:
        out_dir_root = project_root / "results/hcp_1200_homotopic_fc"
    else:
        out_dir_root = Path(out_dir)

    sub_list = arfcexp.hcp.load_hcp_subject_list()
    if n_subjects is not None:
        sub_list = sub_list[:n_subjects]

    combos, excluded = load_combinations(
        method_func_path,
        degenerate_lookup_path,
        include_skarf_lag1=include_skarf_lag1,
        method=method,
        func=func,
        lag=lag,
        max_combos=max_combos,
    )

    if len(combos) == 0:
        raise ValueError(
            "No method/function combinations selected. "
            "Check method/func/lag filters and degenerate exclusions."
        )

    out_dir_p = build_output_dir(
        out_dir_root,
        parc_size=parc_size,
        reducer=reducer,
        use_abs=use_abs,
        include_skarf_lag1=include_skarf_lag1,
        n_subjects=n_subjects,
        max_combos=max_combos,
        perm_test=perm_test,
        seed=seed,
        n_perm=n_perm,
        method=method,
        func=func,
        lag=lag,
    )

    if out_dir_p.exists():
        logging.info("Output already exists; exiting.")
        return out_dir_p
    out_dir_p.mkdir(exist_ok=True, parents=True)

    params = {
        "data_path": str(data_path_p),
        "parc_size": parc_size,
        "reducer": reducer,
        "use_abs": use_abs,
        "include_skarf_lag1": include_skarf_lag1,
        "method": method,
        "func": func,
        "lag": lag,
        "min_valid_sub_fraction": min_valid_sub_fraction,
        "n_subjects": n_subjects,
        "max_combos": max_combos,
        "perm_test": perm_test,
        "seed": seed,
        "n_perm": n_perm,
        "perm_strategy": "heterotopic_interhemi",
    }
    with (out_dir_p / "params.yaml").open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    logging.info(
        "Running homotopic FC benchmark with %d combinations (%d excluded).",
        len(combos),
        len(excluded),
    )
    logging.info(
        "Using rank-based homotopic scoring (global off-diagonal percentile ranks, rank abs=%s).",
        use_abs,
    )
    if perm_test:
        logging.info(
            "Heterotopic null enabled (n_perm=%d, seed=%d, strategy=heterotopic_interhemi).",
            n_perm,
            seed,
        )

    random_state = check_random_state(seed)
    symmetry_lookup = load_symmetry_lookup(project_root)
    reader = EfficientMatrixReader(data_path_p)

    combo_keys = [f"{combo['method']}__{combo['func']}" for combo in combos]
    missing_lookup = sorted({key for key in combo_keys if key not in symmetry_lookup})
    if missing_lookup:
        raise KeyError(
            "matrix_symmetry_lookup.json is missing keys for selected combinations: "
            + ", ".join(missing_lookup[:10])
            + (" ..." if len(missing_lookup) > 10 else "")
        )

    combo_records = []
    score_records = []
    method_summary_records = []

    for combo in combos:
        combo_record, combo_score_records, method_summary_record = evaluate_combo(
            combo,
            data_path=data_path_p,
            sub_list=sub_list,
            reader=reader,
            symmetry_lookup=symmetry_lookup,
            parc_size=parc_size,
            reducer=reducer,
            use_abs=use_abs,
            min_valid_sub_fraction=min_valid_sub_fraction,
            perm_test=perm_test,
            n_perm=n_perm,
            random_state=random_state,
        )
        combo_records.append(combo_record)
        score_records.extend(combo_score_records)
        if method_summary_record is not None:
            method_summary_records.append(method_summary_record)

    combo_df = pd.DataFrame(combo_records)
    scores_df = pd.DataFrame(score_records)
    method_summary_df = pd.DataFrame(method_summary_records)
    excluded_df = pd.DataFrame(excluded)

    if len(combo_df) > 0:
        combo_df.sort_values(["method", "func", "lag"], inplace=True)
    if len(scores_df) > 0:
        scores_df.sort_values(["method", "func", "lag", "sub"], inplace=True)
    if len(method_summary_df) > 0:
        method_summary_df.sort_values(["method", "func", "lag"], inplace=True)

    combo_df.to_csv(out_dir_p / "combination_status.csv", index=False)
    excluded_df.to_csv(out_dir_p / "excluded_combinations.csv", index=False)
    scores_df.to_parquet(out_dir_p / "subject_homotopic_scores.parquet", index=False)
    method_summary_df.to_csv(out_dir_p / "method_homotopic_summary.csv", index=False)

    stats_summary = compute_stats(scores_df, method_summary_df)
    with (out_dir_p / "stats_summary.json").open("w") as f:
        json.dump(stats_summary, f, indent=2)

    if len(scores_df) > 0:
        family_subject_df = compute_family_subject_scores(scores_df)
        family_subject_df.to_csv(out_dir_p / "family_subject_scores.csv", index=False)

    logging.info("Saved homotopic FC benchmark outputs to %s", out_dir_p)
    return out_dir_p


def main(
    data_path: str = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet",
    parc_size: int = 200,
    reducer: str = "mean",
    use_abs: bool = True,
    include_skarf_lag1: bool = True,
    method: str | None = None,
    func: str | None = None,
    lag: int = 0,
    min_valid_sub_fraction: float = MIN_VALID_SUB_FRACTION,
    n_subjects: int | None = None,
    max_combos: int | None = None,
    perm_test: bool = False,
    seed: int = 2142,
    n_perm: int = 200,
    out_dir: str | None = None,
):
    run_homotopic_benchmark(
        data_path=data_path,
        parc_size=parc_size,
        reducer=reducer,
        use_abs=use_abs,
        include_skarf_lag1=include_skarf_lag1,
        method=method,
        func=func,
        lag=lag,
        min_valid_sub_fraction=min_valid_sub_fraction,
        n_subjects=n_subjects,
        max_combos=max_combos,
        perm_test=perm_test,
        seed=seed,
        n_perm=n_perm,
        out_dir=out_dir,
    )


def make_combo_key(method: str, func: str, lag: int) -> str:
    if method == "skarf":
        return f"{method}__{func}__lag-{lag}"
    return f"{method}__{func}"


def build_combinations(
    base_pairs: list[tuple[str, str]],
    degenerate_lookup: dict[str, bool],
    *,
    include_skarf_lag1: bool,
) -> tuple[list[dict], list[dict]]:
    combos = []
    excluded = []

    for method, func in base_pairs:
        key = f"{method}__{func}"
        if method == "pyspi" and degenerate_lookup.get(key, False):
            excluded.append(
                {
                    "method": method,
                    "func": func,
                    "reason": "degenerate_matrix",
                }
            )
            continue

        combos.append({"method": method, "func": func, "lag": 0})
        if method == "skarf" and include_skarf_lag1:
            combos.append({"method": method, "func": func, "lag": 1})

    return combos, excluded


def select_combinations(
    combos: list[dict],
    *,
    method: str | None,
    func: str | None,
    lag: int,
) -> list[dict]:
    """Select a specific method/function/lag subset when requested."""
    if (method is None) != (func is None):
        raise ValueError("method and func must be provided together.")

    if method is None:
        return combos

    selected_lag = lag if method == "skarf" else 0
    return [
        combo
        for combo in combos
        if combo["method"] == method and combo["func"] == func and combo["lag"] == selected_lag
    ]


def compute_family_subject_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    if len(scores_df) == 0:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "sub",
                    "method",
                    "family_mean_score",
                    "family_mean_score_upper",
                    "family_mean_score_lower",
                ]
            )
        )

    valid_df = scores_df.dropna(subset=["homotopic_score"]).copy()
    agg_cols = {
        "homotopic_score": "family_mean_score",
    }
    if "homotopic_score_upper" in valid_df.columns:
        agg_cols["homotopic_score_upper"] = "family_mean_score_upper"
    if "homotopic_score_lower" in valid_df.columns:
        agg_cols["homotopic_score_lower"] = "family_mean_score_lower"

    out = valid_df.groupby(["sub", "method"], as_index=False)[list(agg_cols.keys())].mean()
    out.rename(columns=agg_cols, inplace=True)
    return out


def compute_stats(scores_df: pd.DataFrame, method_summary_df: pd.DataFrame) -> dict:
    out = {
        "family_subject_test": None,
        "method_distribution_test": None,
        "skarf_lag_test": None,
    }

    family_subject_df = compute_family_subject_scores(scores_df)
    if len(family_subject_df) > 0:
        pivot = family_subject_df.pivot(index="sub", columns="method", values="family_mean_score")
        if "pyspi" in pivot.columns and "skarf" in pivot.columns:
            paired = pivot[["pyspi", "skarf"]].dropna()
            if len(paired) > 0:
                py = paired["pyspi"].to_numpy(dtype=np.float64)
                sk = paired["skarf"].to_numpy(dtype=np.float64)
                test = scipy.stats.wilcoxon(py, sk, alternative="two-sided")
                out["family_subject_test"] = {
                    "n": int(len(paired)),
                    "wilcoxon_stat": float(test.statistic),
                    "pvalue": float(test.pvalue),
                    "mean_pyspi": float(np.mean(py)),
                    "mean_skarf": float(np.mean(sk)),
                    "mean_diff_pyspi_minus_skarf": float(np.mean(py - sk)),
                    "cohen_dz": float(cohen_d_paired(py, sk)),
                }

    if len(method_summary_df) > 0:
        py = method_summary_df.loc[method_summary_df["method"] == "pyspi", "mean"].to_numpy(
            dtype=np.float64
        )
        sk = method_summary_df.loc[method_summary_df["method"] == "skarf", "mean"].to_numpy(
            dtype=np.float64
        )
        py = py[np.isfinite(py)]
        sk = sk[np.isfinite(sk)]
        if len(py) > 0 and len(sk) > 0:
            test = scipy.stats.mannwhitneyu(py, sk, alternative="two-sided")
            out["method_distribution_test"] = {
                "n_pyspi": int(len(py)),
                "n_skarf": int(len(sk)),
                "mannwhitney_u": float(test.statistic),
                "pvalue": float(test.pvalue),
                "mean_pyspi": float(np.mean(py)),
                "mean_skarf": float(np.mean(sk)),
                "cohen_d": float(cohen_d_independent(py, sk)),
                "cliffs_delta": float(cliffs_delta(py, sk)),
            }

        # Compare skarf lag-0 vs lag-1 per function (paired over function names).
        skarf = method_summary_df.query("method == 'skarf'").copy()
        if len(skarf) > 0:
            pivot = skarf.pivot(index="func", columns="lag", values="mean")
            if 0 in pivot.columns and 1 in pivot.columns:
                paired = pivot[[0, 1]].dropna()
                if len(paired) > 0:
                    lag0 = paired[0].to_numpy(dtype=np.float64)
                    lag1 = paired[1].to_numpy(dtype=np.float64)
                    test = scipy.stats.wilcoxon(lag0, lag1, alternative="two-sided")
                    out["skarf_lag_test"] = {
                        "n": int(len(paired)),
                        "wilcoxon_stat": float(test.statistic),
                        "pvalue": float(test.pvalue),
                        "mean_lag0": float(np.mean(lag0)),
                        "mean_lag1": float(np.mean(lag1)),
                        "cohen_dz": float(cohen_d_paired(lag0, lag1)),
                    }

    return out


def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    sd = np.std(diff, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float(np.mean(diff) / sd)


def cohen_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan

    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return np.nan

    comparisons = np.subtract.outer(x, y)
    n_pos = int(np.sum(comparisons > 0))
    n_neg = int(np.sum(comparisons < 0))
    return float((n_pos - n_neg) / (len(x) * len(y)))


if __name__ == "__main__":
    typer.run(main)