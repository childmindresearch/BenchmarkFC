#!/usr/bin/env python
"""Evaluate weight-distance correlations across PySPI and skarf methods.

This benchmark follows Liu et al. (Nature Methods, 2025) and the referenced
FC-PySPI implementation by computing raw signed Spearman correlations between
Schaefer-200 parcel centroid distance and FC edge weights. Directed matrices are
scored separately for upper and lower triangles, plus a combined off-diagonal
summary. Alexander-Bloch spin nulls are reused through cached parcel-index
permutations and permuted distance rank vectors.
"""

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import typer
import yaml

import arfcexp.hcp
from arfcexp.benchmark_utils import (
    infer_combo_is_directed,
    load_combinations,
    make_combo_key,
)
from arfcexp.matrices import (
    EfficientMatrixReader,
    load_avg_mats_and_impose_sparsity,
    load_symmetry_lookup,
)
from arfcexp.schaefer_metadata import find_schaefer_dlabel
from arfcexp.weight_distance import (
    WeightDistanceCache,
    parcel_info_to_frame,
    prepare_weight_distance_cache,
    score_weight_distance_matrix,
)


logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


DEFAULT_DATA_PATH = "/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet"
MIN_VALID_SUB_FRACTION = 0.9
NULL_SUFFIXES = (
    "mean",
    "std",
    "q025",
    "q25",
    "q50",
    "q75",
    "q975",
    "p_two_sided",
    "z",
)
SCORE_COLUMNS = (
    "weight_distance_score",
    "weight_distance_score_upper",
    "weight_distance_score_lower",
    "weight_distance_score_offdiag",
)
NULL_PREFIXES = (
    "spin_null",
    "spin_null_upper",
    "spin_null_lower",
    "spin_null_offdiag",
)


def build_output_dir(
    out_dir: Path,
    *,
    parc_size: int,
    include_skarf_lag1: bool,
    n_subjects: int | None,
    max_combos: int | None,
    seed: int,
    n_perm: int,
    method: str | None,
    func: str | None,
    lag: int,
) -> Path:
    """Build the canonical output directory path for a weight-distance run."""
    sub_tag = f"subs-{n_subjects}" if n_subjects is not None else "subs-all"
    combo_tag = f"max-{max_combos}" if max_combos is not None else "max-all"
    selector_tag = "selector-all"
    if method is not None and func is not None:
        selector_lag = lag if method == "skarf" else 0
        selector_tag = f"selector-{method}__{func}__lag-{selector_lag}"

    return out_dir / (
        f"parc-{parc_size}__metric-rawspearman__null-alexanderbloch"
        f"__skarf-lag1-{int(include_skarf_lag1)}__{sub_tag}__{combo_tag}"
        f"__seed-{seed}__nperm-{n_perm}__{selector_tag}"
    )


def empty_weight_distance_summary(is_directed: bool) -> dict[str, float | bool]:
    """Return NaN-filled score/null columns for missing matrices."""
    out: dict[str, float | bool] = {
        "weight_distance_score": np.nan,
        "weight_distance_score_upper": np.nan,
        "weight_distance_score_lower": np.nan,
        "weight_distance_score_offdiag": np.nan,
        "is_directed": bool(is_directed),
    }
    for prefix in NULL_PREFIXES:
        for suffix in NULL_SUFFIXES:
            out[f"{prefix}_{suffix}"] = np.nan
    return out


def summarize_score_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize subject-level scores per method/function/lag."""
    if len(scores_df) == 0:
        return pd.DataFrame()

    rows = []
    for (method, func, lag, combo_key), group in scores_df.groupby(
        ["method", "func", "lag", "combo_key"], sort=False
    ):
        row = {
            "method": method,
            "func": func,
            "lag": int(lag),
            "combo_key": combo_key,
            "is_directed": bool(group["is_directed"].iloc[0]),
            "n_subjects": int(len(group)),
            "n_valid_subjects": int((group["run_count"] > 0).sum()),
        }

        for score_col in SCORE_COLUMNS:
            values = group[score_col].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            suffix = score_col.replace("weight_distance_score", "score")
            row[f"{suffix}_n_finite"] = int(finite.sum())
            row[f"{suffix}_mean"] = float(np.nanmean(values)) if np.any(finite) else np.nan
            row[f"{suffix}_median"] = float(np.nanmedian(values)) if np.any(finite) else np.nan
            row[f"{suffix}_std"] = (
                float(np.nanstd(values, ddof=1)) if finite.sum() > 1 else np.nan
            )
            row[f"{suffix}_q25"] = float(np.nanpercentile(values, 25)) if np.any(finite) else np.nan
            row[f"{suffix}_q75"] = float(np.nanpercentile(values, 75)) if np.any(finite) else np.nan

        for prefix in NULL_PREFIXES:
            for suffix in ("mean", "std", "q25", "q75", "p_two_sided", "z"):
                col = f"{prefix}_{suffix}"
                values = group[col].to_numpy(dtype=np.float64)
                finite = np.isfinite(values)
                row[f"{col}_across_subjects"] = (
                    float(np.nanmean(values)) if np.any(finite) else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def score_group_mean_matrix(
    avg_mats_df: pd.DataFrame,
    *,
    combo: dict,
    is_directed: bool,
    cache: WeightDistanceCache,
) -> dict | None:
    """Score the paper-style group-mean matrix for one combination."""
    valid_mats = [
        row["Matrix"]
        for _, row in avg_mats_df.iterrows()
        if int(row["Count"]) > 0 and row["Matrix"] is not None
    ]
    if len(valid_mats) == 0:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        group_mean_mat = np.nanmean(np.stack(valid_mats), axis=0)
    summary = score_weight_distance_matrix(
        group_mean_mat,
        cache.geometry.distance_matrix,
        is_directed=is_directed,
        null_ranks_upper=cache.null_ranks_upper,
        null_ranks_lower=cache.null_ranks_lower,
        null_ranks_offdiag=cache.null_ranks_offdiag,
    )
    method = combo["method"]
    func = combo["func"]
    lag = combo["lag"]
    return {
        "method": method,
        "func": func,
        "lag": lag,
        "combo_key": make_combo_key(method, func, lag),
        "n_valid_subjects": len(valid_mats),
        **summary,
    }


def evaluate_combo(
    combo: dict,
    *,
    data_path: Path,
    sub_list: list[str],
    reader: EfficientMatrixReader,
    symmetry_lookup: dict[str, bool],
    cache: WeightDistanceCache,
    min_valid_sub_fraction: float,
) -> tuple[dict, list[dict], dict | None]:
    """Evaluate one method/function(/lag) combination."""
    method = combo["method"]
    func = combo["func"]
    lag = combo["lag"]
    combo_key = make_combo_key(method, func, lag)
    is_directed = infer_combo_is_directed(
        method=method,
        func=func,
        symmetry_lookup=symmetry_lookup,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
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

    counts = avg_mats_df["Count"].to_numpy(dtype=np.int64)
    valid_mask = counts > 0
    valid_fraction = float(np.mean(valid_mask)) if len(valid_mask) else 0.0
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
        "is_directed": is_directed,
    }

    if combo_status != "ok":
        return combo_record, [], None

    score_records = []
    for sub, row in avg_mats_df.iterrows():
        count = int(row["Count"])
        mat = row["Matrix"]
        if count <= 0 or mat is None:
            summary = empty_weight_distance_summary(is_directed)
        else:
            summary = score_weight_distance_matrix(
                mat,
                cache.geometry.distance_matrix,
                is_directed=is_directed,
                null_ranks_upper=cache.null_ranks_upper,
                null_ranks_lower=cache.null_ranks_lower,
                null_ranks_offdiag=cache.null_ranks_offdiag,
            )

        score_records.append(
            {
                "sub": sub,
                "method": method,
                "func": func,
                "lag": lag,
                "combo_key": combo_key,
                "run_count": count,
                "n_perm": cache.n_perm,
                "spin_seed": cache.seed,
                **summary,
            }
        )

    group_record = score_group_mean_matrix(
        avg_mats_df,
        combo=combo,
        is_directed=is_directed,
        cache=cache,
    )
    return combo_record, score_records, group_record


def compute_family_subject_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Average subject scores within each method family for paired comparisons."""
    if len(scores_df) == 0:
        return pd.DataFrame(columns=["sub", "method", "family_mean_score"])

    valid_df = scores_df.dropna(subset=["weight_distance_score"]).copy()
    if len(valid_df) == 0:
        return pd.DataFrame(columns=["sub", "method", "family_mean_score"])
    out = valid_df.groupby(["sub", "method"], as_index=False)["weight_distance_score"].mean()
    out.rename(columns={"weight_distance_score": "family_mean_score"}, inplace=True)
    return out


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.size < 2:
        return np.nan
    std = np.std(diff, ddof=1)
    return float(np.mean(diff) / std) if std > 0 else np.nan


def cohen_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return np.nan
    pooled = np.sqrt(((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else np.nan


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.nan
    greater = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    return float((greater - less) / (a.size * b.size))


def compute_aggregate_stats(scores_df: pd.DataFrame, method_summary_df: pd.DataFrame) -> dict:
    """Compute lightweight aggregate tests for exploratory reporting."""
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

    if len(method_summary_df) > 0 and "score_mean" in method_summary_df.columns:
        py = method_summary_df.loc[
            method_summary_df["method"] == "pyspi", "score_mean"
        ].to_numpy(dtype=np.float64)
        sk = method_summary_df.loc[
            method_summary_df["method"] == "skarf", "score_mean"
        ].to_numpy(dtype=np.float64)
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

        skarf = method_summary_df.query("method == 'skarf'").copy()
        if len(skarf) > 0:
            pivot = skarf.pivot(index="func", columns="lag", values="score_mean")
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
                        "mean_diff_lag0_minus_lag1": float(np.mean(lag0 - lag1)),
                        "cohen_dz": float(cohen_d_paired(lag0, lag1)),
                    }
    return out


def write_geometry_outputs(cache: WeightDistanceCache, out_dir: Path) -> None:
    """Save geometry files alongside benchmark outputs."""
    parcel_info_to_frame(cache.geometry.parcels, cache.geometry.centroids).to_csv(
        out_dir / "parcel_centroids.csv",
        index=False,
    )
    np.save(out_dir / "parcel_distance_matrix.npy", cache.geometry.distance_matrix)
    if cache.spin_indices is not None:
        np.save(out_dir / "spin_indices.npy", cache.spin_indices)


def run_weight_distance_benchmark(
    data_path: str = DEFAULT_DATA_PATH,
    parc_size: int = 200,
    include_skarf_lag1: bool = True,
    method: str | None = None,
    func: str | None = None,
    lag: int = 0,
    min_valid_sub_fraction: float = MIN_VALID_SUB_FRACTION,
    n_subjects: int | None = None,
    max_combos: int | None = None,
    seed: int = 2142,
    n_perm: int = 1000,
    density: str = "32k",
    distance_surface: str = "midthickness",
    cache_dir: str | None = None,
    out_dir: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Run weight-distance evaluation for one or more combinations."""
    project_root = Path(os.environ["PROJECT_ROOT"])
    data_path_p = Path(data_path)
    if not data_path_p.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path_p}")
    if parc_size != 200:
        raise ValueError("The weight-distance benchmark is currently defined for Schaefer-200 only.")

    method_func_path = project_root / "resources/sparse_prediction_method_func_list.txt"
    degenerate_lookup_path = project_root / "resources/matrix_degenerate_lookup.json"
    out_dir_root = Path(out_dir) if out_dir is not None else project_root / "results/hcp_1200_weight_distance"
    out_dir_root.mkdir(exist_ok=True, parents=True)

    out_dir_p = build_output_dir(
        out_dir_root,
        parc_size=parc_size,
        include_skarf_lag1=include_skarf_lag1,
        n_subjects=n_subjects,
        max_combos=max_combos,
        seed=seed,
        n_perm=n_perm,
        method=method,
        func=func,
        lag=lag,
    )
    complete_marker = out_dir_p / "subject_weight_distance_scores.parquet"
    if complete_marker.exists() and not overwrite:
        logging.info("Output already exists; exiting: %s", out_dir_p)
        return out_dir_p
    out_dir_p.mkdir(exist_ok=True, parents=True)

    dlabel_path = find_schaefer_dlabel(project_root, parc_size=parc_size)
    cache_dir_p = Path(cache_dir) if cache_dir is not None else out_dir_root / "weight_distance_cache"
    cache = prepare_weight_distance_cache(
        dlabel_path,
        cache_dir_p,
        n_perm=n_perm,
        seed=seed,
        density=density,
        distance_surface=distance_surface,
    )
    write_geometry_outputs(cache, out_dir_p)

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

    params = {
        "data_path": str(data_path_p),
        "parc_size": parc_size,
        "include_skarf_lag1": include_skarf_lag1,
        "method": method,
        "func": func,
        "lag": lag,
        "min_valid_sub_fraction": min_valid_sub_fraction,
        "n_subjects": n_subjects,
        "max_combos": max_combos,
        "seed": seed,
        "n_perm": n_perm,
        "density": density,
        "distance_surface": distance_surface,
        "dlabel_path": str(dlabel_path),
        "cache_dir": str(cache_dir_p),
        "metric": "raw_signed_spearman_distance_weight_correlation",
        "null_model": "alexander_bloch_parcel_spin_permuted_distances",
    }
    with (out_dir_p / "params.yaml").open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    logging.info(
        "Running weight-distance benchmark with %d combinations (%d excluded).",
        len(combos),
        len(excluded),
    )
    logging.info("Using Schaefer-%d centroid Euclidean distances from %s.", parc_size, dlabel_path)
    logging.info("Using n_perm=%d Alexander-Bloch spin-null distance ranks.", n_perm)

    symmetry_lookup = load_symmetry_lookup(project_root)
    combo_keys = [f"{combo['method']}__{combo['func']}" for combo in combos]
    missing_lookup = sorted({key for key in combo_keys if key not in symmetry_lookup})
    if missing_lookup:
        raise KeyError(
            "matrix_symmetry_lookup.json is missing keys for selected combinations: "
            + ", ".join(missing_lookup[:10])
            + (" ..." if len(missing_lookup) > 10 else "")
        )

    reader = EfficientMatrixReader(data_path_p)
    combo_records = []
    score_records = []
    group_records = []
    for combo in combos:
        combo_record, combo_score_records, group_record = evaluate_combo(
            combo,
            data_path=data_path_p,
            sub_list=sub_list,
            reader=reader,
            symmetry_lookup=symmetry_lookup,
            cache=cache,
            min_valid_sub_fraction=min_valid_sub_fraction,
        )
        combo_records.append(combo_record)
        score_records.extend(combo_score_records)
        if group_record is not None:
            group_records.append(group_record)

    combo_df = pd.DataFrame(combo_records)
    scores_df = pd.DataFrame(score_records)
    method_summary_df = summarize_score_table(scores_df)
    group_df = pd.DataFrame(group_records)
    excluded_df = pd.DataFrame(excluded)

    if len(combo_df) > 0:
        combo_df.sort_values(["method", "func", "lag"], inplace=True)
    if len(scores_df) > 0:
        scores_df.sort_values(["method", "func", "lag", "sub"], inplace=True)
    if len(method_summary_df) > 0:
        method_summary_df.sort_values(["method", "func", "lag"], inplace=True)
    if len(group_df) > 0:
        group_df.sort_values(["method", "func", "lag"], inplace=True)

    combo_df.to_csv(out_dir_p / "combination_status.csv", index=False)
    excluded_df.to_csv(out_dir_p / "excluded_combinations.csv", index=False)
    scores_df.to_parquet(out_dir_p / "subject_weight_distance_scores.parquet", index=False)
    method_summary_df.to_csv(out_dir_p / "method_weight_distance_summary.csv", index=False)
    group_df.to_csv(out_dir_p / "group_weight_distance_scores.csv", index=False)

    aggregate_stats = compute_aggregate_stats(scores_df, method_summary_df)
    with (out_dir_p / "aggregate_stats.json").open("w") as f:
        json.dump(aggregate_stats, f, indent=2)

    logging.info("Saved weight-distance benchmark outputs to %s", out_dir_p)
    return out_dir_p


def main(
    data_path: str = DEFAULT_DATA_PATH,
    parc_size: int = 200,
    include_skarf_lag1: bool = True,
    method: str | None = None,
    func: str | None = None,
    lag: int = 0,
    min_valid_sub_fraction: float = MIN_VALID_SUB_FRACTION,
    n_subjects: int | None = None,
    max_combos: int | None = None,
    seed: int = 2142,
    n_perm: int = 1000,
    density: str = "32k",
    distance_surface: str = "midthickness",
    cache_dir: str | None = None,
    out_dir: str | None = None,
    overwrite: bool = False,
):
    run_weight_distance_benchmark(
        data_path=data_path,
        parc_size=parc_size,
        include_skarf_lag1=include_skarf_lag1,
        method=method,
        func=func,
        lag=lag,
        min_valid_sub_fraction=min_valid_sub_fraction,
        n_subjects=n_subjects,
        max_combos=max_combos,
        seed=seed,
        n_perm=n_perm,
        density=density,
        distance_surface=distance_surface,
        cache_dir=cache_dir,
        out_dir=out_dir,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    typer.run(main)