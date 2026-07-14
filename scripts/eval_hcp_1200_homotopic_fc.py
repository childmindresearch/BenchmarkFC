"""Evaluate network-block homotopic FC across PySPI and skarf methods on HCP-1200.

The primary subject-level outputs are Schaefer-7 summaries computed from all
within-network cross-hemisphere edges. A derived overall observed summary is the
equal-weight mean of those seven network scores. The empirical reference is a
deterministic between-network cross-hemisphere contrast.
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

from arfcexp.benchmark_utils import infer_combo_is_directed, load_combinations, make_combo_key
from arfcexp.homotopy import (
    OVERALL_NETWORK_LABEL,
    get_schaefer_between_network_edges,
    get_schaefer_network_order,
    get_schaefer_overall_between_network_edges,
    get_schaefer_within_network_edges,
    rank_matrix_offdiagonal,
    summarize_between_network_reference_ranked_edges,
    summarize_homotopic_fc_ranked_edges,
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
ANALYSIS_LEVEL_NETWORK = "network"
ANALYSIS_LEVEL_OVERALL = "overall"
ANALYSIS_TAG = "networkblock-v2"

REFERENCE_COLUMNS = (
    "between_network_reference_mean",
    "between_network_reference_std",
    "between_network_reference_q25",
    "between_network_reference_q75",
    "between_network_reference_mean_upper",
    "between_network_reference_mean_lower",
    "between_network_reference_q25_upper",
    "between_network_reference_q75_upper",
    "between_network_reference_q25_lower",
    "between_network_reference_q75_lower",
)


def build_output_dir(
    out_dir: Path,
    *,
    parc_size: int,
    reducer: str,
    use_abs: bool,
    include_skarf_lag1: bool,
    n_subjects: int | None,
    max_combos: int | None,
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
        f"parc-{parc_size}__analysis-{ANALYSIS_TAG}__reducer-{reducer}"
        f"__rankabs-{int(use_abs)}__skarf-lag1-{int(include_skarf_lag1)}"
        f"__{sub_tag}__{combo_tag}__reference-betweennetwork__{selector_tag}"
    )


def build_empty_homotopic_summary(is_directed: bool) -> dict[str, float | bool]:
    return {
        "homotopic_score": np.nan,
        "homotopic_score_upper": np.nan,
        "homotopic_score_lower": np.nan,
        "is_directed": bool(is_directed),
    }


def build_empty_reference_summary(is_directed: bool) -> dict[str, float | bool]:
    out: dict[str, float | bool] = {key: np.nan for key in REFERENCE_COLUMNS}
    out["is_directed"] = bool(is_directed)
    return out


def _safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def average_network_summaries(
    summaries: list[dict[str, float | bool]],
    *,
    is_directed: bool,
) -> dict[str, float | bool]:
    """Average network-level observed summaries into one overall record."""
    return {
        "homotopic_score": _safe_nanmean([float(summary["homotopic_score"]) for summary in summaries]),
        "homotopic_score_upper": (
            _safe_nanmean([float(summary["homotopic_score_upper"]) for summary in summaries])
            if is_directed
            else np.nan
        ),
        "homotopic_score_lower": (
            _safe_nanmean([float(summary["homotopic_score_lower"]) for summary in summaries])
            if is_directed
            else np.nan
        ),
        "is_directed": bool(is_directed),
    }


def _finite_diff(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan


def build_score_record(
    *,
    sub: str,
    method: str,
    func: str,
    lag: int,
    combo_key: str,
    analysis_level: str,
    network: str,
    n_crosshemi_edges: int,
    n_between_network_edges: int,
    n_networks_aggregated: int,
    run_count: int,
    summary: dict[str, float | bool],
    reference_summary: dict[str, float | bool],
    is_directed: bool,
) -> dict[str, object]:
    score = float(summary["homotopic_score"])
    score_upper = float(summary["homotopic_score_upper"])
    score_lower = float(summary["homotopic_score_lower"])
    reference_mean = float(reference_summary["between_network_reference_mean"])
    reference_mean_upper = float(reference_summary["between_network_reference_mean_upper"])
    reference_mean_lower = float(reference_summary["between_network_reference_mean_lower"])

    return {
        "sub": sub,
        "method": method,
        "func": func,
        "lag": lag,
        "combo_key": combo_key,
        "analysis_level": analysis_level,
        "network": network,
        "n_crosshemi_edges": n_crosshemi_edges,
        "n_between_network_edges": n_between_network_edges,
        "n_networks_aggregated": n_networks_aggregated,
        "run_count": run_count,
        "homotopic_score": score,
        "homotopic_score_upper": score_upper,
        "homotopic_score_lower": score_lower,
        **{key: float(reference_summary[key]) for key in REFERENCE_COLUMNS},
        "within_minus_between": _finite_diff(score, reference_mean),
        "within_minus_between_upper": _finite_diff(score_upper, reference_mean_upper),
        "within_minus_between_lower": _finite_diff(score_lower, reference_mean_lower),
        "reference_type": "between_network_crosshemisphere",
        "is_directed": is_directed,
    }


def _summarize_values(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    return {
        "n_finite": int(finite.sum()),
        "mean": float(np.nanmean(values)) if np.any(finite) else np.nan,
        "median": float(np.nanmedian(values)) if np.any(finite) else np.nan,
        "std": float(np.nanstd(values, ddof=1)) if finite.sum() > 1 else np.nan,
        "q25": float(np.nanpercentile(values, 25)) if np.any(finite) else np.nan,
        "q75": float(np.nanpercentile(values, 75)) if np.any(finite) else np.nan,
    }


def summarize_combo_scores(
    score_records: list[dict[str, object]],
    *,
    method: str,
    func: str,
    lag: int,
    combo_key: str,
    is_directed: bool,
    valid_fraction: float,
) -> list[dict[str, object]]:
    """Aggregate subject-level rows into method summaries per analysis level."""
    if not score_records:
        return []

    scores_df = pd.DataFrame(score_records)
    summary_records = []
    for (analysis_level, network), group in scores_df.groupby(
        ["analysis_level", "network"],
        sort=False,
    ):
        values = group["homotopic_score"].to_numpy(dtype=np.float64)
        values_upper = group["homotopic_score_upper"].to_numpy(dtype=np.float64)
        values_lower = group["homotopic_score_lower"].to_numpy(dtype=np.float64)
        reference_values = group["between_network_reference_mean"].to_numpy(dtype=np.float64)
        reference_upper = group["between_network_reference_mean_upper"].to_numpy(dtype=np.float64)
        reference_lower = group["between_network_reference_mean_lower"].to_numpy(dtype=np.float64)
        delta_values = group["within_minus_between"].to_numpy(dtype=np.float64)
        delta_upper = group["within_minus_between_upper"].to_numpy(dtype=np.float64)
        delta_lower = group["within_minus_between_lower"].to_numpy(dtype=np.float64)

        observed = _summarize_values(values)
        observed_upper = _summarize_values(values_upper)
        observed_lower = _summarize_values(values_lower)
        reference = _summarize_values(reference_values)
        reference_upper_summary = _summarize_values(reference_upper)
        reference_lower_summary = _summarize_values(reference_lower)
        delta = _summarize_values(delta_values)
        delta_upper_summary = _summarize_values(delta_upper)
        delta_lower_summary = _summarize_values(delta_lower)

        summary_records.append(
            {
                "method": method,
                "func": func,
                "lag": lag,
                "combo_key": combo_key,
                "analysis_level": analysis_level,
                "network": network,
                "is_directed": is_directed,
                "n_finite": observed["n_finite"],
                "mean": observed["mean"],
                "median": observed["median"],
                "std": observed["std"],
                "q25": observed["q25"],
                "q75": observed["q75"],
                "mean_upper": observed_upper["mean"],
                "median_upper": observed_upper["median"],
                "std_upper": observed_upper["std"],
                "mean_lower": observed_lower["mean"],
                "median_lower": observed_lower["median"],
                "std_lower": observed_lower["std"],
                "between_reference_mean_across_subjects": reference["mean"],
                "between_reference_std_across_subjects": reference["std"],
                "between_reference_q25_across_subjects": reference["q25"],
                "between_reference_q75_across_subjects": reference["q75"],
                "between_reference_mean_upper_across_subjects": reference_upper_summary["mean"],
                "between_reference_mean_lower_across_subjects": reference_lower_summary["mean"],
                "within_minus_between_mean": delta["mean"],
                "within_minus_between_median": delta["median"],
                "within_minus_between_std": delta["std"],
                "within_minus_between_q25": delta["q25"],
                "within_minus_between_q75": delta["q75"],
                "within_minus_between_mean_upper": delta_upper_summary["mean"],
                "within_minus_between_mean_lower": delta_lower_summary["mean"],
                "reference_type": "between_network_crosshemisphere",
                "valid_fraction": valid_fraction,
            }
        )

    return summary_records


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
    within_edge_map: dict[str, np.ndarray],
    between_edge_map: dict[str, np.ndarray],
    overall_between_edges: np.ndarray,
    network_order: tuple[str, ...],
) -> tuple[dict, list[dict], list[dict]]:
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
    valid_fraction = float(np.mean(valid_mask)) if len(valid_mask) else 0.0

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
        return combo_record, [], []

    is_directed = infer_combo_is_directed(method=method, func=func, symmetry_lookup=symmetry_lookup)
    total_within_edge_count = int(sum(len(within_edge_map[network]) for network in network_order))

    score_records = []
    for sub, row in avg_mats_df.iterrows():
        count = int(row["Count"])
        mat = row["Matrix"]

        network_summaries = []
        if count <= 0 or mat is None:
            for network in network_order:
                score_records.append(
                    build_score_record(
                        sub=sub,
                        method=method,
                        func=func,
                        lag=lag,
                        combo_key=combo_key,
                        analysis_level=ANALYSIS_LEVEL_NETWORK,
                        network=network,
                        n_crosshemi_edges=len(within_edge_map[network]),
                        n_between_network_edges=len(between_edge_map[network]),
                        n_networks_aggregated=1,
                        run_count=count,
                        summary=build_empty_homotopic_summary(is_directed),
                        reference_summary=build_empty_reference_summary(is_directed),
                        is_directed=is_directed,
                    )
                )
            score_records.append(
                build_score_record(
                    sub=sub,
                    method=method,
                    func=func,
                    lag=lag,
                    combo_key=combo_key,
                    analysis_level=ANALYSIS_LEVEL_OVERALL,
                    network=OVERALL_NETWORK_LABEL,
                    n_crosshemi_edges=total_within_edge_count,
                    n_between_network_edges=len(overall_between_edges),
                    n_networks_aggregated=len(network_order),
                    run_count=count,
                    summary=build_empty_homotopic_summary(is_directed),
                    reference_summary=build_empty_reference_summary(is_directed),
                    is_directed=is_directed,
                )
            )
            continue

        ranked = rank_matrix_offdiagonal(
            mat,
            parc_size=parc_size,
            use_abs_rank=use_abs,
        )
        for network in network_order:
            within_edges = within_edge_map[network]
            between_edges = between_edge_map[network]
            summary = summarize_homotopic_fc_ranked_edges(
                ranked,
                parc_size=parc_size,
                edge_indices=within_edges,
                reducer=reducer,
                is_directed=is_directed,
            )
            reference_summary = summarize_between_network_reference_ranked_edges(
                ranked,
                parc_size=parc_size,
                edge_indices=between_edges,
                reducer=reducer,
                is_directed=is_directed,
            )

            network_summaries.append(summary)
            score_records.append(
                build_score_record(
                    sub=sub,
                    method=method,
                    func=func,
                    lag=lag,
                    combo_key=combo_key,
                    analysis_level=ANALYSIS_LEVEL_NETWORK,
                    network=network,
                    n_crosshemi_edges=len(within_edges),
                    n_between_network_edges=len(between_edges),
                    n_networks_aggregated=1,
                    run_count=count,
                    summary=summary,
                    reference_summary=reference_summary,
                    is_directed=is_directed,
                )
            )

        overall_summary = average_network_summaries(network_summaries, is_directed=is_directed)
        overall_reference_summary = summarize_between_network_reference_ranked_edges(
            ranked,
            parc_size=parc_size,
            edge_indices=overall_between_edges,
            reducer=reducer,
            is_directed=is_directed,
        )
        score_records.append(
            build_score_record(
                sub=sub,
                method=method,
                func=func,
                lag=lag,
                combo_key=combo_key,
                analysis_level=ANALYSIS_LEVEL_OVERALL,
                network=OVERALL_NETWORK_LABEL,
                n_crosshemi_edges=total_within_edge_count,
                n_between_network_edges=len(overall_between_edges),
                n_networks_aggregated=len(network_order),
                run_count=count,
                summary=overall_summary,
                reference_summary=overall_reference_summary,
                is_directed=is_directed,
            )
        )

    method_summary_records = summarize_combo_scores(
        score_records,
        method=method,
        func=func,
        lag=lag,
        combo_key=combo_key,
        is_directed=is_directed,
        valid_fraction=valid_fraction,
    )
    return combo_record, score_records, method_summary_records


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
    out_dir: str | None = None,
) -> Path:
    import arfcexp.hcp

    project_root = Path(os.environ["PROJECT_ROOT"])
    data_path_p = Path(data_path)
    if not data_path_p.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path_p}")

    method_func_path = project_root / "resources/sparse_prediction_method_func_list.txt"
    degenerate_lookup_path = project_root / "resources/matrix_degenerate_lookup.json"

    out_dir_root = (
        project_root / "results/hcp_1200_homotopic_fc"
        if out_dir is None
        else Path(out_dir)
    )

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
        method=method,
        func=func,
        lag=lag,
    )
    complete_marker = out_dir_p / "subject_homotopic_scores.parquet"
    if complete_marker.exists():
        logging.info("Output already exists; exiting: %s", out_dir_p)
        return out_dir_p
    out_dir_p.mkdir(exist_ok=True, parents=True)

    network_order = get_schaefer_network_order(parc_size=parc_size)
    within_edge_map = get_schaefer_within_network_edges(parc_size=parc_size)
    between_edge_map = get_schaefer_between_network_edges(parc_size=parc_size)
    overall_between_edges = get_schaefer_overall_between_network_edges(parc_size=parc_size)

    params = {
        "data_path": str(data_path_p),
        "parc_size": parc_size,
        "analysis_tag": ANALYSIS_TAG,
        "analysis_level": "per-network + equal-weight overall network mean",
        "network_order": list(network_order),
        "within_network_edge_counts": {
            network: int(len(within_edge_map[network])) for network in network_order
        },
        "between_network_edge_counts": {
            network: int(len(between_edge_map[network])) for network in network_order
        },
        "overall_between_network_edge_count": int(len(overall_between_edges)),
        "reducer": reducer,
        "use_abs": use_abs,
        "include_skarf_lag1": include_skarf_lag1,
        "method": method,
        "func": func,
        "lag": lag,
        "min_valid_sub_fraction": min_valid_sub_fraction,
        "n_subjects": n_subjects,
        "max_combos": max_combos,
        "observed_strategy": "within_network_lh_x_rh_crosshemisphere_blocks; overall=equal_weight_network_mean",
        "reference_strategy": "between_network_crosshemisphere_complement; per_network=network_local; overall=all_crosshemi_minus_all_within_blocks",
        "reference_type": "deterministic_empirical_between_network_reference",
    }
    with (out_dir_p / "params.yaml").open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    logging.info(
        "Running network-block homotopic FC benchmark with %d combinations (%d excluded).",
        len(combos),
        len(excluded),
    )
    logging.info(
        "Using Schaefer-7 within-network LH x RH blocks (overall = equal-weight mean across %d networks, rank abs=%s).",
        len(network_order),
        use_abs,
    )
    logging.info("Using deterministic between-network cross-hemisphere empirical references.")

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
        combo_record, combo_score_records, combo_method_summaries = evaluate_combo(
            combo,
            data_path=data_path_p,
            sub_list=sub_list,
            reader=reader,
            symmetry_lookup=symmetry_lookup,
            parc_size=parc_size,
            reducer=reducer,
            use_abs=use_abs,
            min_valid_sub_fraction=min_valid_sub_fraction,
            within_edge_map=within_edge_map,
            between_edge_map=between_edge_map,
            overall_between_edges=overall_between_edges,
            network_order=network_order,
        )
        combo_records.append(combo_record)
        score_records.extend(combo_score_records)
        method_summary_records.extend(combo_method_summaries)

    combo_df = pd.DataFrame(combo_records)
    scores_df = pd.DataFrame(score_records)
    method_summary_df = pd.DataFrame(method_summary_records)
    excluded_df = pd.DataFrame(excluded)

    if len(combo_df) > 0:
        combo_df.sort_values(["method", "func", "lag"], inplace=True)
    if len(scores_df) > 0:
        scores_df.sort_values(
            ["method", "func", "lag", "analysis_level", "network", "sub"],
            inplace=True,
        )
    if len(method_summary_df) > 0:
        method_summary_df.sort_values(
            ["method", "func", "lag", "analysis_level", "network"],
            inplace=True,
        )

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
        out_dir=out_dir,
    )


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
    if "analysis_level" in valid_df.columns:
        valid_df = valid_df.loc[valid_df["analysis_level"] == ANALYSIS_LEVEL_OVERALL].copy()

    agg_cols = {"homotopic_score": "family_mean_score"}
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
        if "analysis_level" in method_summary_df.columns:
            method_summary_df = method_summary_df.loc[
                method_summary_df["analysis_level"] == ANALYSIS_LEVEL_OVERALL
            ].copy()

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
