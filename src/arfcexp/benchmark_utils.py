"""Shared helpers for benchmark method/function selection."""

import json
from pathlib import Path


def infer_combo_is_directed(
    *,
    method: str,
    func: str,
    symmetry_lookup: dict[str, bool],
) -> bool:
    """Infer matrix directionality from ``matrix_symmetry_lookup.json``."""
    key = f"{method}__{func}"
    if key not in symmetry_lookup:
        raise KeyError(
            f"Missing symmetry lookup key: {key}. "
            "Please update resources/matrix_symmetry_lookup.json before running."
        )
    return not bool(symmetry_lookup[key])


def make_combo_key(method: str, func: str, lag: int) -> str:
    """Return the canonical benchmark key for a method/function(/lag)."""
    if method == "skarf":
        return f"{method}__{func}__lag-{lag}"
    return f"{method}__{func}"


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
            base_pairs.append((parts[0], parts[1]))
    return base_pairs


def build_combinations(
    base_pairs: list[tuple[str, str]],
    degenerate_lookup: dict[str, bool],
    *,
    include_skarf_lag1: bool,
) -> tuple[list[dict], list[dict]]:
    """Expand base method/function pairs and exclude degenerate PySPI matrices."""
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
        if combo["method"] == method
        and combo["func"] == func
        and combo["lag"] == selected_lag
    ]


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