"""Weight-distance benchmark utilities.

The main statistic follows Liu et al. (Nature Methods, 2025) and the referenced
FC-PySPI implementation, where anatomical centroid distance is correlated with
functional connectivity edge weights using raw signed Spearman correlation:
https://www.nature.com/articles/s41592-025-02704-4 and
https://github.com/netneurolab/liu_fc-pyspi/blob/6617f0f6ba7e00c94a7ce59032b92e1f268eb27f/code/02_network_properties.py#L54

Geometry helpers use Schaefer et al. 2018 parcels in fsLR CIFTI space, nibabel
for CIFTI/GIFTI I/O, neuromaps fsLR surfaces, and Alexander-Bloch spin nulls via
neuromaps: https://netneurolab.github.io/neuromaps/generated/neuromaps.nulls.alexander_bloch.html
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from arfcexp.schaefer_metadata import (
    Hemisphere,
    ParcelInfo,
    find_schaefer_dlabel,
    load_schaefer_cifti_parcels,
)

Triangle = Literal["upper", "lower", "offdiag"]


@dataclass(frozen=True)
class WeightDistanceGeometry:
    """Schaefer parcel geometry needed for the weight-distance statistic."""

    parcels: tuple[ParcelInfo, ...]
    centroids: np.ndarray
    distance_matrix: np.ndarray
    dlabel_path: Path


@dataclass(frozen=True)
class WeightDistanceCache:
    """Cached geometry and spin-null predictors."""

    geometry: WeightDistanceGeometry
    cache_dir: Path
    n_perm: int
    seed: int
    spin_indices: np.ndarray | None
    null_ranks_upper: np.ndarray | None
    null_ranks_lower: np.ndarray | None
    null_ranks_offdiag: np.ndarray | None


def as_square_matrix(mat: np.ndarray, parc_size: int | None = None) -> np.ndarray:
    """Return a 2D square matrix from a flattened or already-square matrix."""
    arr = np.asarray(mat)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Expected a square matrix, got shape {arr.shape}.")
        if parc_size is not None and arr.shape != (parc_size, parc_size):
            raise ValueError(
                f"Expected matrix shape ({parc_size}, {parc_size}), got {arr.shape}."
            )
        return arr

    if arr.ndim != 1:
        raise ValueError(f"Expected 1D flattened or 2D matrix input, got ndim={arr.ndim}.")

    n_elements = len(arr)
    n = int(np.sqrt(n_elements))
    if n * n != n_elements:
        raise ValueError(f"Cannot reshape vector of length {n_elements} to a square matrix.")
    if parc_size is not None and n != parc_size:
        raise ValueError(f"Matrix size {n} does not match parc_size={parc_size}.")
    return arr.reshape(n, n)


def compute_centroid_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances between parcel centroids."""
    coords = np.asarray(centroids, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected centroids with shape (n_parcels, 3), got {coords.shape}.")
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def triangle_indices(n: int, triangle: Triangle) -> tuple[np.ndarray, np.ndarray]:
    """Return matrix indices for an upper, lower, or full off-diagonal triangle."""
    if triangle == "upper":
        return np.triu_indices(n, k=1)
    if triangle == "lower":
        return np.tril_indices(n, k=-1)
    if triangle == "offdiag":
        return np.where(~np.eye(n, dtype=bool))
    raise ValueError(f"Unsupported triangle={triangle!r}.")


def extract_triangle_values(mat: np.ndarray, triangle: Triangle) -> np.ndarray:
    """Extract one edge vector from a square matrix."""
    mat2d = as_square_matrix(mat)
    idx = triangle_indices(mat2d.shape[0], triangle)
    return mat2d[idx].astype(np.float64, copy=False)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Rank-transform finite values and L2-normalize centered ranks.

    The dot product of two outputs from this function equals Spearman's rho.
    NaNs are preserved so callers can align finite pairs explicitly.
    """
    vals = np.asarray(values, dtype=np.float64)
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(vals)
    if finite.sum() < 2:
        return out

    ranks = rankdata(vals[finite], method="average").astype(np.float64)
    ranks -= np.mean(ranks)
    norm = np.linalg.norm(ranks)
    if not np.isfinite(norm) or norm == 0.0:
        return out

    out[finite] = ranks / norm
    return out


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute raw signed Spearman correlation with pairwise finite handling."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"Shape mismatch: {x_arr.shape} vs {y_arr.shape}.")

    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if finite.sum() < 2:
        return np.nan

    x_rank = rank_normalize(x_arr[finite])
    y_rank = rank_normalize(y_arr[finite])
    if np.any(np.isnan(x_rank)) or np.any(np.isnan(y_rank)):
        return np.nan
    return float(np.dot(x_rank, y_rank))


def summarize_null_scores(
    *,
    observed: float,
    weight_values: np.ndarray,
    null_distance_ranks: np.ndarray | None,
    prefix: str,
) -> dict[str, float]:
    """Summarize spin-null Spearman scores using cached distance rank predictors."""
    keys = {
        f"{prefix}_mean": np.nan,
        f"{prefix}_std": np.nan,
        f"{prefix}_q025": np.nan,
        f"{prefix}_q25": np.nan,
        f"{prefix}_q50": np.nan,
        f"{prefix}_q75": np.nan,
        f"{prefix}_q975": np.nan,
        f"{prefix}_p_two_sided": np.nan,
        f"{prefix}_z": np.nan,
    }
    if null_distance_ranks is None or null_distance_ranks.size == 0:
        return keys

    weight_rank = rank_normalize(weight_values)
    finite = np.isfinite(weight_rank)
    if finite.sum() < 2:
        return keys

    if np.all(finite):
        null_scores = null_distance_ranks @ weight_rank
    else:
        null_scores = null_distance_ranks[:, finite] @ weight_rank[finite]
    null_scores = null_scores[np.isfinite(null_scores)]
    if null_scores.size == 0:
        return keys

    keys[f"{prefix}_mean"] = float(np.mean(null_scores))
    keys[f"{prefix}_std"] = float(np.std(null_scores, ddof=1)) if null_scores.size > 1 else np.nan
    keys[f"{prefix}_q025"] = float(np.percentile(null_scores, 2.5))
    keys[f"{prefix}_q25"] = float(np.percentile(null_scores, 25))
    keys[f"{prefix}_q50"] = float(np.percentile(null_scores, 50))
    keys[f"{prefix}_q75"] = float(np.percentile(null_scores, 75))
    keys[f"{prefix}_q975"] = float(np.percentile(null_scores, 97.5))

    if np.isfinite(observed):
        keys[f"{prefix}_p_two_sided"] = float(
            (1 + np.sum(np.abs(null_scores) >= abs(observed))) / (null_scores.size + 1)
        )
        std = keys[f"{prefix}_std"]
        if np.isfinite(std) and std > 0:
            keys[f"{prefix}_z"] = float((observed - keys[f"{prefix}_mean"]) / std)

    return keys


def score_weight_distance_matrix(
    mat: np.ndarray,
    distance_matrix: np.ndarray,
    *,
    is_directed: bool,
    null_ranks_upper: np.ndarray | None = None,
    null_ranks_lower: np.ndarray | None = None,
    null_ranks_offdiag: np.ndarray | None = None,
) -> dict[str, float | bool]:
    """Score one FC matrix against centroid distances.

    Symmetric matrices use the upper triangle as the primary statistic. Directed
    matrices return upper and lower triangle scores plus a combined off-diagonal
    primary score.
    """
    mat2d = as_square_matrix(mat)
    dist2d = as_square_matrix(distance_matrix, parc_size=mat2d.shape[0])

    upper_weights = extract_triangle_values(mat2d, "upper")
    upper_distances = extract_triangle_values(dist2d, "upper")
    upper_score = spearman_corr(upper_distances, upper_weights)

    out: dict[str, float | bool] = {
        "weight_distance_score": upper_score,
        "weight_distance_score_upper": upper_score,
        "weight_distance_score_lower": np.nan,
        "weight_distance_score_offdiag": np.nan,
        "is_directed": bool(is_directed),
    }
    out.update(
        summarize_null_scores(
            observed=upper_score,
            weight_values=upper_weights,
            null_distance_ranks=null_ranks_upper,
            prefix="spin_null",
        )
    )
    out.update(
        summarize_null_scores(
            observed=upper_score,
            weight_values=upper_weights,
            null_distance_ranks=null_ranks_upper,
            prefix="spin_null_upper",
        )
    )

    if not is_directed:
        out.update({f"spin_null_lower_{suffix}": np.nan for suffix in _NULL_SUFFIXES})
        out.update({f"spin_null_offdiag_{suffix}": np.nan for suffix in _NULL_SUFFIXES})
        return out

    lower_weights = extract_triangle_values(mat2d, "lower")
    lower_distances = extract_triangle_values(dist2d, "lower")
    offdiag_weights = extract_triangle_values(mat2d, "offdiag")
    offdiag_distances = extract_triangle_values(dist2d, "offdiag")

    lower_score = spearman_corr(lower_distances, lower_weights)
    offdiag_score = spearman_corr(offdiag_distances, offdiag_weights)

    out["weight_distance_score"] = offdiag_score
    out["weight_distance_score_lower"] = lower_score
    out["weight_distance_score_offdiag"] = offdiag_score
    out.update(
        summarize_null_scores(
            observed=offdiag_score,
            weight_values=offdiag_weights,
            null_distance_ranks=null_ranks_offdiag,
            prefix="spin_null",
        )
    )
    out.update(
        summarize_null_scores(
            observed=lower_score,
            weight_values=lower_weights,
            null_distance_ranks=null_ranks_lower,
            prefix="spin_null_lower",
        )
    )
    out.update(
        summarize_null_scores(
            observed=offdiag_score,
            weight_values=offdiag_weights,
            null_distance_ranks=null_ranks_offdiag,
            prefix="spin_null_offdiag",
        )
    )
    return out


_NULL_SUFFIXES = (
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


def get_fslr_surface_paths(
    *,
    density: str = "32k",
    surface: str = "midthickness",
) -> tuple[Path, Path]:
    """Fetch fsLR surface files from neuromaps and return LH/RH paths."""
    from neuromaps.datasets import fetch_fslr

    fetched = fetch_fslr(density=density)
    if not hasattr(fetched, surface) and surface not in fetched:
        available = sorted(list(fetched.keys())) if hasattr(fetched, "keys") else sorted(vars(fetched))
        raise KeyError(f"fsLR surface {surface!r} not available. Options: {available}")

    paths = getattr(fetched, surface) if hasattr(fetched, surface) else fetched[surface]
    if len(paths) != 2:
        raise ValueError(f"Expected two hemisphere files for {surface}, got {paths!r}.")
    return Path(paths[0]), Path(paths[1])


def load_surface_coordinates(surface_paths: tuple[Path, Path]) -> dict[Hemisphere, np.ndarray]:
    """Load LH/RH GIFTI surface coordinate arrays."""
    import nibabel as nib

    coords = {}
    for hemi, path in zip(("L", "R"), surface_paths):
        img = nib.load(str(path))
        coords[hemi] = np.asarray(img.darrays[0].data, dtype=np.float64)
    return coords


def compute_parcel_centroids(
    parcels: tuple[ParcelInfo, ...],
    surface_coordinates: dict[Hemisphere, np.ndarray],
) -> np.ndarray:
    """Compute arithmetic midthickness centroids for each parcel."""
    centroids = np.full((len(parcels), 3), np.nan, dtype=np.float64)
    for parcel in parcels:
        hemi_coords = surface_coordinates[parcel.hemisphere]
        if np.max(parcel.vertices) >= hemi_coords.shape[0]:
            raise ValueError(
                f"Parcel {parcel.name} references vertex {np.max(parcel.vertices)} but "
                f"surface has only {hemi_coords.shape[0]} vertices."
            )
        centroids[parcel.index] = np.mean(hemi_coords[parcel.vertices], axis=0)
    return centroids


def build_weight_distance_geometry(
    dlabel_path: Path,
    *,
    density: str = "32k",
    distance_surface: str = "midthickness",
) -> WeightDistanceGeometry:
    """Build Schaefer centroid geometry using neuromaps fsLR surfaces."""
    parcels = load_schaefer_cifti_parcels(dlabel_path)
    if len(parcels) != 200:
        raise ValueError(f"Expected 200 Schaefer parcels, found {len(parcels)}.")

    hemispheres = [parcel.hemisphere for parcel in parcels]
    if hemispheres.count("L") != 100 or hemispheres.count("R") != 100:
        raise ValueError(
            f"Expected 100 LH and 100 RH parcels, got L={hemispheres.count('L')} "
            f"R={hemispheres.count('R')}."
        )

    surface_paths = get_fslr_surface_paths(density=density, surface=distance_surface)
    surface_coordinates = load_surface_coordinates(surface_paths)
    centroids = compute_parcel_centroids(parcels, surface_coordinates)
    distance_matrix = compute_centroid_distance_matrix(centroids)
    return WeightDistanceGeometry(
        parcels=parcels,
        centroids=centroids,
        distance_matrix=distance_matrix,
        dlabel_path=dlabel_path,
    )


def parcel_info_to_frame(parcels: tuple[ParcelInfo, ...], centroids: np.ndarray) -> pd.DataFrame:
    """Convert parcel metadata and centroids to a tabular representation."""
    rows = []
    for parcel in parcels:
        rows.append(
            {
                "parcel_index": parcel.index,
                "label": parcel.label,
                "name": parcel.name,
                "hemisphere": parcel.hemisphere,
                "structure": parcel.structure,
                "n_vertices": int(len(parcel.vertices)),
                "centroid_x": float(centroids[parcel.index, 0]),
                "centroid_y": float(centroids[parcel.index, 1]),
                "centroid_z": float(centroids[parcel.index, 2]),
            }
        )
    return pd.DataFrame(rows)


def write_parcel_label_giftis(
    parcels: tuple[ParcelInfo, ...],
    out_dir: Path,
    *,
    n_vertices: int = 32492,
) -> tuple[Path, Path]:
    """Write LH/RH GIFTI label files for neuromaps parcellated spin nulls."""
    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

    out_dir.mkdir(exist_ok=True, parents=True)
    out_paths = []
    for hemi, suffix in (("L", "L"), ("R", "R")):
        labels = np.zeros(n_vertices, dtype=np.int32)
        label_table = GiftiLabelTable()
        medial = GiftiLabel(0)
        medial.label = "medial_wall"
        medial.red, medial.green, medial.blue, medial.alpha = 0.0, 0.0, 0.0, 0.0
        label_table.labels.append(medial)

        for parcel in parcels:
            if parcel.hemisphere != hemi:
                continue
            labels[parcel.vertices] = parcel.label
            gifti_label = GiftiLabel(parcel.label)
            gifti_label.label = parcel.name
            gifti_label.red, gifti_label.green, gifti_label.blue, gifti_label.alpha = 0.5, 0.5, 0.5, 1.0
            label_table.labels.append(gifti_label)

        img = GiftiImage()
        img.add_gifti_data_array(
            GiftiDataArray(labels, intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
        )
        img.labeltable = label_table
        path = out_dir / f"Schaefer2018_200Parcels_7Networks_order.{suffix}.label.gii"
        nib.save(img, str(path))
        out_paths.append(path)
    return Path(out_paths[0]), Path(out_paths[1])


def generate_alexander_bloch_spin_indices(
    parcels: tuple[ParcelInfo, ...],
    label_gifti_paths: tuple[Path, Path],
    *,
    density: str = "32k",
    n_perm: int = 1000,
    seed: int = 2142,
) -> np.ndarray:
    """Generate parcel-index spin remappings with neuromaps Alexander-Bloch nulls."""
    from neuromaps.nulls import alexander_bloch

    if n_perm <= 0:
        return np.empty((0, len(parcels)), dtype=np.int32)

    data = np.arange(len(parcels), dtype=np.float64)
    null_maps = alexander_bloch(
        data,
        atlas="fsLR",
        density=density,
        parcellation=tuple(str(path) for path in label_gifti_paths),
        n_perm=n_perm,
        seed=seed,
    )
    null_maps = np.asarray(null_maps)
    if null_maps.shape == (len(parcels), n_perm):
        spin_indices = null_maps.T
    elif null_maps.shape == (n_perm, len(parcels)):
        spin_indices = null_maps
    else:
        raise ValueError(f"Unexpected Alexander-Bloch null shape: {null_maps.shape}.")

    if not np.all(np.isfinite(spin_indices)):
        raise ValueError("Alexander-Bloch null maps contain non-finite parcel indices.")
    spin_indices = np.rint(spin_indices).astype(np.int32)
    if np.any(spin_indices < 0) or np.any(spin_indices >= len(parcels)):
        raise ValueError("Alexander-Bloch null maps contain out-of-bounds parcel indices.")
    return spin_indices


def precompute_null_distance_ranks(
    distance_matrix: np.ndarray,
    spin_indices: np.ndarray,
    *,
    triangle: Triangle,
) -> np.ndarray:
    """Precompute rank-normalized permuted distance vectors for one triangle."""
    dist2d = as_square_matrix(distance_matrix)
    n_perm = spin_indices.shape[0]
    edge_count = len(extract_triangle_values(dist2d, triangle))
    out = np.empty((n_perm, edge_count), dtype=np.float32)
    for perm_i, perm in enumerate(spin_indices):
        permuted_dist = dist2d[np.ix_(perm, perm)]
        out[perm_i] = rank_normalize(extract_triangle_values(permuted_dist, triangle)).astype(
            np.float32,
            copy=False,
        )
    return out


def prepare_weight_distance_cache(
    dlabel_path: Path,
    cache_dir: Path,
    *,
    n_perm: int = 1000,
    seed: int = 2142,
    density: str = "32k",
    distance_surface: str = "midthickness",
    force: bool = False,
) -> WeightDistanceCache:
    """Prepare or load centroid geometry and reusable spin-null distance ranks."""
    cache_dir.mkdir(exist_ok=True, parents=True)
    centroids_path = cache_dir / "parcel_centroids.npy"
    distance_path = cache_dir / "parcel_distance_matrix.npy"
    parcels_path = cache_dir / "parcel_centroids.csv"

    if not force and centroids_path.exists() and distance_path.exists() and parcels_path.exists():
        parcels_df = pd.read_csv(parcels_path)
        parcels = tuple(
            ParcelInfo(
                label=int(row.label),
                index=int(row.parcel_index),
                name=str(row.name),
                hemisphere=str(row.hemisphere),
                structure=str(row.structure),
                vertices=np.empty(0, dtype=np.int32),
            )
            for row in parcels_df.itertuples(index=False)
        )
        geometry = WeightDistanceGeometry(
            parcels=parcels,
            centroids=np.load(centroids_path),
            distance_matrix=np.load(distance_path),
            dlabel_path=dlabel_path,
        )
    else:
        geometry = build_weight_distance_geometry(
            dlabel_path,
            density=density,
            distance_surface=distance_surface,
        )
        np.save(centroids_path, geometry.centroids)
        np.save(distance_path, geometry.distance_matrix)
        parcel_info_to_frame(geometry.parcels, geometry.centroids).to_csv(parcels_path, index=False)

    spin_indices = None
    null_upper = None
    null_lower = None
    null_offdiag = None

    if n_perm > 0:
        spin_path = cache_dir / f"spin_indices_seed-{seed}_nperm-{n_perm}.npy"
        upper_path = cache_dir / f"null_distance_ranks_upper_seed-{seed}_nperm-{n_perm}.npy"
        lower_path = cache_dir / f"null_distance_ranks_lower_seed-{seed}_nperm-{n_perm}.npy"
        offdiag_path = cache_dir / f"null_distance_ranks_offdiag_seed-{seed}_nperm-{n_perm}.npy"

        if not force and spin_path.exists():
            spin_indices = np.load(spin_path)
        else:
            parcels_with_vertices = load_schaefer_cifti_parcels(dlabel_path)
            label_paths = write_parcel_label_giftis(parcels_with_vertices, cache_dir / "label_gifti")
            spin_indices = generate_alexander_bloch_spin_indices(
                parcels_with_vertices,
                label_paths,
                density=density,
                n_perm=n_perm,
                seed=seed,
            )
            np.save(spin_path, spin_indices)

        if not force and upper_path.exists() and lower_path.exists() and offdiag_path.exists():
            null_upper = np.load(upper_path)
            null_lower = np.load(lower_path)
            null_offdiag = np.load(offdiag_path)
        else:
            null_upper = precompute_null_distance_ranks(
                geometry.distance_matrix,
                spin_indices,
                triangle="upper",
            )
            null_lower = precompute_null_distance_ranks(
                geometry.distance_matrix,
                spin_indices,
                triangle="lower",
            )
            null_offdiag = precompute_null_distance_ranks(
                geometry.distance_matrix,
                spin_indices,
                triangle="offdiag",
            )
            np.save(upper_path, null_upper)
            np.save(lower_path, null_lower)
            np.save(offdiag_path, null_offdiag)

    return WeightDistanceCache(
        geometry=geometry,
        cache_dir=cache_dir,
        n_perm=n_perm,
        seed=seed,
        spin_indices=spin_indices,
        null_ranks_upper=null_upper,
        null_ranks_lower=null_lower,
        null_ranks_offdiag=null_offdiag,
    )