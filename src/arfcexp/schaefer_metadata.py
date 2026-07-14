"""Shared Schaefer parcellation metadata helpers."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np


Hemisphere = Literal["L", "R"]
SUPPORTED_SCHAEFER_PARC_SIZES = (200,)


@dataclass(frozen=True)
class ParcelInfo:
    """Surface vertex membership and label metadata for one Schaefer parcel."""

    label: int
    index: int
    name: str
    hemisphere: Hemisphere
    structure: str
    vertices: np.ndarray


@dataclass(frozen=True)
class SchaeferParcelLabel:
    """Parsed Schaefer-7-network label fields for one parcel."""

    index: int
    hemisphere: Hemisphere
    network: str
    ordinal: int


def _validate_parc_size(parc_size: int) -> None:
    if parc_size not in SUPPORTED_SCHAEFER_PARC_SIZES:
        raise ValueError(
            f"Unsupported Schaefer parcellation size {parc_size}. "
            f"Supported sizes: {SUPPORTED_SCHAEFER_PARC_SIZES}."
        )


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_schaefer_dlabel(
    project_root: Path,
    *,
    parc_size: int = 200,
    networks: int = 7,
) -> Path:
    """Return the expected Schaefer CIFTI label path, with an actionable error."""
    dlabel_path = (
        project_root
        / "resources"
        / "schaefer_parcellations"
        / f"Schaefer2018_{parc_size}Parcels_{networks}Networks_order.dlabel.nii"
    )
    if not dlabel_path.exists():
        raise FileNotFoundError(
            f"Missing Schaefer-{parc_size} dlabel file: {dlabel_path}. "
            "Run `just download_schaefer` before this benchmark."
        )
    return dlabel_path


def parse_schaefer_7network_parcel_name(name: str) -> tuple[Hemisphere, str, int]:
    """Parse a Schaefer-7-network parcel label.

    Expected formats include ``7Networks_LH_Vis_1`` and
    ``7Networks_LH_DorsAttn_Post_1``. The returned network label is the
    top-level Schaefer-7 family, for example ``DorsAttn``.
    """
    parts = name.split("_")
    if len(parts) < 4 or parts[0] != "7Networks" or parts[1] not in {"LH", "RH"}:
        raise ValueError(f"Unsupported Schaefer parcel label format: {name!r}.")

    try:
        ordinal = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid Schaefer parcel ordinal in label: {name!r}.") from exc

    network_parts = parts[2:-1]
    if len(network_parts) == 0:
        raise ValueError(f"Missing Schaefer network label in parcel name: {name!r}.")

    hemisphere: Hemisphere = "L" if parts[1] == "LH" else "R"
    return hemisphere, network_parts[0], ordinal


def load_schaefer_cifti_parcels(dlabel_path: Path) -> tuple[ParcelInfo, ...]:
    """Load Schaefer parcel labels and fsLR vertex memberships from a CIFTI dlabel."""
    import nibabel as nib
    from nibabel.cifti2.cifti2_axes import BrainModelAxis, LabelAxis

    img = nib.load(str(dlabel_path))
    axes = [img.header.get_axis(i) for i in range(img.ndim)]
    label_axis = next((axis for axis in axes if isinstance(axis, LabelAxis)), None)
    brain_axis = next((axis for axis in axes if isinstance(axis, BrainModelAxis)), None)
    if label_axis is None or brain_axis is None:
        raise ValueError(f"Expected LabelAxis and BrainModelAxis in {dlabel_path}.")

    data = np.asarray(img.get_fdata()).squeeze()
    if data.ndim != 1:
        data = data.reshape(-1)
    if data.shape[0] != len(brain_axis):
        raise ValueError(
            f"Label data length {data.shape[0]} does not match brain axis length {len(brain_axis)}."
        )

    label_table = label_axis.label[0]
    label_names = {
        int(label): str(value[0])
        for label, value in label_table.items()
        if int(label) != 0
    }

    parcel_vertices: dict[int, dict[str, object]] = {}
    for structure, slc, sub_axis in brain_axis.iter_structures():
        if structure not in {"CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"}:
            continue
        hemisphere: Hemisphere = "L" if structure.endswith("LEFT") else "R"
        labels = data[slc].astype(np.int32, copy=False)
        vertices = np.asarray(sub_axis.vertex, dtype=np.int32)
        for label in np.unique(labels):
            label_int = int(label)
            if label_int == 0:
                continue
            vertex_values = vertices[labels == label_int]
            entry = parcel_vertices.setdefault(
                label_int,
                {"hemisphere": hemisphere, "structure": structure, "vertices": []},
            )
            if entry["hemisphere"] != hemisphere:
                raise ValueError(f"Parcel label {label_int} appears in both hemispheres.")
            entry["vertices"].append(vertex_values)

    parcels = []
    for index, label in enumerate(sorted(parcel_vertices)):
        entry = parcel_vertices[label]
        vertices = np.concatenate(entry["vertices"]).astype(np.int32, copy=False)
        parcels.append(
            ParcelInfo(
                label=label,
                index=index,
                name=label_names.get(label, f"label_{label}"),
                hemisphere=entry["hemisphere"],
                structure=str(entry["structure"]),
                vertices=vertices,
            )
        )

    return tuple(parcels)


@lru_cache(maxsize=8)
def _get_schaefer_parcels_cached(parc_size: int, project_root: str) -> tuple[ParcelInfo, ...]:
    _validate_parc_size(parc_size)
    dlabel_path = find_schaefer_dlabel(Path(project_root), parc_size=parc_size)
    parcels = load_schaefer_cifti_parcels(dlabel_path)
    if len(parcels) != parc_size:
        raise ValueError(f"Expected {parc_size} Schaefer parcels, found {len(parcels)}.")
    return parcels


def get_schaefer_parcels(
    parc_size: int = 200,
    *,
    project_root: Path | None = None,
) -> tuple[ParcelInfo, ...]:
    """Return Schaefer parcels from the repository dlabel file."""
    root = project_root if project_root is not None else _default_project_root()
    return _get_schaefer_parcels_cached(parc_size, str(root.resolve()))


@lru_cache(maxsize=8)
def get_schaefer_parcel_labels(parc_size: int = 200) -> tuple[SchaeferParcelLabel, ...]:
    """Return parsed Schaefer network and hemisphere labels for each parcel index."""
    parcels = get_schaefer_parcels(parc_size=parc_size)
    labels: list[SchaeferParcelLabel | None] = [None] * parc_size
    for parcel in parcels:
        hemisphere, network, ordinal = parse_schaefer_7network_parcel_name(parcel.name)
        if hemisphere != parcel.hemisphere:
            raise ValueError(
                f"Parcel hemisphere mismatch for {parcel.name!r}: "
                f"label={hemisphere} metadata={parcel.hemisphere}."
            )
        labels[parcel.index] = SchaeferParcelLabel(
            index=parcel.index,
            hemisphere=hemisphere,
            network=network,
            ordinal=ordinal,
        )

    if any(label is None for label in labels):
        raise ValueError("Missing Schaefer labels for one or more parcel indices.")
    return tuple(label for label in labels if label is not None)


@lru_cache(maxsize=8)
def get_schaefer_parcel_network_labels(parc_size: int = 200) -> tuple[str, ...]:
    """Return Schaefer-7 network labels for each parcel index."""
    return tuple(label.network for label in get_schaefer_parcel_labels(parc_size=parc_size))


@lru_cache(maxsize=8)
def get_schaefer_parcel_hemisphere_labels(parc_size: int = 200) -> tuple[Hemisphere, ...]:
    """Return hemisphere labels for each parcel index."""
    return tuple(label.hemisphere for label in get_schaefer_parcel_labels(parc_size=parc_size))


@lru_cache(maxsize=8)
def get_schaefer_network_order(parc_size: int = 200) -> tuple[str, ...]:
    """Return the Schaefer-7 network order induced by the parcel file."""
    seen: set[str] = set()
    ordered: list[str] = []
    for network in get_schaefer_parcel_network_labels(parc_size=parc_size):
        if network not in seen:
            seen.add(network)
            ordered.append(network)
    return tuple(ordered)


@lru_cache(maxsize=8)
def get_schaefer_network_hemisphere_indices(
    parc_size: int = 200,
) -> dict[str, dict[Hemisphere, np.ndarray]]:
    """Return parcel indices grouped by Schaefer network and hemisphere."""
    labels = get_schaefer_parcel_labels(parc_size=parc_size)
    out: dict[str, dict[Hemisphere, list[int]]] = {
        network: {"L": [], "R": []}
        for network in get_schaefer_network_order(parc_size=parc_size)
    }
    for label in labels:
        out[label.network][label.hemisphere].append(label.index)

    return {
        network: {
            "L": np.asarray(groups["L"], dtype=np.int32),
            "R": np.asarray(groups["R"], dtype=np.int32),
        }
        for network, groups in out.items()
    }
