#!/usr/bin/env -S uv run --script

import logging
import os
from pathlib import Path

import datasets as hfds
import nibabel as nib
import numpy as np
from nisc.cifti import get_cifti_surf_data
from sklearn.preprocessing import scale

import arfcexp.timeseries as ts
from arfcexp.hcp import parse_hcp_metadata

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
HCP_ROOT = Path(os.environ["HCP_1200_DIR"])

PARC_SIZES = [200, 400, 800, 1000]
SUB_LIST_PATH = (
    PROJECT_ROOT / "resources/subject_lists/hcp_complete_data_867_subject_list.txt"
)

TASK_DIR_SES_RUN_MAP = {
    ("REST1", "LR"): (1, 1),
    ("REST1", "RL"): (1, 2),
    ("REST2", "LR"): (2, 1),
    ("REST2", "RL"): (2, 2),
}

HCP_TR = 0.72
DEBUG = False


def main():
    logging.info(
        "Extracting HCP rfMRI Schaefer time series.\n"
        f"\tParcellation sizes: {PARC_SIZES}\n"
        f"\tSubject list: {SUB_LIST_PATH}\n"
    )

    out_path = PROJECT_ROOT / "data/hcp_1200_rfmri_schaefer_timeseries"
    logging.info("Saving to: %s", out_path)
    out_path.mkdir(exist_ok=DEBUG, parents=True)

    # Get number of cpus available to current process
    num_proc = max(len(os.sched_getaffinity(0)) // 3, 1)
    if DEBUG:
        num_proc = 2
    logging.info("Running with %d processes.", num_proc)

    # Load parcellation
    parc_one_hots = {}
    for size in PARC_SIZES:
        parc_path = (
            PROJECT_ROOT
            / "resources"
            / "schaefer_parcellations"
            / f"Schaefer2018_{size}Parcels_7Networks_order.dscalar.nii"
        )
        logging.info("Loading parcellation:\n\t%s", parc_path)
        parc = load_cifti(parc_path).flatten()
        parc_one_hot = ts.parc_one_hot_encode(parc, sparse=True)
        parc_one_hots[size] = parc_one_hot

    subjects = np.genfromtxt(SUB_LIST_PATH, dtype=str).tolist()
    if DEBUG:
        subjects = subjects[:4]
    logging.info(f"Subjects:\n\t{subjects[:10]}")

    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "ses": hfds.Value("uint8"),
            "run": hfds.Value("uint8"),
            **{
                f"timeseries_n{size}": hfds.Array2D((None, size), "float32")
                for size in PARC_SIZES
            },
        }
    )

    dataset = hfds.Dataset.from_generator(
        dataset_generator,
        features=features,
        gen_kwargs={
            "hcp_root": HCP_ROOT,
            "subjects": subjects,
            "parc_one_hots": parc_one_hots,
        },
        num_proc=num_proc,
    )

    dataset.save_to_disk(out_path, max_shard_size="256MB", num_proc=num_proc)


def dataset_generator(
    hcp_root: Path,
    subjects: list[str],
    parc_one_hots: dict[int, np.ndarray],
):
    for sub in subjects:
        paths = sorted(
            (hcp_root / sub).rglob(
                "rfMRI_REST[12]_[LR][LR]_Atlas_MSMAll_hp2000_clean.dtseries.nii"
            )
        )
        if DEBUG:
            paths = paths[:2]

        for path in paths:
            meta = parse_hcp_metadata(path)
            ses, run = TASK_DIR_SES_RUN_MAP[(meta["task"], meta["dir"])]

            full_series = load_cifti(path)
            if DEBUG:
                full_series = scale(full_series.astype(np.float32))
            else:
                full_series = ts.preprocess_timeseries(full_series, tr=HCP_TR)

            parc_series = {
                f"timeseries_n{size}": ts.extract_timeseries(full_series, parc_one_hot)
                for size, parc_one_hot in parc_one_hots.items()
            }

            record = {"sub": sub, "ses": ses, "run": run, **parc_series}
            yield record


def load_cifti(path: Path) -> np.ndarray:
    """Load cifti time series data."""
    data = nib.load(path)
    data = get_cifti_surf_data(data)
    data = np.ascontiguousarray(data.T)
    return data


if __name__ == "__main__":
    main()
