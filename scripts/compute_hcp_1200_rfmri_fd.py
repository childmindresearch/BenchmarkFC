import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from arfcexp.hcp import parse_hcp_metadata
from arfcexp.motion import compute_fd, censor_motion_spikes

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

# Censoring config following Li et al., NeuroImage 2019.
# But note, using more lenient spike threshold of 0.4. This is still in the typical
# range, and close to the true volumewise FD outlier threshold. Li et al used
# fsl_motion_outliers, which computes Jenkinson FD, which is not the same as Power FD we
# use here.
CENSOR_FD_THRESHOLD = 0.4
CENSOR_SEGMENT_THRESHOLD = 5


def main():
    logging.info("Computing FD for HCP rest runs.")

    root = Path(os.environ["PROJECT_ROOT"])
    hcp_root = Path(os.environ["HCP_1200_DIR"])

    motion_regressor_paths = sorted(hcp_root.rglob("Movement_Regressors.txt"))
    logging.info("Found %d motion regressor paths.", len(motion_regressor_paths))

    # Get number of cpus available to current process
    n_cpus = len(os.sched_getaffinity(0)) or 1
    logging.info("Running with %d cpu cores.", n_cpus)

    with ProcessPoolExecutor(max_workers=n_cpus) as pool:
        records = pool.map(generate_hcp_fd_record, motion_regressor_paths, chunksize=32)
        df = pd.DataFrame.from_records(
            [rec for rec in tqdm(records, total=len(motion_regressor_paths)) if rec]
        )

    out_path = root / "results/hcp_1200_rfmri_fd/hcp_1200_rfmri_fd.parquet"
    logging.info("Saving to %s", out_path)
    out_path.parent.mkdir(exist_ok=True)
    df.to_parquet(out_path, index=False)


def generate_hcp_fd_record(path: Path) -> dict | None:
    meta = parse_hcp_metadata(path)
    meta.pop("clean", None)

    if meta["mod"] == "rfMRI":
        motion_regressors = np.genfromtxt(path)
        n_frames = len(motion_regressors)

        # Compute framewise displacement.
        fd_values = compute_fd(motion_regressors)
        fd_values = fd_values.astype(np.float32)
        mean_fd = np.mean(fd_values[1:])

        # Censor motion spikes following Li et al., 2019.
        sample_mask = censor_motion_spikes(
            fd_values,
            threshold=CENSOR_FD_THRESHOLD,
            segment_threshold=CENSOR_SEGMENT_THRESHOLD,
        )
        n_censor_frames = np.sum(~sample_mask)
        censor_frac = n_censor_frames / n_frames

        record = {
            **meta,
            "n_frames": n_frames,
            "mean_fd": mean_fd,
            "n_censor_frames": n_censor_frames,
            "censor_frac": censor_frac,
            "fd_values": fd_values,
            "sample_mask": sample_mask,
        }
        return record


if __name__ == "__main__":
    main()
