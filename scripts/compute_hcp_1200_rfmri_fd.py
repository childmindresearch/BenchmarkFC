import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from arfcexp.hcp import parse_hcp_metadata
from arfcexp.compute_fd import compute_fd

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


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
        mean_fd = compute_fd(motion_regressors).mean()
        record = {**meta, "fd": mean_fd}
        return record


if __name__ == "__main__":
    main()
