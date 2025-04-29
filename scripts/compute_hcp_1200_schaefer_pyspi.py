import contextlib
import io
import logging
import json
import os
import time
from functools import partial
from pathlib import Path

import cdt
import datasets as hfds
import fire
import numpy as np
from sklearn.preprocessing import scale

from skarf import set_cache_dir
from skarf.covariance import SPICovariance, create_spi

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

cdt.SETTINGS.NJOBS = 2


def main(spi: str, parc_size: int = 200, pool: int = 3):
    project_root = Path(os.environ["PROJECT_ROOT"])

    # Set cache dir to local spi lists, for reproducibility
    spi_list_dir = project_root / "resources/spi_lists"
    set_cache_dir(spi_list_dir)

    spi_list = (spi_list_dir / "spi_list_all_284.txt").read_text().strip().split()
    spi_id_map = {spi: ii for ii, spi in enumerate(spi_list)}
    spi_id = spi_id_map[spi]

    logging.info(
        "Computing PySPI on HCP 1200 rfMRI Schaefer:\n"
        f"\t{spi_id=:03d} {spi=} {parc_size=} {pool=}"
    )

    out_path = (
        project_root
        / "data/hcp_1200_rfmri_schaefer_pyspi"
        / f"parc-{parc_size}__pool-{pool}"
        / f"{spi_id:03d}__spi-{spi}"
    )
    logging.info("Saving to: %s", out_path)

    if out_path.exists():
        logging.info("Output dataset already exists; exiting.")
        return

    num_proc = max(len(os.sched_getaffinity(0)) // 2, 1)
    logging.info("Running with %d processes.", num_proc)

    logging.info("Loading SPI: %s", spi)
    spi_fun = create_spi(spi)
    spi_cov = SPICovariance(spi_fun)
    logging.info("%s", spi_cov)

    data_dir = project_root / "data/hcp_1200_rfmri_schaefer_timeseries"
    logging.info("Loading time series dataset:\n\t%s", data_dir)
    ts_dataset = hfds.load_from_disk(data_dir)
    ts_dataset.select_columns(["sub", "ses", "run", f"timeseries_n{parc_size}"])
    ts_dataset.set_format("numpy")

    logging.info("Computing SPI matrices")
    func = partial(
        compute_spi,
        spi_cov=spi_cov,
        spi=spi,
        spi_id=spi_id,
        parc_size=parc_size,
        pool=pool,
    )
    features = hfds.Features(
        {
            "spi_id": hfds.Value("int32"),
            "spi": hfds.Value("string"),
            "sub": hfds.Value("string"),
            "ses": hfds.Value("uint8"),
            "run": hfds.Value("uint8"),
            "success": hfds.Value("bool"),
            "run_time": hfds.Value("float32"),
            "err": hfds.Value("string"),
            "mat": hfds.Sequence(hfds.Value("float32")),
        }
    )
    mat_dataset = ts_dataset.map(
        func,
        remove_columns=ts_dataset.column_names,
        features=features,
        num_proc=num_proc,
    )

    logging.info("Saving to disk")
    mat_dataset.save_to_disk(out_path, max_shard_size="256MB", num_proc=num_proc)

    run_time_mean = float(mat_dataset["run_time"].mean())
    run_time_std = float(mat_dataset["run_time"].std())
    success_count = int(mat_dataset["success"].sum())
    total_count = len(mat_dataset)
    success_rate = success_count / total_count
    summary_result = {
        "spi": spi,
        "spi_id": spi_id,
        "success_count": success_count,
        "total_count": total_count,
        "success_rate": success_rate,
        "run_time_mean": run_time_mean,
        "run_time_std": run_time_std,
    }
    logging.info(
        f"Done {spi_id:03d} {spi}:  "
        f"success: {success_count}/{total_count} ({100 * success_rate:.1f}%), "
        f"rt: {run_time_mean:.2f}s +/- {run_time_std:.2f}s\n"
        f"{json.dumps(summary_result)}"
    )


def compute_spi(
    sample: dict[str, any],
    *,
    spi_cov: SPICovariance,
    spi: str,
    spi_id: int,
    parc_size: int,
    pool: int,
):
    sub, ses, run = [sample[k] for k in ["sub", "ses", "run"]]

    X = np.asarray(sample[f"timeseries_n{parc_size}"], dtype=np.float64)
    # Pool consecutive time points to reduce effective TR
    X = X.reshape(len(X) // pool, pool, X.shape[1]).mean(axis=1)
    # Standard scale
    X = scale(X)

    # Monitor run time
    tic = time.perf_counter()
    try:
        # Suppress data normalisation print message
        with contextlib.redirect_stdout(io.StringIO()):
            spi_cov.fit(X)
        err = None
    except Exception as exc:
        err = repr(exc)

    success = err is None
    run_time = time.perf_counter() - tic
    if success:
        mat = spi_cov.covariance_.astype(np.float32).flatten()
    else:
        mat = None

    result = {
        "spi_id": spi_id,
        "spi": spi,
        "sub": sub,
        "ses": ses,
        "run": run,
        "success": success,
        "run_time": run_time,
        "err": err,
        "mat": mat,
    }
    return result


if __name__ == "__main__":
    fire.Fire(main)
