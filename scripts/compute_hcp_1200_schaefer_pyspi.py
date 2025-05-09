import contextlib
import io
import json
import logging
import os
import time
import tracemalloc
from pathlib import Path

# Single-threaded script, parallelize outside
# Need to set before importing other packages to take effect.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "2"

# Suppress cdt GPU/NJOBS message and override to 1 cpu.
with contextlib.redirect_stderr(io.StringIO()):
    import cdt

    cdt.SETTINGS.NJOBS = int(os.environ["OMP_NUM_THREADS"])

import datasets as hfds
import numpy as np
import pyarrow as pa
import typer
from pyarrow import feather
from sklearn.preprocessing import scale

from skarf import set_cache_dir
from skarf.covariance import SPICovariance, create_spi

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

logging.getLogger("skarf").setLevel(logging.WARNING)

hfds.utils.disable_progress_bar()


def main(
    spi: str,
    sub: str,
    parc_size: int = 200,
    pool: int = 3,
    out_dir: str | None = None,
):
    project_root = Path(os.environ["PROJECT_ROOT"])

    # Set cache dir to local spi lists, for reproducibility
    spi_list_dir = project_root / "resources/spi_lists"
    set_cache_dir(spi_list_dir)

    # Look up SPI and subject index IDs.
    spi_list = (spi_list_dir / "spi_list_all_284.txt").read_text().strip().split()
    spi_id_map = {spi: ii for ii, spi in enumerate(spi_list)}
    spi_id = spi_id_map[spi]

    sub_list_path = (
        project_root / "resources/subject_lists/hcp_complete_data_867_subject_list.txt"
    )
    sub_list = sub_list_path.read_text().strip().split()
    sub_id_map = {sub: ii for ii, sub in enumerate(sub_list)}
    sub_id = sub_id_map[sub]

    logging.info(
        "Computing PySPI on HCP 1200 rfMRI Schaefer:\n"
        f"\t{spi_id=:03d} {sub_id=:03d} {spi=} {sub=} {parc_size=} {pool=}"
    )

    if out_dir is None:
        out_dir = project_root / "data/hcp_1200_rfmri_schaefer_pyspi"
    else:
        out_dir = Path(out_dir)

    out_path = (
        out_dir
        / f"parc-{parc_size}__pool-{pool}"
        / f"{spi_id:03d}__spi-{spi}"
        / f"{sub_id:03d}__sub-{sub}.arrow"
    )
    if out_path.exists():
        logging.info("Output already exists; exiting.")
        return

    # Create SPI.
    # Note we create the function up front to avoid timing the import during execution
    spi_cov = SPICovariance(create_spi(spi))

    # Load time series data.
    data_dir = project_root / "data/hcp_1200_rfmri_schaefer_timeseries"
    ts_dataset = hfds.load_from_disk(data_dir)
    ts_dataset = ts_dataset.select_columns(
        ["sub", "ses", "run", f"timeseries_n{parc_size}"]
    )
    ts_dataset = ts_dataset.filter(lambda sub_: sub_ == sub, input_columns="sub")
    ts_dataset.set_format("numpy")
    assert len(ts_dataset) == 4, "Expected 4 runs for each subject."

    schema = pa.schema(
        {
            "spi_id": pa.int32(),
            "sub_id": pa.int32(),
            "spi": pa.string(),
            "sub": pa.string(),
            "ses": pa.uint8(),
            "run": pa.uint8(),
            "success": pa.bool_(),
            "run_time": pa.float32(),
            "peak_mem_mb": pa.float32(),
            "err": pa.string(),
            "mat": pa.list_(pa.float32()),
        }
    )
    records = []

    tracemalloc.start()

    for ii, sample in enumerate(ts_dataset):
        X = np.asarray(sample[f"timeseries_n{parc_size}"], dtype=np.float64)
        # Pool consecutive time points to reduce effective TR
        X = pool_timeseries(X, pool=pool)
        # Standard scale
        X = scale(X)

        # Monitor memory and run time
        tracemalloc.clear_traces()
        tracemalloc.reset_peak()
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
        _, peak_mem = tracemalloc.get_traced_memory()
        peak_mem_mb = peak_mem / 1024**2

        if success:
            mat = spi_cov.covariance_.astype(np.float32).flatten()
        else:
            mat = None

        result = {
            "spi_id": spi_id,
            "sub_id": sub_id,
            "spi": spi,
            "sub": sub,
            "ses": sample["ses"],
            "run": sample["run"],
            "success": success,
            "run_time": run_time,
            "peak_mem_mb": peak_mem_mb,
            "err": err,
            "mat": mat,
        }
        records.append(result)

    tracemalloc.stop()

    tab = pa.Table.from_pylist(records, schema=schema)

    # Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(tab, out_path, compression="uncompressed")

    # Print summary
    success = float(tab["success"].to_numpy().mean())
    run_time = float(tab["run_time"].to_numpy().mean())
    peak_mem_mb = float(tab["peak_mem_mb"].to_numpy().mean())

    summary = {
        "spi_id": spi_id,
        "sub_id": sub_id,
        "spi": spi,
        "sub": sub,
        "success": success,
        "run_time": run_time,
        "peak_mem_mb": peak_mem_mb,
    }

    logging.info(
        f"Done ({spi_id:03d}, {sub_id:03d}): {spi} {sub}\n{json.dumps(summary)}"
    )


def pool_timeseries(X: np.ndarray, pool: int) -> np.ndarray:
    N, D = X.shape
    length = pool * (N // pool)
    X = X[:length]
    X = X.reshape(length // pool, pool, D).mean(axis=1)
    return X


if __name__ == "__main__":
    typer.run(main)
