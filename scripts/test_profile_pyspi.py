import json
import logging
import os
import time
import tracemalloc
from pathlib import Path

import cdt
import datasets as hfds
import fire
import numpy as np
import pyarrow as pa
from pyarrow import feather
from pyspi.utils import check_optional_deps
from sklearn.preprocessing import scale

from skarf.covariance import SPICovariance, create_spi

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

N_CPUS = max(len(os.sched_getaffinity(0)), 1)
cdt.SETTINGS.NJOBS = N_CPUS
os.environ["OMP_NUM_THREADS"] = str(N_CPUS)


def main(
    spi: str,
    n_samples: list[int],
    n_features: list[int],
    parc_size: int = 200,
):
    logging.info(
        "Testing and profiling PySPI SPIs:\n"
        f"\t{spi=}\n"
        f"\t{n_samples=}\n"
        f"\t{n_features=}\n"
        f"\t{parc_size=}"
    )

    project_root = Path(os.environ["PROJECT_ROOT"])
    outdir = project_root / "results/test_profile_pyspi"
    outdir.mkdir(exist_ok=True)

    # Nb, this implicitly starts the JVM.
    logging.info("PySPI optional depedencies: %s", check_optional_deps())

    logging.info("Loading SPI: %s", spi)
    spi_fun = create_spi(spi)
    spi_cov = SPICovariance(spi_fun)
    logging.info("%s", spi_cov)

    logging.info("Loading sample time series")
    ts_dataset = hfds.load_from_disk(
        project_root / "data/hcp_1200_rfmri_schaefer_timeseries"
    )
    ts_dataset.set_format("numpy")
    sample = ts_dataset[0]

    # Upscale to float64, some methods need it.
    X = np.asarray(sample[f"timeseries_n{parc_size}"], dtype=np.float64)
    logging.info("Time series shape: %s", X.shape)

    schema = pa.schema(
        {
            "spi": pa.string(),
            "n_samples": pa.int32(),
            "n_features": pa.int32(),
            "success": pa.bool_(),
            "run_time": pa.float32(),
            "peak_mem_mb": pa.float32(),
            "err": pa.string(),
            "mat": pa.list_(pa.float32()),
        }
    )
    records = []

    tracemalloc.start()

    for n, d in zip(n_samples, n_features):
        Xi = scale(X[:n, :d])

        # Monitor memory and run time
        tracemalloc.clear_traces()
        tracemalloc.reset_peak()
        tic = time.perf_counter()

        try:
            spi_cov.fit(Xi)
            err = None
        except Exception as exc:
            err = repr(exc)
            logging.error("SPI %s failed (n=%d, d=%d):\n", spi, n, d, exc_info=exc)

        success = err is None
        run_time = time.perf_counter() - tic
        _, peak_mem = tracemalloc.get_traced_memory()
        peak_mem_mb = peak_mem / 1024**2

        if success:
            mat = spi_cov.covariance_.astype(np.float32).flatten()
        else:
            mat = None

        result = {
            "spi": spi,
            "n_samples": n,
            "n_features": d,
            "success": success,
            "run_time": run_time,
            "peak_mem_mb": peak_mem_mb,
            "err": err,
        }

        logging.info("Result:\n%s", json.dumps(result))

        record = {**result, "mat": mat}
        records.append(record)

        # Write out after every step in case of time out.
        tab = pa.Table.from_pylist(records, schema=schema)
        feather.write_feather(
            tab,
            outdir / f"spi-{spi}__parc-{parc_size}.arrow",
            compression="uncompressed",
        )


if __name__ == "__main__":
    fire.Fire(main)
