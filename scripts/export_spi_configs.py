import logging
import os
from pathlib import Path

import cdt
import yaml
from pyspi.utils import check_optional_deps
from skarf.covariance import load_spi_config_map

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


def main():
    logging.info("Exporting available PySPI SPI configs")

    # Nb, this implicitly starts the JVM.
    logging.info("PySPI optional depedencies:%s", check_optional_deps())

    root = Path(os.environ["PROJECT_ROOT"])
    outdir = root / "resources/spi_lists"
    outdir.mkdir(exist_ok=True)

    n_cpus = len(os.sched_getaffinity(0)) or 1
    # otherwise this is set to cpu_count = 128
    # No GPU automatically detected. Setting SETTINGS.GPU to 0, and SETTINGS.NJOBS to
    # cpu_count.
    cdt.SETTINGS.NJOBS = n_cpus

    spi_config_map, unavailable_spi_configs = load_spi_config_map()
    logging.info("Loaded SPIs: %d\n\n%s", len(spi_config_map), list(spi_config_map))
    logging.info(
        "Unavailable SPIs: %d\n\n%s",
        len(unavailable_spi_configs),
        unavailable_spi_configs,
    )

    with (outdir / "spi_config_map_all.yaml").open("w") as f:
        yaml.safe_dump(spi_config_map, f)

    with (outdir / f"spi_list_all_{len(spi_config_map)}.txt").open("w") as f:
        f.write("\n".join(spi_config_map))


if __name__ == "__main__":
    main()
