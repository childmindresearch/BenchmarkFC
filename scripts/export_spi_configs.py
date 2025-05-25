import logging
import os
from pathlib import Path

from skarf.covariance import load_spi_config_map, load_pyspi_optional_deps

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


def main():
    logging.info("Exporting available PySPI SPI configs")

    root = Path(os.environ["PROJECT_ROOT"])
    outdir = root / "resources/spi_lists"
    outdir.mkdir(exist_ok=True)

    load_pyspi_optional_deps()
    spi_config_map, unavailable_spi_configs = load_spi_config_map(cache_dir=outdir)
    logging.info("Loaded SPIs: %d\n\n%s", len(spi_config_map), list(spi_config_map))
    logging.info(
        "Unavailable SPIs: %d\n\n%s",
        len(unavailable_spi_configs),
        unavailable_spi_configs,
    )

    # Save list of all SPIs.
    all_spis = list(spi_config_map)
    with (outdir / f"spi_list_all_{len(all_spis)}.txt").open("w") as f:
        print("\n".join(all_spis), file=f)

    # Exclude squared SPIs which are redundant and can be easily computed after the
    # fact.
    distinct_spis = []
    for spi in all_spis:
        identifier = spi.split("_")[0]
        if not identifier.endswith("-sq"):
            distinct_spis.append(spi)

    logging.info(f"Found {len(distinct_spis)}/{len(all_spis)} distinct SPIs.")
    with (outdir / f"spi_list_distinct_{len(distinct_spis)}.txt").open("w") as f:
        print("\n".join(distinct_spis), file=f)


if __name__ == "__main__":
    main()
