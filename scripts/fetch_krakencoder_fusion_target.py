"""Fetch krakencoder's precomputed fusion-latent target and subject split file.

Downloads two files from the krakencoder project's fetchable data registry
(hosted on OSF, mirrored on Box) into the krakencoder cache directory
(``$KRAKENCODER_DATA``, see krakencoder/fetch.py):

    - kraken_target_hcp_20240413_210723_ep002000_mse.w1000_encoded.mat
      993 HCP-YA subject x 128-d precomputed "fusion" latent representations.
      Used as the ``--encodedinputfile`` training target when bridging a new
      connectivity flavor into the pretrained krakencoder latent space (see
      "Extending the Krakencoder with new connectivity flavors" in the
      manuscript).
    - subject_splits_993subj_683train_79val_196test_retestInTest.mat
      The 993 HCP-YA subject IDs (in the same order as the fusion latents
      above) plus train/val/test split indices used to train the original
      pretrained model.

Usage:
    uv run python scripts/fetch_krakencoder_fusion_target.py
"""

import logging
import os

import typer
from krakencoder.fetch import fetch_model_data, model_data_folder

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

FUSION_TARGET_FILENAME = "kraken_target_hcp_20240413_210723_ep002000_mse.w1000_encoded.mat"
SUBJECT_SPLIT_FILENAME = "subject_splits_993subj_683train_79val_196test_retestInTest.mat"


def main(force: bool = False):
    cache_dir = model_data_folder()
    logging.info(f"krakencoder cache directory: {cache_dir}")
    if "KRAKENCODER_DATA" not in os.environ:
        logging.warning(
            "KRAKENCODER_DATA env var not set; using default OS cache location above. "
            "Set KRAKENCODER_DATA in .env to control where krakencoder data is stored."
        )

    files_to_fetch = [FUSION_TARGET_FILENAME, SUBJECT_SPLIT_FILENAME]
    logging.info(f"Fetching: {files_to_fetch}")
    paths = fetch_model_data(files_to_fetch=files_to_fetch, force_download=force, verbose=True)

    for name, path in zip(files_to_fetch, paths):
        logging.info(f"{name} -> {path}")

    logging.info("Done.")


if __name__ == "__main__":
    typer.run(main)
