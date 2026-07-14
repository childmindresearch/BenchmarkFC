"""Build krakencoder bridging inputs: subject overlap + per-flavor .mat files.

Computes the overlap between krakencoder's pretrained-model training cohort
(993 HCP-YA subjects, from ``subject_splits_993subj_..._retestInTest.mat``) and
our 867-subject HCP-1200 cohort, then writes:

    - A shared subject-split .mat (subjects restricted to the overlap, with
      subjidx_train/val/test remapped to indices into the reduced list) —
      used as ``--subjectfile`` for every per-flavor training run.
    - A shared fusion-latent target .mat (``encoded`` field, restricted +
      reordered to the overlap subjects) — used as ``--encodedinputfile``.
    - One per-method-flavor input .mat (``data`` field, [N_overlap x parc x
      parc] square matrices, restricted + reordered to the overlap subjects)
      — used as the ``--inputdata`` flavor for
      scripts/train_ensemble_krakencoder_bridge.py.

All files are written under data/ensemble_krakencoder/bridge/top{N}/.

Usage:
    uv run python scripts/build_krakencoder_bridge_inputs.py
    uv run python scripts/build_krakencoder_bridge_inputs.py --top-n 5 --top-n 10
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from krakencoder.data_notorch import clean_subject_list
from scipy.io import loadmat, savemat

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

DEFAULT_SUBJECT_SPLIT_FILE = (
    "data/krakencoder_cache/subject_splits_993subj_683train_79val_196test_retestInTest.mat"
)
DEFAULT_FUSION_TARGET_FILE = (
    "data/krakencoder_cache/kraken_target_hcp_20240413_210723_ep002000_mse.w1000_encoded.mat"
)
DEFAULT_METHOD_LIST_DIR = "resources/ensemble_method_lists"
DEFAULT_INPUT_DIR = "data/ensemble_krakencoder/inputs"
DEFAULT_OUT_DIR = "data/ensemble_krakencoder/bridge"
DEFAULT_TOP_N = [5, 10, 15]


def main(
    subject_split_file: str | None = None,
    fusion_target_file: str | None = None,
    method_list_dir: str | None = None,
    input_dir: str | None = None,
    out_dir: str | None = None,
    top_n: list[int] = DEFAULT_TOP_N,
):
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))

    subject_split_path = (
        Path(subject_split_file) if subject_split_file is not None
        else project_root / DEFAULT_SUBJECT_SPLIT_FILE
    )
    fusion_target_path = (
        Path(fusion_target_file) if fusion_target_file is not None
        else project_root / DEFAULT_FUSION_TARGET_FILE
    )
    method_list_dir_path = (
        Path(method_list_dir) if method_list_dir is not None
        else project_root / DEFAULT_METHOD_LIST_DIR
    )
    input_dir_path = Path(input_dir) if input_dir is not None else project_root / DEFAULT_INPUT_DIR
    out_dir_path = Path(out_dir) if out_dir is not None else project_root / DEFAULT_OUT_DIR

    for path in [subject_split_path, fusion_target_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/fetch_krakencoder_fusion_target.py first."
            )

    # --- Load krakencoder's 993-subject cohort + splits ---
    split_mat = loadmat(subject_split_path, simplify_cells=True)
    kraken_subjects = clean_subject_list(split_mat["subjects"])
    subjidx_train = np.asarray(split_mat["subjidx_train"]).astype(int)
    subjidx_val = np.asarray(split_mat["subjidx_val"]).astype(int)
    subjidx_test = np.asarray(split_mat["subjidx_test"]).astype(int)
    logging.info(
        f"Loaded krakencoder subject split: {len(kraken_subjects)} subjects "
        f"({len(subjidx_train)} train / {len(subjidx_val)} val / {len(subjidx_test)} test)"
    )

    # --- Load precomputed fusion latent target ---
    target_mat = loadmat(fusion_target_path, simplify_cells=True)
    target_subjects = clean_subject_list(target_mat["subjects"])
    fusion_encoded = target_mat["predicted_alltypes"]["fusion"]["encoded"]
    if not np.array_equal(target_subjects, kraken_subjects):
        raise ValueError(
            "Fusion target subject order does not match subject split file order; "
            "expected identical ordering per krakencoder OSF release."
        )
    logging.info(f"Loaded fusion latent target: shape={fusion_encoded.shape}")

    for n in sorted(top_n):
        list_path = method_list_dir_path / f"top{n}_methods.csv"
        if not list_path.exists():
            raise FileNotFoundError(
                f"Method list not found: {list_path}. Run scripts/select_ensemble_methods.py first."
            )
        methods_df = pd.read_csv(list_path)

        n_input_dir = input_dir_path / f"top{n}"
        with open(n_input_dir / "subjects.txt") as f:
            our_subjects = clean_subject_list([line.strip() for line in f if line.strip()])
        our_sub_to_idx = {sub: i for i, sub in enumerate(our_subjects)}

        # Overlap order = krakencoder's 993-subject order, filtered to subjects
        # present in our 867-subject export.
        our_sub_set = set(our_subjects)
        overlap_mask = np.array([sub in our_sub_set for sub in kraken_subjects])
        overlap_subjects = kraken_subjects[overlap_mask]
        n_overlap = len(overlap_subjects)
        frac = n_overlap / len(kraken_subjects)
        logging.info(
            f"top{n}: subject overlap = {n_overlap}/{len(kraken_subjects)} "
            f"krakencoder cohort ({frac:.1%}), out of {len(our_subjects)} in our cohort"
        )
        if n_overlap == 0:
            raise ValueError(f"top{n}: zero subject overlap with krakencoder cohort; cannot bridge.")

        n_out_dir = out_dir_path / f"top{n}"
        n_out_dir.mkdir(parents=True, exist_ok=True)

        # Remap subjidx_train/val/test (indices into the original 993 list) to
        # indices into the reduced overlap list.
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.flatnonzero(overlap_mask))}
        subjidx_train_new = np.array(
            [old_to_new[i] for i in subjidx_train if i in old_to_new], dtype=int
        )
        subjidx_val_new = np.array(
            [old_to_new[i] for i in subjidx_val if i in old_to_new], dtype=int
        )
        subjidx_test_new = np.array(
            [old_to_new[i] for i in subjidx_test if i in old_to_new], dtype=int
        )
        logging.info(
            f"top{n}: remapped split = {len(subjidx_train_new)} train / "
            f"{len(subjidx_val_new)} val / {len(subjidx_test_new)} test"
        )

        split_out_path = n_out_dir / "bridge_subject_split.mat"
        savemat(
            split_out_path,
            {
                "subjects": np.array([str(s) for s in overlap_subjects], dtype=object),
                "subjidx_train": subjidx_train_new,
                "subjidx_val": subjidx_val_new,
                "subjidx_test": subjidx_test_new,
            },
            format="5",
            do_compression=True,
        )
        logging.info(f"Wrote {split_out_path}")

        fusion_out_path = n_out_dir / "bridge_fusion_target.mat"
        savemat(
            fusion_out_path,
            {
                "subjects": np.array([str(s) for s in overlap_subjects], dtype=object),
                "encoded": fusion_encoded[overlap_mask].astype(np.float32),
            },
            format="5",
            do_compression=True,
        )
        logging.info(f"Wrote {fusion_out_path} (encoded shape={fusion_encoded[overlap_mask].shape})")

        for row in methods_df.itertuples(index=False):
            combo_key = row.combo_key
            npy_path = n_input_dir / f"{combo_key}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"{npy_path} not found. Run scripts/export_ensemble_input_matrices.py first."
                )
            full_mats = np.load(npy_path)  # (867, parc, parc), ordered per our_subjects
            row_indices = np.array([our_sub_to_idx[sub] for sub in overlap_subjects])
            overlap_mats = full_mats[row_indices].astype(np.float32)

            flavor_out_path = n_out_dir / f"{combo_key}.mat"
            savemat(
                flavor_out_path,
                {
                    "subjects": np.array([str(s) for s in overlap_subjects], dtype=object),
                    "data": overlap_mats,
                },
                format="5",
                do_compression=True,
            )
            logging.info(f"Wrote {flavor_out_path} (data shape={overlap_mats.shape})")

    logging.info("Done.")


if __name__ == "__main__":
    typer.run(main)
