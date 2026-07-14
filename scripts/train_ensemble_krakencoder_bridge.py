"""Bridge each top-N ensemble method into krakencoder's pretrained latent space.

For each method flavor in resources/ensemble_method_lists/top{N}_methods.csv,
invokes krakencoder's run_training.py with the "adding a new flavor" recipe
(--targetencoding --onlyselfpathtargetencoding --targetencodingname fusion),
training a lightweight self-only encoder/decoder that maps the method's FC
matrices to/from the pretrained model's existing 128-d fusion latent space.
Each flavor is trained independently (no cross-flavor paths), so training one
flavor does not depend on any other.

Hyperparameters follow the krakencoder manuscript's "adding a new flavor"
recipe: 128-d latent (unit-normalized), 256-d PCA input transform, linear
encoder/decoder, 50% dropout, 500 epochs (~1-2 min/flavor per the manuscript).

Requires scripts/build_krakencoder_bridge_inputs.py to have been run first.

Usage:
    uv run python scripts/train_ensemble_krakencoder_bridge.py
    uv run python scripts/train_ensemble_krakencoder_bridge.py --top-n 5
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

DEFAULT_METHOD_LIST_DIR = "resources/ensemble_method_lists"
DEFAULT_BRIDGE_DIR = "data/ensemble_krakencoder/bridge"
DEFAULT_OUT_DIR = "data/ensemble_krakencoder/checkpoints"
DEFAULT_TOP_N = [5, 10, 15]

LATENT_SIZE = 128
TRANSFORMATION = "pca256"
DROPOUT = 0.5
LOSS_TYPE = "correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000"
EPOCHS = 500
CHECKPOINT_EPOCHS_EVERY = 500
DISPLAY_EPOCHS = 100
RANDOM_SEED = 0


def _find_run_training_script(project_root: Path) -> Path:
    path = project_root / "submodules/krakencoder/run_training.py"
    if not path.exists():
        raise FileNotFoundError(f"krakencoder run_training.py not found at {path}")
    return path


def main(
    method_list_dir: str | None = None,
    bridge_dir: str | None = None,
    out_dir: str | None = None,
    top_n: list[int] = DEFAULT_TOP_N,
    epochs: int = EPOCHS,
    force: bool = False,
):
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))

    method_list_dir_path = (
        Path(method_list_dir) if method_list_dir is not None
        else project_root / DEFAULT_METHOD_LIST_DIR
    )
    bridge_dir_path = Path(bridge_dir) if bridge_dir is not None else project_root / DEFAULT_BRIDGE_DIR
    out_dir_path = Path(out_dir) if out_dir is not None else project_root / DEFAULT_OUT_DIR
    out_dir_path.mkdir(parents=True, exist_ok=True)

    run_training_script = _find_run_training_script(project_root)

    manifest: dict[str, dict] = {}
    manifest_path = out_dir_path / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())

    # Deduplicate combos across N's — each combo only needs to be trained once.
    combos_seen: dict[str, tuple[int, str]] = {}
    for n in sorted(top_n):
        list_path = method_list_dir_path / f"top{n}_methods.csv"
        if not list_path.exists():
            raise FileNotFoundError(f"{list_path} not found. Run scripts/select_ensemble_methods.py first.")
        methods_df = pd.read_csv(list_path)
        for row in methods_df.itertuples(index=False):
            if row.combo_key not in combos_seen:
                combos_seen[row.combo_key] = (n, row.combo_key)

    logging.info(f"Training {len(combos_seen)} unique method flavors: {sorted(combos_seen)}")

    for combo_key, (n, _) in combos_seen.items():
        if combo_key in manifest and not force:
            logging.info(f"Skipping {combo_key} (already in manifest; use --force to retrain)")
            continue

        n_bridge_dir = bridge_dir_path / f"top{n}"
        flavor_mat = n_bridge_dir / f"{combo_key}.mat"
        subject_split_mat = n_bridge_dir / "bridge_subject_split.mat"
        fusion_target_mat = n_bridge_dir / "bridge_fusion_target.mat"
        for path in [flavor_mat, subject_split_mat, fusion_target_mat]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run scripts/build_krakencoder_bridge_inputs.py first."
                )

        combo_out_dir = out_dir_path / combo_key
        combo_out_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = str(combo_out_dir / combo_key)

        cmd = [
            sys.executable,
            str(run_training_script),
            "--inputdata", f"{combo_key}={flavor_mat}",
            "--subjectfile", str(subject_split_mat),
            "--encodedinputfile", str(fusion_target_mat),
            "--targetencoding",
            "--onlyselfpathtargetencoding",
            "--targetencodingname", "fusion",
            "--latentsize", str(LATENT_SIZE),
            "--latentunit",
            "--transformation", TRANSFORMATION,
            "--dropout", str(DROPOUT),
            "--losstype", LOSS_TYPE,
            "--epochs", str(epochs),
            "--checkpointepochsevery", str(CHECKPOINT_EPOCHS_EVERY),
            "--displayepochs", str(DISPLAY_EPOCHS),
            "--outputprefix", output_prefix,
            "--randseed", str(RANDOM_SEED),
        ]

        logging.info(f"Training flavor {combo_key} (from top{n} list) ...")
        logging.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)

        log_path = combo_out_dir / "train_log.txt"
        log_path.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)

        if result.returncode != 0:
            logging.error(f"Training FAILED for {combo_key} (see {log_path}); return code {result.returncode}")
            manifest[combo_key] = {"status": "failed", "log": str(log_path)}
            continue

        checkpoints = sorted(combo_out_dir.glob(f"{combo_key}_chkpt_*_ep*.pt"))
        xforms = sorted(combo_out_dir.glob(f"{combo_key}_ioxfm_*.npy"))
        if not checkpoints:
            logging.error(f"Training completed but no checkpoint found for {combo_key} (see {log_path})")
            manifest[combo_key] = {"status": "missing_checkpoint", "log": str(log_path)}
            continue

        manifest[combo_key] = {
            "status": "success",
            "checkpoint": str(checkpoints[-1]),
            "xform": str(xforms[-1]) if xforms else None,
            "log": str(log_path),
            "top_n_source": n,
        }
        logging.info(f"Success: {combo_key} -> {checkpoints[-1]}")

        manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    n_success = sum(1 for v in manifest.values() if v.get("status") == "success")
    logging.info(f"Done. {n_success}/{len(manifest)} flavors trained successfully. Manifest: {manifest_path}")


if __name__ == "__main__":
    typer.run(main)
