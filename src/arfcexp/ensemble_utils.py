"""Utilities for the krakencoder ensembling module.

Provides:
    - A small registry of ensemble reconstruction rules (mirrors the
      AVAILABLE_SKARF_FUNCS / create_skarf_func pattern in skarf_utils.py).
    - Helpers to load a per-method-flavor bridged krakencoder model
      (checkpoint + input transform) trained by
      scripts/train_ensemble_krakencoder_bridge.py.
    - Encode/decode helpers wrapping the Krakencoder model's forward pass.
    - Three reconstruction rules that combine N methods' FC matrices into a
      single ensemble matrix via the shared pretrained krakencoder latent
      space:
        - "simple_average": equal-weight latent fusion, decode with every
          method's own decoder, average the resulting matrices.
        - "weighted_average": same, but weighted by normalized
          maxnorm_rank_sum at both the latent-fusion and matrix-averaging
          stages.
        - "reference_decoder": equal-weight latent fusion, decode with only
          the #1-ranked method's own decoder.
"""

import json
from pathlib import Path

import numpy as np
import torch
from krakencoder.data import load_transformers_from_file
from krakencoder.model import Krakencoder
from krakencoder.utils import numpyvar, torchfloat, tri2square

AVAILABLE_ENSEMBLE_RULES = ["simple_average", "weighted_average", "reference_decoder"]


def load_ensemble_manifest(manifest_path: str | Path) -> dict:
    """Load the manifest.json produced by train_ensemble_krakencoder_bridge.py."""
    return json.loads(Path(manifest_path).read_text())


def load_flavor_model(combo_key: str, manifest: dict) -> tuple[Krakencoder, object]:
    """Load a single method flavor's bridged krakencoder encoder/decoder + input transform.

    Args:
        combo_key: method/func/lag combo key (e.g. "skarf__linear_ridge__lag-0").
        manifest: dict loaded via load_ensemble_manifest.

    Returns:
        (net, transformer) where net is a Krakencoder model with a single
        self-only encoder/decoder path (index 0), and transformer is the
        fitted PCA input transform for this flavor.
    """
    if combo_key not in manifest:
        raise KeyError(f"No manifest entry for {combo_key}")
    entry = manifest[combo_key]
    if entry.get("status") != "success":
        raise ValueError(f"No successful checkpoint for {combo_key}: {entry}")

    net, _extra = Krakencoder.load_checkpoint(entry["checkpoint"], eval_mode=True)
    transformer_list, _info = load_transformers_from_file(entry["xform"], input_names=[combo_key])
    transformer = transformer_list[combo_key]
    return net, transformer


def load_flavor_models(combo_keys: list[str], manifest: dict) -> dict[str, tuple[Krakencoder, object]]:
    """Load flavor models for a list of combo keys."""
    return {key: load_flavor_model(key, manifest) for key in combo_keys}


def _matrix_to_triu_vector(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    triu = np.triu_indices(n, k=1)
    return matrix[triu]


def encode_matrix(net: Krakencoder, transformer: object, matrix: np.ndarray) -> np.ndarray:
    """Encode a single [parc x parc] connectivity matrix to its 128-d latent vector."""
    vec = _matrix_to_triu_vector(matrix)[np.newaxis, :]
    x = transformer.transform(torchfloat(vec))
    with torch.no_grad():
        latent = net(x, 0, -1)  # encoder_index=0 (self-only path), decoder_index=-1 -> latent only
    return numpyvar(latent)


def decode_latent(net: Krakencoder, transformer: object, latent: np.ndarray, parc: int) -> np.ndarray:
    """Decode a [1 x 128] latent vector back to a [parc x parc] connectivity matrix."""
    latent_t = torchfloat(latent)
    with torch.no_grad():
        _, decoded = net(latent_t, -1, 0)  # encoder_index=-1 (already latent), decoder_index=0
    decoded_np = numpyvar(transformer.inverse_transform(decoded))
    triu = np.triu_indices(parc, k=1)
    return tri2square(decoded_np[0], tri_indices=triu, numroi=parc, diagval=0.0)


def _encode_all(
    combo_matrices: dict[str, np.ndarray],
    models: dict[str, tuple[Krakencoder, object]],
) -> dict[str, np.ndarray]:
    return {key: encode_matrix(*models[key], mat) for key, mat in combo_matrices.items()}


def reconstruct_simple_average(
    combo_matrices: dict[str, np.ndarray],
    models: dict[str, tuple[Krakencoder, object]],
    parc: int,
) -> np.ndarray:
    """Equal-weight latent fusion; decode with every method's own decoder and average."""
    latents = _encode_all(combo_matrices, models)
    fused_latent = np.mean(list(latents.values()), axis=0)
    decoded = [decode_latent(*models[key], fused_latent, parc) for key in combo_matrices]
    return np.mean(decoded, axis=0)


def reconstruct_weighted_average(
    combo_matrices: dict[str, np.ndarray],
    models: dict[str, tuple[Krakencoder, object]],
    weights: dict[str, float],
    parc: int,
) -> np.ndarray:
    """Rank-weighted latent fusion; decode with every decoder and rank-weighted average."""
    keys = list(combo_matrices.keys())
    w = np.array([weights[key] for key in keys], dtype=float)
    w = w / w.sum()

    latents = np.stack([encode_matrix(*models[key], combo_matrices[key])[0] for key in keys])
    fused_latent = (w[:, np.newaxis] * latents).sum(axis=0, keepdims=True)

    decoded = np.stack([decode_latent(*models[key], fused_latent, parc) for key in keys])
    return (w[:, np.newaxis, np.newaxis] * decoded).sum(axis=0)


def reconstruct_reference_decoder(
    combo_matrices: dict[str, np.ndarray],
    models: dict[str, tuple[Krakencoder, object]],
    reference_key: str,
    parc: int,
) -> np.ndarray:
    """Equal-weight latent fusion; decode with only the #1-ranked method's own decoder."""
    latents = _encode_all(combo_matrices, models)
    fused_latent = np.mean(list(latents.values()), axis=0)
    return decode_latent(*models[reference_key], fused_latent, parc)


def apply_ensemble_rule(
    rule: str,
    combo_matrices: dict[str, np.ndarray],
    models: dict[str, tuple[Krakencoder, object]],
    parc: int,
    weights: dict[str, float] | None = None,
    reference_key: str | None = None,
) -> np.ndarray:
    """Dispatch to the requested reconstruction rule.

    Args:
        rule: one of AVAILABLE_ENSEMBLE_RULES.
        combo_matrices: {combo_key: [parc x parc] matrix} for one subject.
        models: {combo_key: (net, transformer)} as returned by load_flavor_models.
        parc: number of ROIs (e.g. 200 for Schaefer-200).
        weights: required for "weighted_average" - {combo_key: weight}.
        reference_key: required for "reference_decoder" - the #1-ranked combo_key.
    """
    match rule:
        case "simple_average":
            return reconstruct_simple_average(combo_matrices, models, parc)
        case "weighted_average":
            if weights is None:
                raise ValueError("weights required for weighted_average rule")
            return reconstruct_weighted_average(combo_matrices, models, weights, parc)
        case "reference_decoder":
            if reference_key is None:
                raise ValueError("reference_key required for reference_decoder rule")
            return reconstruct_reference_decoder(combo_matrices, models, reference_key, parc)
        case _:
            raise NotImplementedError(f"Unknown ensemble rule: {rule}")
