import numpy as np
import pytest

from arfcexp.homotopy import (
    extract_homotopic_values_directional,
    get_homotopic_partner_indices,
    get_schaefer_homotopic_pairs,
    rank_matrix_offdiagonal,
    summarize_heterotopic_null_fc_ranked,
    summarize_homotopic_fc_ranked,
    validate_schaefer_homotopic_pairs,
)


def test_get_schaefer_homotopic_pairs():
    pairs = get_schaefer_homotopic_pairs(parc_size=200, zero_indexed=True)
    assert pairs.shape == (100, 2)
    assert np.all(pairs[:, 0] == np.arange(100))
    assert np.all(pairs[:, 1] == np.arange(100, 200))


def test_partner_indices_round_trip():
    partner = get_homotopic_partner_indices(parc_size=200)
    assert partner.shape == (200,)
    for idx in range(200):
        assert partner[partner[idx]] == idx


def test_extract_homotopic_values_directional():
    mat = np.zeros((200, 200), dtype=np.float64)
    pairs = get_schaefer_homotopic_pairs(parc_size=200)

    upper_vals = np.linspace(0.0, 1.0, len(pairs))
    lower_vals = np.linspace(2.0, 3.0, len(pairs))
    mat[pairs[:, 0], pairs[:, 1]] = upper_vals
    mat[pairs[:, 1], pairs[:, 0]] = lower_vals

    upper, lower = extract_homotopic_values_directional(mat, parc_size=200)
    assert np.allclose(upper, upper_vals)
    assert np.allclose(lower, lower_vals)


def test_summarize_homotopic_fc_ranked_directed_has_upper_lower():
    mat = np.zeros((200, 200), dtype=np.float64)
    pairs = get_schaefer_homotopic_pairs(parc_size=200)

    mat[pairs[:, 0], pairs[:, 1]] = np.linspace(0.1, 0.9, len(pairs))
    mat[pairs[:, 1], pairs[:, 0]] = np.linspace(0.9, 0.1, len(pairs))

    out = summarize_homotopic_fc_ranked(
        mat,
        parc_size=200,
        reducer="mean",
        use_abs_rank=True,
        is_directed=True,
    )
    assert np.isfinite(out["homotopic_score"])
    assert np.isfinite(out["homotopic_score_upper"])
    assert np.isfinite(out["homotopic_score_lower"])
    assert out["is_directed"] is True


def test_rank_matrix_offdiagonal_bounds():
    rng = np.random.default_rng(123)
    mat = rng.normal(size=(200, 200))
    ranked = rank_matrix_offdiagonal(mat, parc_size=200, use_abs_rank=True)

    offdiag = ranked[~np.eye(200, dtype=bool)]
    assert np.nanmin(offdiag) >= 0.0
    assert np.nanmax(offdiag) <= 1.0
    assert np.all(np.isnan(np.diag(ranked)))


def test_validate_schaefer_homotopic_pairs():
    assert validate_schaefer_homotopic_pairs(parc_size=200)


def test_summarize_heterotopic_null_fc_ranked_bounds_and_ordering():
    rng = np.random.default_rng(123)
    mat = rng.normal(size=(200, 200))

    out = summarize_heterotopic_null_fc_ranked(
        mat,
        parc_size=200,
        reducer="mean",
        use_abs_rank=True,
        is_directed=False,
        n_perm=100,
        seed=42,
    )

    assert 0.0 <= out["heterotopic_null_mean"] <= 1.0
    assert out["heterotopic_null_q25"] <= out["heterotopic_null_mean"] <= out["heterotopic_null_q75"]
    assert np.isfinite(out["heterotopic_null_std"])


def test_summarize_heterotopic_null_fc_ranked_reproducible_seed():
    rng = np.random.default_rng(321)
    mat = rng.normal(size=(200, 200))

    out1 = summarize_heterotopic_null_fc_ranked(
        mat,
        parc_size=200,
        reducer="mean",
        use_abs_rank=True,
        is_directed=False,
        n_perm=80,
        seed=777,
    )
    out2 = summarize_heterotopic_null_fc_ranked(
        mat,
        parc_size=200,
        reducer="mean",
        use_abs_rank=True,
        is_directed=False,
        n_perm=80,
        seed=777,
    )

    assert out1 == out2


def test_summarize_heterotopic_null_fc_ranked_directed_has_upper_lower_stats():
    rng = np.random.default_rng(999)
    mat = rng.normal(size=(200, 200))

    out = summarize_heterotopic_null_fc_ranked(
        mat,
        parc_size=200,
        reducer="mean",
        use_abs_rank=True,
        is_directed=True,
        n_perm=120,
        seed=9,
    )

    assert 0.0 <= out["heterotopic_null_mean_upper"] <= 1.0
    assert 0.0 <= out["heterotopic_null_mean_lower"] <= 1.0
    assert out["heterotopic_null_q25_upper"] <= out["heterotopic_null_q75_upper"]
    assert out["heterotopic_null_q25_lower"] <= out["heterotopic_null_q75_lower"]
