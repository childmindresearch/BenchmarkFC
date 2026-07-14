import numpy as np

from arfcexp.weight_distance import (
    compute_centroid_distance_matrix,
    extract_triangle_values,
    precompute_null_distance_ranks,
    rank_normalize,
    score_weight_distance_matrix,
    spearman_corr,
)


def test_compute_centroid_distance_matrix():
    centroids = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
        ]
    )
    dist = compute_centroid_distance_matrix(centroids)

    assert dist.shape == (3, 3)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.isclose(dist[0, 1], 5.0)
    assert np.isclose(dist[0, 2], 12.0)
    assert np.allclose(dist, dist.T)


def test_extract_triangle_values():
    mat = np.arange(16).reshape(4, 4)

    assert np.array_equal(extract_triangle_values(mat, "upper"), np.array([1, 2, 3, 6, 7, 11]))
    assert np.array_equal(extract_triangle_values(mat, "lower"), np.array([4, 8, 9, 12, 13, 14]))
    assert len(extract_triangle_values(mat, "offdiag")) == 12


def test_spearman_corr_signed_behavior():
    x = np.arange(10, dtype=np.float64)
    y = x * 2
    z = -x

    assert np.isclose(spearman_corr(x, y), 1.0)
    assert np.isclose(spearman_corr(x, z), -1.0)


def test_spearman_corr_constant_returns_nan():
    x = np.arange(5, dtype=np.float64)
    y = np.ones(5)

    assert np.isnan(spearman_corr(x, y))


def test_rank_normalize_dot_is_spearman():
    x = np.array([4.0, 1.0, 2.0, 3.0])
    y = np.array([40.0, 10.0, 20.0, 30.0])

    assert np.isclose(np.dot(rank_normalize(x), rank_normalize(y)), 1.0)


def test_score_weight_distance_symmetric_uses_upper_triangle():
    dist = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    mat = dist.copy()

    out = score_weight_distance_matrix(mat, dist, is_directed=False)

    assert np.isclose(out["weight_distance_score"], 1.0)
    assert np.isclose(out["weight_distance_score_upper"], 1.0)
    assert np.isnan(out["weight_distance_score_lower"])
    assert out["is_directed"] is False


def test_score_weight_distance_directed_has_upper_lower_and_offdiag():
    dist = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    mat = np.zeros((3, 3), dtype=np.float64)
    iu = np.triu_indices(3, k=1)
    il = np.tril_indices(3, k=-1)
    mat[iu] = dist[iu]
    mat[il] = -dist[il]

    out = score_weight_distance_matrix(mat, dist, is_directed=True)

    assert np.isclose(out["weight_distance_score_upper"], 1.0)
    assert np.isclose(out["weight_distance_score_lower"], -1.0)
    assert np.isfinite(out["weight_distance_score_offdiag"])
    assert out["is_directed"] is True


def test_precompute_null_distance_ranks_shape_and_permutation_effect():
    dist = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 0.0, 6.0],
            [3.0, 5.0, 6.0, 0.0],
        ]
    )
    spins = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)

    null_ranks = precompute_null_distance_ranks(dist, spins, triangle="upper")


    assert null_ranks.shape == (2, 6)
    assert np.all(np.isfinite(null_ranks))
    assert not np.allclose(null_ranks[0], null_ranks[1])


def test_score_weight_distance_with_null_summary():
    dist = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 0.0, 6.0],
            [3.0, 5.0, 6.0, 0.0],
        ]
    )
    spins = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2]], dtype=np.int32)
    null_upper = precompute_null_distance_ranks(dist, spins, triangle="upper")

    out = score_weight_distance_matrix(
        dist,
        dist,
        is_directed=False,
        null_ranks_upper=null_upper,
    )

    assert np.isfinite(out["spin_null_mean"])
    assert np.isfinite(out["spin_null_q25"])
    assert np.isfinite(out["spin_null_p_two_sided"])