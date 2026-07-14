import numpy as np

from arfcexp.homotopy import (
    extract_edge_values_directional,
    get_schaefer_all_crosshemisphere_edges,
    get_schaefer_between_network_edges,
    get_schaefer_overall_between_network_edges,
    get_schaefer_within_network_edges,
    rank_matrix_offdiagonal,
    summarize_between_network_reference_fc_ranked,
    summarize_homotopic_fc_ranked,
)
from arfcexp.schaefer_metadata import (
    get_schaefer_network_hemisphere_indices,
    get_schaefer_network_order,
    parse_schaefer_7network_parcel_name,
)


def test_parse_schaefer_7network_parcel_name():
    hemisphere, network, ordinal = parse_schaefer_7network_parcel_name("7Networks_LH_Vis_12")
    assert hemisphere == "L"
    assert network == "Vis"
    assert ordinal == 12

    hemisphere, network, ordinal = parse_schaefer_7network_parcel_name(
        "7Networks_RH_DorsAttn_Post_7"
    )
    assert hemisphere == "R"
    assert network == "DorsAttn"
    assert ordinal == 7


def test_schaefer_network_hemisphere_counts_are_asymmetric():
    indices = get_schaefer_network_hemisphere_indices(parc_size=200)
    counts = {
        network: (len(groups["L"]), len(groups["R"]))
        for network, groups in indices.items()
    }

    assert tuple(counts) == get_schaefer_network_order(parc_size=200)
    assert counts == {
        "Vis": (14, 15),
        "SomMot": (16, 19),
        "DorsAttn": (13, 13),
        "SalVentAttn": (11, 11),
        "Limbic": (6, 6),
        "Cont": (13, 17),
        "Default": (27, 19),
    }
    assert sum(left for left, _ in counts.values()) == 100
    assert sum(right for _, right in counts.values()) == 100


def test_schaefer_within_network_edge_counts():
    edge_map = get_schaefer_within_network_edges(parc_size=200)
    counts = {network: len(edges) for network, edges in edge_map.items()}

    assert counts == {
        "Vis": 210,
        "SomMot": 304,
        "DorsAttn": 169,
        "SalVentAttn": 121,
        "Limbic": 36,
        "Cont": 221,
        "Default": 513,
    }
    assert sum(counts.values()) == 1574


def test_overall_between_network_edges_are_crosshemisphere_complement():
    all_edges = get_schaefer_all_crosshemisphere_edges(parc_size=200)
    within_edges = np.vstack(list(get_schaefer_within_network_edges(parc_size=200).values()))
    between_edges = get_schaefer_overall_between_network_edges(parc_size=200)

    all_set = set(map(tuple, all_edges.tolist()))
    within_set = set(map(tuple, within_edges.tolist()))
    between_set = set(map(tuple, between_edges.tolist()))

    assert len(all_edges) == 10000
    assert len(within_edges) == 1574
    assert len(between_edges) == 8426
    assert within_set.isdisjoint(between_set)
    assert within_set | between_set == all_set


def test_network_between_edges_exclude_network_within_block():
    within_map = get_schaefer_within_network_edges(parc_size=200)
    between_map = get_schaefer_between_network_edges(parc_size=200)

    for network in get_schaefer_network_order(parc_size=200):
        within_set = set(map(tuple, within_map[network].tolist()))
        between_set = set(map(tuple, between_map[network].tolist()))
        assert len(between_set) == len(between_map[network])
        assert within_set.isdisjoint(between_set)
        assert len(between_set) > len(within_set)


def test_extract_edge_values_directional_accepts_arbitrary_edges():
    mat = np.arange(16, dtype=np.float64).reshape(4, 4)
    edges = np.array([[0, 2], [1, 3]], dtype=np.int32)

    upper, lower = extract_edge_values_directional(mat, parc_size=4, edge_indices=edges)

    assert np.array_equal(upper, np.array([2.0, 7.0]))
    assert np.array_equal(lower, np.array([8.0, 13.0]))


def test_summarize_homotopic_fc_ranked_edges_matches_manual_values():
    mat = np.arange(16, dtype=np.float64).reshape(4, 4)
    edges = np.array([[0, 2], [1, 3]], dtype=np.int32)
    ranked = rank_matrix_offdiagonal(mat, parc_size=4, use_abs_rank=False)

    out = summarize_homotopic_fc_ranked(
        mat,
        parc_size=4,
        edge_indices=edges,
        reducer="mean",
        use_abs_rank=False,
        is_directed=True,
    )
    upper, lower = extract_edge_values_directional(ranked, parc_size=4, edge_indices=edges)

    assert np.isclose(out["homotopic_score_upper"], np.mean(upper))
    assert np.isclose(out["homotopic_score_lower"], np.mean(lower))
    assert np.isclose(out["homotopic_score"], np.mean(np.concatenate([upper, lower])))
    assert out["is_directed"] is True


def test_summarize_between_network_reference_is_deterministic():
    rng = np.random.default_rng(123)
    mat = rng.normal(size=(200, 200))
    edges = get_schaefer_overall_between_network_edges(parc_size=200)

    out1 = summarize_between_network_reference_fc_ranked(
        mat,
        parc_size=200,
        edge_indices=edges,
        reducer="mean",
        use_abs_rank=True,
        is_directed=False,
    )
    out2 = summarize_between_network_reference_fc_ranked(
        mat,
        parc_size=200,
        edge_indices=edges,
        reducer="mean",
        use_abs_rank=True,
        is_directed=False,
    )

    assert out1 == out2
    assert 0.0 <= out1["between_network_reference_mean"] <= 1.0
    assert out1["between_network_reference_q25"] <= out1["between_network_reference_q75"]
    assert np.isfinite(out1["between_network_reference_std"])


def test_rank_matrix_offdiagonal_bounds():
    rng = np.random.default_rng(123)
    mat = rng.normal(size=(200, 200))
    ranked = rank_matrix_offdiagonal(mat, parc_size=200, use_abs_rank=True)

    offdiag = ranked[~np.eye(200, dtype=bool)]
    assert np.nanmin(offdiag) >= 0.0
    assert np.nanmax(offdiag) <= 1.0
    assert np.all(np.isnan(np.diag(ranked)))
