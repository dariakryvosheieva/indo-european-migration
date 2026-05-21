"""Sanity checks for branch-level analysis utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pie_sim import (
    CognateEvolver,
    IE_BRANCH_SAMPLES,
    Population,
    build_upgma_tree,
    cognate_distance_matrix,
    extract_branch_lexicons,
    stylized_eurasia,
)


# ----- IE_BRANCH_SAMPLES sanity ------------------------------------------


def test_branch_samples_have_unique_names():
    names = [s.name for s in IE_BRANCH_SAMPLES]
    assert len(names) == len(set(names))


def test_branch_samples_are_inside_grid_and_habitable():
    geo = stylized_eurasia(cell_size=1.0)
    for s in IE_BRANCH_SAMPLES:
        assert geo.lat_min <= s.lat < geo.lat_max, s.name
        assert geo.lon_min <= s.lon < geo.lon_max, s.name
        i, j = geo.coord_to_cell(s.lat, s.lon)
        assert geo.is_habitable(i, j), f"{s.name} at ({s.lat}, {s.lon}) is on ocean"


def test_branch_samples_cover_major_families():
    families = {s.family for s in IE_BRANCH_SAMPLES}
    expected = {
        "Anatolian", "Tocharian", "Indo-Iranian", "Hellenic",
        "Armenian", "Albanian", "Italic", "Celtic", "Germanic", "Balto-Slavic",
    }
    assert expected.issubset(families)


# ----- Extraction --------------------------------------------------------


def _seeded_pop(geo, lat, lon, ev) -> Population:
    pop = Population.empty(geo)
    pop.seed(lat, lon, count=0.5, lexicon=ev.initial_lexicon())
    return pop


def test_extract_finds_exact_match():
    geo = stylized_eurasia(cell_size=1.0)
    ev = CognateEvolver(n_meanings=20, rng=np.random.default_rng(0))
    sample = IE_BRANCH_SAMPLES[0]
    pop = _seeded_pop(geo, sample.lat, sample.lon, ev)
    out = extract_branch_lexicons(pop, [sample])
    assert sample.name in out
    s, lex, cell = out[sample.name]
    assert s == sample
    assert cell == geo.coord_to_cell(sample.lat, sample.lon)


def test_extract_falls_back_to_nearby_cell():
    """If the sample cell isn't populated, the nearest populated one is used."""
    geo = stylized_eurasia(cell_size=1.0)
    ev = CognateEvolver(n_meanings=20, rng=np.random.default_rng(0))
    sample = IE_BRANCH_SAMPLES[0]
    pop = _seeded_pop(geo, sample.lat + 1.5, sample.lon + 1.5, ev)
    out = extract_branch_lexicons(pop, [sample], max_distance_km=500.0)
    assert sample.name in out


def test_extract_skips_distant_samples():
    """Samples beyond max_distance_km are not returned."""
    geo = stylized_eurasia(cell_size=1.0)
    ev = CognateEvolver(n_meanings=20, rng=np.random.default_rng(0))
    pop = _seeded_pop(geo, 50.0, 45.0, ev)  # only the steppe seed
    out = extract_branch_lexicons(pop, [IE_BRANCH_SAMPLES[0]], max_distance_km=10.0)
    assert IE_BRANCH_SAMPLES[0].name not in out


def test_extract_handles_empty_population():
    geo = stylized_eurasia(cell_size=1.0)
    pop = Population.empty(geo)
    out = extract_branch_lexicons(pop, IE_BRANCH_SAMPLES)
    assert out == {}


# ----- Distance matrix ---------------------------------------------------


def test_distance_matrix_shape_and_symmetry():
    geo = stylized_eurasia(cell_size=1.0)
    ev = CognateEvolver(n_meanings=30, rng=np.random.default_rng(0))
    samples = IE_BRANCH_SAMPLES[:5]
    pop = Population.empty(geo)
    for s in samples:
        pop.seed(s.lat, s.lon, count=0.5, lexicon=ev.initial_lexicon())
    extracted = extract_branch_lexicons(pop, samples)
    M, names = cognate_distance_matrix(extracted)
    assert M.shape == (5, 5)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)
    # All extracted samples were created with disjoint fresh IDs, so distances
    # should all be exactly 1.
    off_diag = M[~np.eye(5, dtype=bool)]
    assert np.allclose(off_diag, 1.0)


def test_distance_matrix_respects_explicit_order():
    geo = stylized_eurasia(cell_size=1.0)
    ev = CognateEvolver(n_meanings=20, rng=np.random.default_rng(0))
    samples = IE_BRANCH_SAMPLES[:3]
    pop = Population.empty(geo)
    for s in samples:
        pop.seed(s.lat, s.lon, count=0.5, lexicon=ev.initial_lexicon())
    extracted = extract_branch_lexicons(pop, samples)
    desired = [samples[2].name, samples[0].name, samples[1].name]
    M, names = cognate_distance_matrix(extracted, order=desired)
    assert names == desired


# ----- UPGMA tree --------------------------------------------------------


def test_upgma_groups_close_samples_together():
    """Samples with smaller pairwise distance should fuse first."""
    M = np.array([
        [0.0, 0.1, 0.5, 0.6],
        [0.1, 0.0, 0.5, 0.6],
        [0.5, 0.5, 0.0, 0.2],
        [0.6, 0.6, 0.2, 0.0],
    ])
    names = ["a", "b", "c", "d"]
    Z = build_upgma_tree(M, names)
    assert Z.shape == (3, 4)
    # First merge should be either {a, b} or {c, d}, both with distance ≤ 0.2.
    first_merge_dist = Z[0, 2]
    assert first_merge_dist <= 0.2 + 1e-9


def test_upgma_rejects_asymmetric_input():
    M = np.array([
        [0.0, 0.1, 0.5],
        [0.2, 0.0, 0.5],  # asymmetric
        [0.5, 0.5, 0.0],
    ])
    with pytest.raises(ValueError):
        build_upgma_tree(M, ["a", "b", "c"])


def test_upgma_rejects_size_mismatch():
    M = np.zeros((3, 3))
    with pytest.raises(ValueError):
        build_upgma_tree(M, ["a", "b"])
