"""Sanity checks for the demographic / migration simulator."""

from __future__ import annotations

import numpy as np
import pytest

from pie_sim.cognates import CognateEvolver
from pie_sim.geography import Geography, Terrain, stylized_eurasia
from pie_sim.population import MigrationSimulator, Population


# ----- Helpers -----------------------------------------------------------


def _flat_geo(n: int = 10, terrain: Terrain = Terrain.STEPPE) -> Geography:
    """Tiny uniform-terrain geography for unit tests."""
    return Geography(
        lat_min=40.0, lat_max=40.0 + n, lon_min=20.0, lon_max=20.0 + n,
        cell_size=1.0,
        terrain=np.full((n, n), int(terrain), dtype=np.int8),
    )


# ----- Population container ----------------------------------------------


def test_empty_population():
    geo = _flat_geo(5)
    pop = Population.empty(geo)
    assert pop.total_population == 0
    assert pop.n_populated_cells == 0
    assert pop.lexicons == {}


def test_seeding_a_population():
    geo = _flat_geo(5)
    pop = Population.empty(geo)
    ev = CognateEvolver(n_meanings=20, rng=np.random.default_rng(0))
    lex = ev.initial_lexicon()
    i, j = pop.seed(lat=42.5, lon=22.5, count=0.5, lexicon=lex)
    assert pop.counts[i, j] == 0.5
    assert (i, j) in pop.lexicons
    assert pop.n_populated_cells == 1


def test_seed_rejects_ocean():
    n = 5
    terrain = np.full((n, n), int(Terrain.OCEAN), dtype=np.int8)
    geo = Geography(40.0, 45.0, 20.0, 25.0, 1.0, terrain)
    pop = Population.empty(geo)
    ev = CognateEvolver(n_meanings=10)
    with pytest.raises(ValueError):
        pop.seed(lat=42.5, lon=22.5, count=1.0, lexicon=ev.initial_lexicon())


# ----- Demography in isolation -------------------------------------------


def test_logistic_growth_in_isolation():
    """A single seeded cell on an island grows toward K_max."""
    n = 5
    # Island: one habitable cell surrounded by ocean.
    terrain = np.full((n, n), int(Terrain.OCEAN), dtype=np.int8)
    terrain[2, 2] = int(Terrain.STEPPE)
    geo = Geography(40.0, 45.0, 20.0, 25.0, 1.0, terrain)
    ev = CognateEvolver(n_meanings=10, rng=np.random.default_rng(0))
    sim = MigrationSimulator(geo, ev, growth_rate=0.01, migration_rate=0.0)
    pop = Population.empty(geo)
    pop.counts[2, 2] = 0.01
    pop.lexicons[(2, 2)] = ev.initial_lexicon()
    snaps = sim.run(pop, n_years=2000, dt=10.0,
                    snapshot_years=[0, 500, 2000])
    counts = [s.counts[2, 2] for _, s in snaps]
    assert counts[0] < counts[1] < counts[2]
    # K = K_max * capacity_steppe = 1.0 * 0.5 = 0.5 in default props.
    assert counts[-1] == pytest.approx(sim.K[2, 2], rel=0.05)


def test_migration_spreads_population():
    """A seed in the middle of a uniform plain spreads outward over time."""
    geo = _flat_geo(21, Terrain.STEPPE)
    ev = CognateEvolver(n_meanings=20, rng=np.random.default_rng(0))
    sim = MigrationSimulator(
        geo, ev,
        growth_rate=0.01,
        migration_rate=0.05,
        founding_threshold=0.05,
    )
    pop = Population.empty(geo)
    pop.counts[10, 10] = 0.5
    pop.lexicons[(10, 10)] = ev.initial_lexicon()

    snaps = sim.run(pop, n_years=600, dt=10.0, snapshot_years=[0, 200, 600])
    initial, mid, final = (s for _, s in snaps)
    assert initial.n_populated_cells == 1
    assert mid.n_populated_cells > 1
    assert final.n_populated_cells > mid.n_populated_cells
    # Spread should remain bounded — not the entire grid yet at year 600.
    assert final.n_populated_cells < 21 * 21


def test_migration_blocked_by_ocean():
    """An island seed cannot migrate to surrounding ocean."""
    n = 7
    terrain = np.full((n, n), int(Terrain.OCEAN), dtype=np.int8)
    terrain[3, 3] = int(Terrain.STEPPE)
    geo = Geography(40.0, 47.0, 20.0, 27.0, 1.0, terrain)
    ev = CognateEvolver(n_meanings=10, rng=np.random.default_rng(0))
    sim = MigrationSimulator(geo, ev, migration_rate=0.5)
    pop = Population.empty(geo)
    pop.counts[3, 3] = 0.5
    pop.lexicons[(3, 3)] = ev.initial_lexicon()

    snaps = sim.run(pop, n_years=200, dt=10.0)
    final = snaps[-1][1]
    assert final.n_populated_cells == 1


def test_mountains_slow_migration():
    """Spread east is slower with a mountain band than with all-steppe."""
    n = 25

    def grid_with_band(band_terrain: Terrain) -> Geography:
        t = np.full((n, n), int(Terrain.STEPPE), dtype=np.int8)
        t[:, 12] = int(band_terrain)  # single-column vertical band
        return Geography(40.0, 40.0 + n, 20.0, 20.0 + n, 1.0, t)

    def east_population(geo: Geography) -> float:
        ev = CognateEvolver(n_meanings=10, rng=np.random.default_rng(0))
        sim = MigrationSimulator(
            geo, ev, growth_rate=0.01, migration_rate=0.1,
            founding_threshold=0.05,
        )
        pop = Population.empty(geo)
        pop.counts[n // 2, 0] = 0.5
        pop.lexicons[(n // 2, 0)] = ev.initial_lexicon()
        snap = sim.run(pop, n_years=800, dt=10.0)[-1][1]
        return float(snap.counts[:, 13:].sum())

    plain = east_population(grid_with_band(Terrain.STEPPE))
    mountains = east_population(grid_with_band(Terrain.MOUNTAIN))
    assert mountains < plain, (
        f"mountain-band run accumulated {mountains:.3f} east of barrier vs "
        f"{plain:.3f} for plain"
    )


# ----- Founding events propagate lexicons --------------------------------


def test_founding_inherits_source_lexicon():
    """A newly-settled cell starts with a copy of its source's cognate IDs."""
    geo = _flat_geo(5, Terrain.STEPPE)
    ev = CognateEvolver(
        n_meanings=20,
        innovation_rate=0.0,    # no drift, so we can compare exactly
        borrowing_rate=0.0,
        rng=np.random.default_rng(0),
    )
    sim = MigrationSimulator(
        geo, ev, growth_rate=0.05, migration_rate=0.2,
        founding_threshold=0.001,
    )
    pop = Population.empty(geo)
    seed_lex = ev.initial_lexicon()
    pop.counts[2, 2] = 0.5
    pop.lexicons[(2, 2)] = seed_lex.copy()
    snap = sim.run(pop, n_years=200, dt=10.0)[-1][1]
    # Every populated cell should share the seed's IDs because lambda=0.
    seed_ids = seed_lex.cognate_ids
    for cell, lex in snap.lexicons.items():
        assert np.array_equal(lex.cognate_ids, seed_ids), f"{cell} diverged with lambda=0"


def test_two_seeds_meet_and_remain_distinct():
    """Two different lexicons spreading toward each other don't fuse."""
    geo = _flat_geo(15, Terrain.STEPPE)
    ev = CognateEvolver(n_meanings=30, innovation_rate=0.0,
                        borrowing_rate=0.0, rng=np.random.default_rng(0))
    sim = MigrationSimulator(
        geo, ev,
        growth_rate=0.02, migration_rate=0.1,
        founding_threshold=0.001,
    )
    lex_a = ev.initial_lexicon()
    lex_b = ev.initial_lexicon()
    assert set(lex_a.cognate_ids.tolist()).isdisjoint(lex_b.cognate_ids.tolist())

    pop = Population.empty(geo)
    pop.counts[7, 1] = 0.5
    pop.lexicons[(7, 1)] = lex_a.copy()
    pop.counts[7, 13] = 0.5
    pop.lexicons[(7, 13)] = lex_b.copy()

    snap = sim.run(pop, n_years=500, dt=10.0)[-1][1]
    # Some cells should still match each seed exactly (no innovation, no borrowing)
    a_ids = set(lex_a.cognate_ids.tolist())
    b_ids = set(lex_b.cognate_ids.tolist())
    saw_a = saw_b = False
    for lex in snap.lexicons.values():
        cell_ids = set(lex.cognate_ids.tolist())
        if cell_ids == a_ids:
            saw_a = True
        elif cell_ids == b_ids:
            saw_b = True
    assert saw_a, "A descendants vanished"
    assert saw_b, "B descendants vanished"


# ----- Pinned isolated breakaways ----------------------------------------


def test_isolated_cell_does_not_migrate_or_transmit():
    """A pinned isolated cell stays put, accumulates no inflow, and is
    never offered as a contact source to its neighbours.
    """
    n = 9
    geo = _flat_geo(n, Terrain.STEPPE)
    # Use unique IDs and zero drift/contact so we can test by cognate ID
    # equality rather than approximate distance.
    ev = CognateEvolver(
        n_meanings=20, innovation_rate=0.0, borrowing_rate=0.0,
        rng=np.random.default_rng(0),
    )
    isolated = (4, 4)
    sim = MigrationSimulator(
        geo, ev,
        growth_rate=0.02, migration_rate=0.2,
        founding_threshold=0.01,
        isolated_cells={isolated},
    )

    pop = Population.empty(geo)
    homeland_lex = ev.initial_lexicon()
    pop.counts[0, 0] = 0.5
    pop.lexicons[(0, 0)] = homeland_lex.copy()

    iso_lex = ev.initial_lexicon()
    pop.counts[isolated] = 0.5 * sim.K[isolated]
    pop.lexicons[isolated] = iso_lex.copy()

    initial_iso_count = pop.counts[isolated]
    snap = sim.run(pop, n_years=500, dt=10.0)[-1][1]

    # The isolated cell never received migrants from elsewhere (its
    # count only grew via logistic growth toward K, capped there).
    assert snap.counts[isolated] <= sim.K[isolated] + 1e-9
    # The isolated cell still holds the breakaway lexicon (zero drift).
    assert np.array_equal(snap.lexicons[isolated].cognate_ids,
                          iso_lex.cognate_ids)
    # No other cell shares the isolated cell's IDs — it never transmitted.
    iso_ids = set(iso_lex.cognate_ids.tolist())
    home_ids = set(homeland_lex.cognate_ids.tolist())
    assert iso_ids.isdisjoint(home_ids)
    contaminated = [
        cell for cell, lex in snap.lexicons.items()
        if cell != isolated
        and any(cid in iso_ids for cid in lex.cognate_ids.tolist())
    ]
    assert not contaminated, (
        f"isolated cell leaked IDs to {contaminated}"
    )
    # Sanity check: the homeland did spread normally.
    assert snap.n_populated_cells > 1


def test_isolated_cell_blocks_inflow_from_neighbour_wave():
    """Even a high-rate migration wave reaching the isolated cell's
    neighbours never founds the isolated cell from the wave: the cell
    keeps the lexicon it was seeded with.
    """
    geo = _flat_geo(7, Terrain.STEPPE)
    ev = CognateEvolver(
        n_meanings=15, innovation_rate=0.0, borrowing_rate=0.0,
        rng=np.random.default_rng(0),
    )
    isolated = (3, 3)
    sim = MigrationSimulator(
        geo, ev,
        growth_rate=0.05, migration_rate=0.5,
        founding_threshold=0.001,
        isolated_cells={isolated},
    )

    pop = Population.empty(geo)
    home_lex = ev.initial_lexicon()
    iso_lex = ev.initial_lexicon()
    pop.counts[0, 0] = 0.5
    pop.lexicons[(0, 0)] = home_lex.copy()
    pop.counts[isolated] = 0.5 * sim.K[isolated]
    pop.lexicons[isolated] = iso_lex.copy()

    snap = sim.run(pop, n_years=400, dt=10.0)[-1][1]

    # The wave from (0,0) saturates the grid.
    saturated = sum(1 for v in snap.counts.flatten() if v > 0.4)
    assert saturated > 5
    # But (3,3) still holds the *isolated* lexicon, not the homeland's.
    assert np.array_equal(
        snap.lexicons[isolated].cognate_ids,
        iso_lex.cognate_ids,
    )


# ----- Smoke test against the real geography -----------------------------


def test_steppe_seed_runs_without_error():
    """The realistic Eurasia geography accepts a Steppe seed and runs briefly."""
    geo = stylized_eurasia(cell_size=2.0)  # coarse for speed
    ev = CognateEvolver(n_meanings=50, rng=np.random.default_rng(0))
    sim = MigrationSimulator(geo, ev, founding_threshold=0.01)
    pop = Population.empty(geo)
    pop.seed(lat=50.0, lon=45.0, count=0.5, lexicon=ev.initial_lexicon())
    snap = sim.run(pop, n_years=500, dt=25.0)[-1][1]
    assert snap.n_populated_cells > 1
