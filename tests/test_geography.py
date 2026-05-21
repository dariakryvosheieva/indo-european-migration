"""Sanity checks for the geography module."""

from __future__ import annotations

import numpy as np
import pytest

from pie_sim.geography import (
    Geography,
    Terrain,
    TERRAIN_PROPS,
    haversine_km,
    stylized_eurasia,
)


# ----- Haversine ---------------------------------------------------------


@pytest.mark.parametrize(
    "lat1, lon1, lat2, lon2, expected_km, tol",
    [
        # Moscow (55.75 N, 37.62 E) → Berlin (52.52 N, 13.41 E) ≈ 1610 km
        (55.75, 37.62, 52.52, 13.41, 1610, 50),
        # London → Paris ≈ 344 km
        (51.51, -0.13, 48.86, 2.35, 344, 20),
        # Same point
        (50.0, 30.0, 50.0, 30.0, 0.0, 0.001),
        # Antipodes-ish
        (0.0, 0.0, 0.0, 180.0, 20015, 50),
    ],
)
def test_haversine_known_distances(lat1, lon1, lat2, lon2, expected_km, tol):
    assert abs(haversine_km(lat1, lon1, lat2, lon2) - expected_km) < tol


# ----- Stylized Eurasia --------------------------------------------------


def test_stylized_shape_and_dtype():
    geo = stylized_eurasia(cell_size=1.0)
    # 64° latitude span, 125° longitude span at 1° resolution.
    assert geo.shape == (64, 125)
    assert geo.terrain.dtype == np.int8


def test_terrain_codes_are_in_range():
    geo = stylized_eurasia(cell_size=1.0)
    valid = {int(t) for t in Terrain}
    assert set(np.unique(geo.terrain).tolist()) <= valid


@pytest.mark.parametrize(
    "place, lat, lon, expected_terrain",
    [
        ("Pontic-Caspian steppe", 50.0, 45.0, Terrain.STEPPE),
        ("Kazakh steppe", 49.0, 70.0, Terrain.STEPPE),
        ("Caucasus", 42.5, 43.5, Terrain.MOUNTAIN),
        ("Himalayas", 30.0, 85.0, Terrain.MOUNTAIN),
        ("Black Sea", 43.0, 35.0, Terrain.OCEAN),
        ("Caspian Sea", 42.0, 51.0, Terrain.OCEAN),
        ("Western Mediterranean", 38.0, 5.0, Terrain.OCEAN),
        ("Atlantic west of Iberia", 40.0, -13.0, Terrain.OCEAN),
        ("Indian subcontinent", 22.0, 78.0, Terrain.TEMPERATE),
        ("Anatolian Pontic Mts", 40.0, 38.0, Terrain.MOUNTAIN),
        ("Iberian peninsula", 40.0, -4.0, Terrain.TEMPERATE),
        ("Central Siberian taiga", 60.0, 80.0, Terrain.FOREST),
        ("Tundra (far north)", 70.0, 80.0, Terrain.TUNDRA),
        ("Taklamakan", 39.0, 82.0, Terrain.DESERT),
        ("Arabian desert", 25.0, 45.0, Terrain.DESERT),
        # Migration corridors
        ("Iranian plateau (Khorasan)", 36.5, 58.0, Terrain.STEPPE),
        ("Bactria / BMAC corridor", 40.0, 65.0, Terrain.STEPPE),
        ("Anatolian Plateau", 38.5, 33.0, Terrain.STEPPE),
        ("Bosphorus / east Thrace", 41.0, 28.0, Terrain.TEMPERATE),
        ("Aral Sea", 45.0, 60.0, Terrain.OCEAN),
        ("Dzungaria steppe", 45.0, 85.0, Terrain.STEPPE),
        ("Elburz Mts (south Caspian)", 36.0, 53.0, Terrain.MOUNTAIN),
        ("Taurus Mts (south Anatolia)", 37.0, 33.0, Terrain.MOUNTAIN),
        # Western European refinements
        ("Bay of Biscay", 47.0, -5.0, Terrain.OCEAN),
        ("Brittany peninsula", 48.5, -3.0, Terrain.TEMPERATE),
        ("English Channel", 50.5, -1.0, Terrain.STRAIT),
        ("Irish Sea", 54.0, -5.0, Terrain.STRAIT),
        ("Britain (south England)", 51.5, -1.0, Terrain.TEMPERATE),
        ("Britain (Scotland)", 56.0, -3.0, Terrain.TEMPERATE),
        ("Ireland (centre)", 53.0, -8.0, Terrain.TEMPERATE),
        # Italian peninsula
        ("Po valley (north Italy)", 45.0, 10.0, Terrain.TEMPERATE),
        ("Apennines (Italian spine)", 42.0, 13.0, Terrain.MOUNTAIN),
        ("Southern Italy (Apulia)", 40.0, 17.0, Terrain.TEMPERATE),
        ("Tyrrhenian Sea", 39.0, 11.0, Terrain.OCEAN),
        ("Adriatic Sea", 43.0, 16.0, Terrain.OCEAN),
        ("Strait of Otranto", 40.5, 19.0, Terrain.STRAIT),
        # Greek peninsula and Balkans
        ("Greek mainland (Thessaly)", 39.5, 23.0, Terrain.TEMPERATE),
        ("Pindus Mts (Greece)", 39.0, 21.0, Terrain.MOUNTAIN),
        ("Aegean Sea", 38.0, 25.0, Terrain.OCEAN),
        ("Dinaric Alps (W Balkans)", 43.0, 19.0, Terrain.MOUNTAIN),
        ("Rhodope Mts (Bulgaria)", 42.5, 25.0, Terrain.MOUNTAIN),
        # Arabian peninsula
        ("Red Sea (mid)", 20.0, 38.0, Terrain.OCEAN),
        ("Bab-el-Mandeb (Red Sea S)", 13.0, 43.0, Terrain.OCEAN),
        ("Strait of Hormuz", 26.0, 57.0, Terrain.OCEAN),
        ("Yemen highlands (Sana'a)", 15.0, 45.0, Terrain.MOUNTAIN),
        ("Yemen lowlands (S Arabia)", 14.0, 48.0, Terrain.DESERT),
        # Indian subcontinent
        ("Arabian Sea", 15.0, 65.0, Terrain.OCEAN),
        ("Western Ghats (S India)", 15.0, 75.0, Terrain.MOUNTAIN),
        ("Deccan Plateau", 17.0, 78.0, Terrain.TEMPERATE),
        ("Thar Desert (Rajasthan)", 27.0, 71.0, Terrain.DESERT),
        ("Indus / Punjab corridor", 30.0, 73.0, Terrain.TEMPERATE),
        ("Bay of Bengal (W edge)", 15.0, 85.0, Terrain.OCEAN),
        # Cosmetic carves: Africa removed, water bodies linked
        ("NE Africa (Egypt / Nile)", 25.0, 30.0, Terrain.OCEAN),
        ("NE Africa (Sudan)", 15.0, 30.0, Terrain.OCEAN),
        ("Gulf of Oman", 24.0, 58.0, Terrain.OCEAN),
        ("Ionian Sea (Adriatic ↔ Med)", 37.0, 18.0, Terrain.OCEAN),
    ],
)
def test_known_locations(place, lat, lon, expected_terrain):
    geo = stylized_eurasia(cell_size=1.0)
    i, j = geo.coord_to_cell(lat, lon)
    actual = Terrain(geo.terrain[i, j])
    assert actual == expected_terrain, f"{place} at ({lat},{lon}): got {actual.name}"


# ----- Friction & capacity -----------------------------------------------


def test_friction_and_capacity_arrays():
    geo = stylized_eurasia(cell_size=1.0)
    fric = geo.friction
    cap = geo.capacity
    assert fric.shape == geo.shape
    assert cap.shape == geo.shape

    ocean = geo.terrain == int(Terrain.OCEAN)
    assert np.all(np.isinf(fric[ocean]))
    assert np.all(cap[ocean] == 0.0)

    land = ~ocean
    assert np.all(np.isfinite(fric[land]))
    assert np.all(cap[land] > 0.0)


def test_terrain_props_complete():
    """Every Terrain enum value has registered properties."""
    for t in Terrain:
        assert int(t) in TERRAIN_PROPS


def test_strait_is_passable_but_high_friction():
    """STRAIT cells must have finite friction and small but positive capacity.

    This is what lets migration cross the English Channel etc. —
    impassable ocean would block Gaelic from ever being founded.
    """
    props = TERRAIN_PROPS[int(Terrain.STRAIT)]
    assert np.isfinite(props.friction)
    # Sea crossing is at least as difficult as the easiest hard terrain.
    assert props.friction >= TERRAIN_PROPS[int(Terrain.DESERT)].friction
    # But strait cells should not be habitable enough to host real
    # populations — they exist as transit pipes / borrowing bridges.
    assert 0.0 < props.capacity <= TERRAIN_PROPS[int(Terrain.MOUNTAIN)].capacity


# ----- Indexing & neighbors ----------------------------------------------


def test_coord_to_cell_roundtrip():
    geo = stylized_eurasia(cell_size=1.0)
    for lat, lon in [(50.5, 45.5), (35.5, -10.5), (71.9, 100.0)]:
        i, j = geo.coord_to_cell(lat, lon)
        clat, clon = geo.cell_center(i, j)
        assert abs(clat - lat) < 1.0
        assert abs(clon - lon) < 1.0


def test_coord_to_cell_rejects_out_of_bounds():
    geo = stylized_eurasia(cell_size=1.0)
    with pytest.raises(ValueError):
        geo.coord_to_cell(80.0, 50.0)  # too far north
    with pytest.raises(ValueError):
        geo.coord_to_cell(50.0, -20.0)  # too far west


def test_neighbors_diagonal_and_orthogonal():
    geo = stylized_eurasia(cell_size=1.0)
    interior = geo.neighbors(20, 50, diagonal=True)
    assert len(interior) == 8
    interior4 = geo.neighbors(20, 50, diagonal=False)
    assert len(interior4) == 4
    corner = geo.neighbors(0, 0, diagonal=True)
    assert len(corner) == 3


def test_neighbors_habitable_only():
    geo = stylized_eurasia(cell_size=1.0)
    # Sit on the Pontic-Caspian shore: some neighbors are Black Sea (ocean).
    i, j = geo.coord_to_cell(46.0, 35.0)
    all_neigh = geo.neighbors(i, j, habitable_only=False)
    land_neigh = geo.neighbors(i, j, habitable_only=True)
    assert len(land_neigh) <= len(all_neigh)
    for ni, nj in land_neigh:
        assert geo.is_habitable(ni, nj)


# ----- Distances ---------------------------------------------------------


def test_great_circle_in_geography():
    geo = stylized_eurasia(cell_size=1.0)
    # 5° of longitude at 50°N is roughly 358 km.
    c1 = geo.coord_to_cell(50.0, 30.0)
    c2 = geo.coord_to_cell(50.0, 35.0)
    d = geo.great_circle_km(c1, c2)
    assert 300 < d < 400


def test_pairwise_distance_matches_pairwise_haversine():
    geo = stylized_eurasia(cell_size=1.0)
    cells = [
        geo.coord_to_cell(50.0, 45.0),    # PC steppe
        geo.coord_to_cell(38.0, 35.0),    # Anatolia
        geo.coord_to_cell(55.0, 10.0),    # northern Europe
    ]
    M = geo.pairwise_distance_km(cells)
    assert M.shape == (3, 3)
    assert np.allclose(np.diag(M), 0.0)
    assert np.allclose(M, M.T)
    # Spot-check one entry against direct haversine.
    expected = geo.great_circle_km(cells[0], cells[1])
    assert abs(M[0, 1] - expected) < 1e-6
