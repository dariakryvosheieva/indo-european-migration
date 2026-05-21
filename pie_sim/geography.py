"""Geography for the Indo-European migration simulation.

A coarse lat/lon grid covering Eurasia. Each cell is annotated with a
:class:`Terrain` type from which two derived quantities follow:

- **friction** — multiplicative cost of moving *into* the cell
  (``np.inf`` for ocean, ``1`` for steppe, ``5`` for high mountain, etc.).
- **capacity** — relative carrying capacity for human population (used
  later by the demographic module).

The :func:`stylized_eurasia` constructor returns a hand-crafted approximation
of the major terrain regions relevant to Indo-European migrations: the
Pontic-Caspian steppe, Anatolian highlands, European forest belt, Iranian
plateau, Himalayas, etc. The shapes are crude rectangles — designed for
transparency and reproducibility, not cartographic accuracy.

Distances are computed with the haversine formula on a sphere of radius
6371 km. The grid is regular in degrees; if you need equal-area cells,
weight by ``cos(lat)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

EARTH_RADIUS_KM = 6371.0


class Terrain(IntEnum):
    """Coarse terrain classes used throughout the simulation.

    ``STRAIT`` is a narrow water body (English Channel, Irish Sea, Strait
    of Otranto) — high but *finite* friction, so migration across is
    slow but possible (capturing the well-attested fact that Bronze and
    Iron Age peoples crossed these waters in boats). Ocean proper has
    infinite friction and is impassable.
    """

    OCEAN = 0
    TEMPERATE = 1
    STEPPE = 2
    FOREST = 3
    DESERT = 4
    MOUNTAIN = 5
    TUNDRA = 6
    STRAIT = 7


@dataclass(frozen=True)
class TerrainProps:
    friction: float
    capacity: float
    color: str
    label: str


TERRAIN_PROPS: dict[int, TerrainProps] = {
    int(Terrain.OCEAN):     TerrainProps(np.inf, 0.00, "#9EC9E5", "ocean"),
    int(Terrain.TEMPERATE): TerrainProps(1.5,    1.00, "#A8C775", "temperate"),
    int(Terrain.STEPPE):    TerrainProps(1.0,    0.50, "#D4CC83", "steppe"),
    int(Terrain.FOREST):    TerrainProps(2.0,    0.70, "#3E7548", "forest"),
    int(Terrain.DESERT):    TerrainProps(3.0,    0.05, "#E0C786", "desert"),
    int(Terrain.MOUNTAIN):  TerrainProps(5.0,    0.10, "#888888", "mountain"),
    int(Terrain.TUNDRA):    TerrainProps(2.5,    0.10, "#D5E5E0", "tundra"),
    # Narrow strait: a sea easily crossed by boat in calm season
    # (English Channel, Strait of Otranto). Friction matches desert
    # — a real but moderate barrier — and capacity matches mountain,
    # so strait cells carry enough population to act as a meaningful
    # borrowing channel between the two coasts they connect (e.g.,
    # Britain ↔ continental Celtic, Italy ↔ Balkans). They are not
    # quite habitable in their own right, but neither are they the
    # near-impassable barriers that a higher-friction setting makes
    # them.
    int(Terrain.STRAIT):    TerrainProps(3.0,    0.10, "#7DB6D9", "strait"),
}


def _build_lookups() -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables indexed by terrain code for vectorized queries."""
    n = max(TERRAIN_PROPS.keys()) + 1
    friction = np.zeros(n, dtype=np.float64)
    capacity = np.zeros(n, dtype=np.float64)
    for code, props in TERRAIN_PROPS.items():
        friction[code] = props.friction
        capacity[code] = props.capacity
    return friction, capacity


_FRICTION_LOOKUP, _CAPACITY_LOOKUP = _build_lookups()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres on a sphere of radius 6371 km."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


@dataclass
class Geography:
    """A regular lat/lon grid of terrain cells.

    Cell ``(i, j)`` covers latitudes ``[lat_min + i*cell_size,
    lat_min + (i+1)*cell_size]`` and longitudes ``[lon_min + j*cell_size,
    lon_min + (j+1)*cell_size]``. Index ``i`` increases from south to north,
    ``j`` from west to east; this makes ``imshow(..., origin='lower')``
    display the map right-side-up.

    Attributes:
        lat_min: Southern edge of the grid (degrees).
        lat_max: Northern edge of the grid (degrees).
        lon_min: Western edge of the grid (degrees).
        lon_max: Eastern edge of the grid (degrees).
        cell_size: Cell side length in degrees.
        terrain: ``(n_lat, n_lon)`` int array of :class:`Terrain` codes.
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    cell_size: float
    terrain: np.ndarray

    @property
    def n_lat(self) -> int:
        return int(self.terrain.shape[0])

    @property
    def n_lon(self) -> int:
        return int(self.terrain.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.terrain.shape)  # type: ignore[return-value]

    @property
    def friction(self) -> np.ndarray:
        """Per-cell friction array (vectorized lookup)."""
        return _FRICTION_LOOKUP[self.terrain]

    @property
    def capacity(self) -> np.ndarray:
        """Per-cell carrying capacity array (vectorized lookup)."""
        return _CAPACITY_LOOKUP[self.terrain]

    @property
    def land_mask(self) -> np.ndarray:
        """Boolean mask of habitable cells (everything except ocean)."""
        return self.terrain != int(Terrain.OCEAN)

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        """Return ``(lat, lon)`` of the centre of cell ``(i, j)``."""
        lat = self.lat_min + (i + 0.5) * self.cell_size
        lon = self.lon_min + (j + 0.5) * self.cell_size
        return lat, lon

    def coord_to_cell(self, lat: float, lon: float) -> tuple[int, int]:
        """Convert geographic coordinates to grid indices.

        Raises:
            ValueError: if the point falls outside the grid.
        """
        if not (self.lat_min <= lat < self.lat_max):
            raise ValueError(f"lat {lat} outside [{self.lat_min}, {self.lat_max})")
        if not (self.lon_min <= lon < self.lon_max):
            raise ValueError(f"lon {lon} outside [{self.lon_min}, {self.lon_max})")
        i = int((lat - self.lat_min) / self.cell_size)
        j = int((lon - self.lon_min) / self.cell_size)
        return i, j

    def is_habitable(self, i: int, j: int) -> bool:
        return self.terrain[i, j] != int(Terrain.OCEAN)

    def neighbors(
        self,
        i: int,
        j: int,
        *,
        diagonal: bool = True,
        habitable_only: bool = False,
    ) -> list[tuple[int, int]]:
        """Return the in-bounds 4- or 8-neighbours of cell ``(i, j)``."""
        offsets: Iterable[tuple[int, int]]
        if diagonal:
            offsets = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]
        else:
            offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        out: list[tuple[int, int]] = []
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.n_lat and 0 <= nj < self.n_lon:
                if habitable_only and not self.is_habitable(ni, nj):
                    continue
                out.append((ni, nj))
        return out

    def great_circle_km(self, c1: tuple[int, int], c2: tuple[int, int]) -> float:
        """Great-circle distance between the centres of two cells, in km."""
        lat1, lon1 = self.cell_center(*c1)
        lat2, lon2 = self.cell_center(*c2)
        return haversine_km(lat1, lon1, lat2, lon2)

    def pairwise_distance_km(self, cells: list[tuple[int, int]]) -> np.ndarray:
        """``(n, n)`` matrix of pairwise great-circle distances between cells."""
        n = len(cells)
        lats = np.empty(n)
        lons = np.empty(n)
        for k, (i, j) in enumerate(cells):
            lats[k], lons[k] = self.cell_center(i, j)
        phi = np.radians(lats)
        lam = np.radians(lons)
        dphi = phi[:, None] - phi[None, :]
        dlam = lam[:, None] - lam[None, :]
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2.0) ** 2
        return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    def plot(
        self,
        ax: plt.Axes | None = None,
        *,
        show_legend: bool = True,
        gridlines: bool = False,
    ) -> plt.Axes:
        """Render the terrain map on ``ax`` (or a new figure if None)."""
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 6))

        rgb = np.zeros((self.n_lat, self.n_lon, 3), dtype=np.float32)
        for code, props in TERRAIN_PROPS.items():
            mask = self.terrain == code
            rgb[mask] = mcolors.to_rgb(props.color)

        extent = (self.lon_min, self.lon_max, self.lat_min, self.lat_max)
        ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")

        if gridlines:
            for lat in range(int(self.lat_min), int(self.lat_max) + 1, 10):
                ax.axhline(lat, color="white", lw=0.3, alpha=0.5)
            for lon in range(int(self.lon_min), int(self.lon_max) + 1, 10):
                ax.axvline(lon, color="white", lw=0.3, alpha=0.5)

        if show_legend:
            from matplotlib.patches import Patch
            handles = [
                Patch(facecolor=props.color, edgecolor="black", lw=0.4, label=props.label)
                for props in TERRAIN_PROPS.values()
            ]
            ax.legend(handles=handles, loc="lower left", ncol=4, fontsize=8,
                      framealpha=0.9)

        ax.set_xlabel("longitude (°E)")
        ax.set_ylabel("latitude (°N)")
        ax.set_aspect("equal")
        return ax


def stylized_eurasia(cell_size: float = 1.0) -> Geography:
    """Anatomically-improved approximation of Eurasian geography.

    Coverage: 8-72 °N, 15 °W to 110 °E. Designed to encode the terrain
    features relevant to Indo-European migrations as accurately as
    axis-aligned rectangles allow:

    Migration corridors (steppe / passable):
    - Pontic-Caspian + Kazakh steppes (the Yamnaya/Andronovo highway).
    - Pannonian plain (steppe arm into Europe).
    - Iranian plateau north of the central desert (Elburz fringe).
    - Bactria / Margiana / Sogdiana steppe (the BMAC route Indo-Iranians took).
    - Dzungaria steppe (Andronovo → Tarim, distinct from BMAC).
    - Anatolian plateau (semi-steppe, between Pontic and Taurus mountains).
    - Bosphorus / east Thracian land bridge (Greek-Anatolian connection).

    Narrow straits (slow but passable — see :class:`Terrain.STRAIT`):
    - English Channel (Britain - France), Irish Sea (Ireland - Britain),
      Strait of Otranto (Italy heel - Albania). These let Celtic and
      Italic populations diffuse across narrow seas in our model rather
      than being blocked entirely by impassable ocean.

    Mountain barriers:
    - Caucasus, Alps, Carpathians, Pyrenees, Apennines, Urals.
    - Himalayas, Hindu Kush, Pamir Knot, Tien Shan, Kunlun.
    - Zagros, Elburz / Kopet Dag (separating Iranian plateau from Caspian).
    - Pontic Mts (north Anatolia), Taurus Mts (south Anatolia).
    - Pindus (Greek peninsula), Dinaric Alps + Balkan / Rhodope (Balkans).
    - Yemen highlands (SW Arabia), Western Ghats (Indian west coast).

    Water bodies:
    - Mediterranean, Tyrrhenian, Adriatic + Ionian (linking Adriatic
      to the Mediterranean), Aegean, Marmara.
    - Black, Caspian, Aral, Persian Gulf + Strait of Hormuz + Gulf of
      Oman (linking the Persian Gulf to the Arabian Sea), Red Sea
      (extended to Bab-el-Mandeb), Arabian Sea, Bay of Bengal.
    - Atlantic, Bay of Biscay, North Sea, Baltic, Arctic.

    Africa is intentionally not modeled — NE Africa (Egypt, Sudan,
    Sinai) is carved away as ocean so the map shows only the Eurasian
    landmass relevant to Indo-European migration.

    Peninsulas with explicit coastlines:
    - Iberia (Bay of Biscay separates from France; Pyrenees connect).
    - Italy (narrow boot; Adriatic + Tyrrhenian seas).
    - Greek peninsula (Pindus mountains + Aegean).
    - Britain + Ireland (disconnected from continent by the STRAIT
      cells of the English Channel and Irish Sea).
    - Arabian peninsula (Red Sea + Persian Gulf + Arabian Sea
      coastlines; Yemen highlands in the SW corner).
    - Indian peninsula (Arabian Sea + Bay of Bengal coastlines;
      Western Ghats along the W coast; Thar Desert at NW corner
      channels migration north into the Punjab corridor).

    Deserts:
    - Iranian central, Karakum (south), Kyzylkum, Taklamakan, Gobi.
    - Arabian (incl. Yemen lowlands), Syrian, Thar (NW India).

    Painting order is OCEAN → land defaults → broad steppes → forests →
    tundra → deserts → narrow steppe corridors (override deserts) →
    mountains (override everything) → water carves → narrow strait carves
    → land restorations (with re-applied mountains where carves erased
    them).

    Args:
        cell_size: Side length of grid cells in degrees. Default 1°
            (~80-100 km at IE-relevant latitudes).
    """
    LAT_MIN, LAT_MAX = 8.0, 72.0
    LON_MIN, LON_MAX = -15.0, 110.0

    n_lat = int(round((LAT_MAX - LAT_MIN) / cell_size))
    n_lon = int(round((LON_MAX - LON_MIN) / cell_size))
    terrain = np.full((n_lat, n_lon), int(Terrain.OCEAN), dtype=np.int8)

    def paint(lat_lo: float, lat_hi: float, lon_lo: float, lon_hi: float, t: Terrain) -> None:
        # Clamp endpoints to [0, n_*] to avoid negative-index wrap-around
        # when the rectangle extends beyond the grid bounds.
        i0 = int(round((lat_lo - LAT_MIN) / cell_size))
        i1 = int(round((lat_hi - LAT_MIN) / cell_size))
        j0 = int(round((lon_lo - LON_MIN) / cell_size))
        j1 = int(round((lon_hi - LON_MIN) / cell_size))
        i0 = max(0, min(n_lat, i0))
        i1 = max(0, min(n_lat, i1))
        j0 = max(0, min(n_lon, j0))
        j1 = max(0, min(n_lon, j1))
        if i0 < i1 and j0 < j1:
            terrain[i0:i1, j0:j1] = int(t)

    # 1. Default landmass.
    paint(35, 72, -10, 110, Terrain.TEMPERATE)
    paint(8, 35, 60, 95, Terrain.TEMPERATE)      # subcontinent and Iranian plateau
    paint(12, 35, 30, 60, Terrain.TEMPERATE)     # Levant + Arabia footprint (south to Yemen)

    # 2. Broad steppes (the well-known migration highways).
    paint(45, 53, 25, 58, Terrain.STEPPE)        # Pontic-Caspian
    paint(45, 53, 58, 85, Terrain.STEPPE)        # Kazakh steppe
    paint(45, 49, 16, 22, Terrain.STEPPE)        # Pannonian plain

    # 3. Forest belts.
    paint(48, 60, -5, 35, Terrain.FOREST)        # Central European broadleaf
    paint(53, 65, 35, 100, Terrain.FOREST)       # Russian taiga (south)

    # 4. Tundra.
    paint(65, 72, -10, 110, Terrain.TUNDRA)

    # 5. Deserts.
    paint(36, 39, 53, 65, Terrain.DESERT)        # Karakum (southern core)
    paint(40, 44, 60, 67, Terrain.DESERT)        # Kyzylkum
    paint(37, 42, 75, 95, Terrain.DESERT)        # Taklamakan
    paint(40, 45, 95, 110, Terrain.DESERT)       # Gobi (western edge)
    paint(28, 35, 50, 65, Terrain.DESERT)        # Iranian central desert
    paint(12, 30, 32, 56, Terrain.DESERT)        # Arabia (extended south to Yemen lowlands)
    paint(32, 35, 36, 44, Terrain.DESERT)        # Syrian desert
    paint(24, 30, 68, 75, Terrain.DESERT)        # Thar Desert (NW India / Pakistan)

    # 6. Narrow steppe corridors — these override deserts where they overlap.
    #    These are the actual IE migration routes on real geography.
    paint(35, 39, 48, 60, Terrain.STEPPE)        # Iranian plateau (Elburz fringe / Khorasan)
    paint(38, 43, 55, 75, Terrain.STEPPE)        # Bactria / Margiana / Sogdiana (BMAC corridor)
    paint(43, 47, 80, 90, Terrain.STEPPE)        # Dzungaria (north of Tien Shan, gateway to Tarim)

    # 7. Mountains (last over land, so they override deserts and steppes).
    paint(40, 44, 38, 47, Terrain.MOUNTAIN)      # Caucasus (narrowed east)
    paint(43, 47, 5, 16, Terrain.MOUNTAIN)       # Alps
    paint(45, 49, 22, 27, Terrain.MOUNTAIN)      # Carpathians
    paint(42, 44, -2, 3, Terrain.MOUNTAIN)       # Pyrenees
    paint(38, 46, 12, 15, Terrain.MOUNTAIN)      # Apennines (Italian spine, east of Tyrrhenian coast)
    paint(50, 68, 56, 65, Terrain.MOUNTAIN)      # Urals (extended south)
    paint(28, 36, 76, 95, Terrain.MOUNTAIN)      # Himalayas (main range, east of Punjab)
    paint(33, 37, 74, 77, Terrain.MOUNTAIN)      # Karakoram (Kashmir spur)
    paint(34, 38, 67, 73, Terrain.MOUNTAIN)      # Hindu Kush / Pamir (narrowed)
    paint(40, 43, 70, 78, Terrain.MOUNTAIN)      # Tien Shan (shifted west, leaves Dzungar gap)
    paint(35, 38, 76, 92, Terrain.MOUNTAIN)      # Kunlun (north Tibet edge)
    paint(30, 37, 45, 50, Terrain.MOUNTAIN)      # Zagros
    paint(35, 38, 50, 56, Terrain.MOUNTAIN)      # Elburz / Kopet Dag (south Caspian)
    paint(39, 41, 32, 42, Terrain.MOUNTAIN)      # Pontic Mts (north Anatolia)
    paint(36, 38, 28, 44, Terrain.MOUNTAIN)      # Taurus Mts (south Anatolia)
    paint(13, 18, 44, 47, Terrain.MOUNTAIN)      # Yemen highlands (Sana'a)
    paint(10, 21, 73, 76, Terrain.MOUNTAIN)      # Western Ghats (W India coastal range)

    # 8. Water bodies — carve from land.
    paint(30, 36, -5, 36, Terrain.OCEAN)         # Mediterranean south
    paint(36, 41, -5, 7, Terrain.OCEAN)          # Western Mediterranean
    paint(36, 40, 8, 13, Terrain.OCEAN)          # Tyrrhenian Sea (south + west of Italy)
    paint(40, 46, 13, 19, Terrain.OCEAN)         # Adriatic Sea (between Italy and Balkans)
    paint(36, 40, 18, 20, Terrain.OCEAN)         # Ionian Sea (links Adriatic ↔ Mediterranean)
    paint(36, 41, 23, 27, Terrain.OCEAN)         # Aegean Sea (Greece - Anatolia)
    paint(40, 41, 26, 30, Terrain.OCEAN)         # Sea of Marmara
    paint(41, 46, 28, 42, Terrain.OCEAN)         # Black Sea
    paint(37, 47, 47, 54, Terrain.OCEAN)         # Caspian Sea
    paint(44, 46, 58, 62, Terrain.OCEAN)         # Aral Sea
    paint(24, 30, 48, 58, Terrain.OCEAN)         # Persian Gulf + Strait of Hormuz
    paint(22, 27, 57, 60, Terrain.OCEAN)         # Gulf of Oman (links Persian Gulf ↔ Arabian Sea)
    paint(12, 28, 33, 44, Terrain.OCEAN)         # Red Sea (extended south to Bab-el-Mandeb)
    paint(8, 22, 55, 73, Terrain.OCEAN)          # Arabian Sea (between Arabia and India)
    paint(8, 22, 80, 95, Terrain.OCEAN)          # Bay of Bengal (west to Chennai meridian)
    paint(8, 30, 25, 34, Terrain.OCEAN)          # NE Africa carved away (we model Eurasia only)
    paint(8, 72, -15, -10, Terrain.OCEAN)        # Atlantic west edge (extended south)
    paint(44, 50, -10, -2, Terrain.OCEAN)        # Bay of Biscay (NEW: separates Iberia from France)
    paint(53, 60, 1, 8, Terrain.OCEAN)           # North Sea (lon 1+ to spare Britain east coast)
    paint(55, 65, 14, 30, Terrain.OCEAN)         # Baltic Sea

    # 8b. Narrow straits — passable but high friction (see Terrain.STRAIT).
    paint(50, 51, -5, 2, Terrain.STRAIT)         # English Channel
    paint(52, 56, -6, -4, Terrain.STRAIT)        # Irish Sea + North Channel
    paint(40, 41, 18, 20, Terrain.STRAIT)        # Strait of Otranto (Italy heel - Albania)

    # 9. Restore peninsulas and add corridors carved by step 8.
    paint(36, 44, -10, 0, Terrain.TEMPERATE)     # Iberia (peninsula proper)
    paint(42, 44, -2, 3, Terrain.MOUNTAIN)       # Pyrenees (re-apply)
    paint(48, 50, -5, -2, Terrain.TEMPERATE)     # Brittany (re-establish W French coast)

    # Italy — narrow peninsular boot rather than wide rectangle.
    paint(43, 46, 7, 14, Terrain.TEMPERATE)      # Po valley + N Italy
    paint(40, 43, 9, 14, Terrain.TEMPERATE)      # Central Italy
    paint(38, 41, 14, 19, Terrain.TEMPERATE)     # Southern Italy (boot, east of Apennines)
    paint(38, 46, 12, 15, Terrain.MOUNTAIN)      # Apennines (re-apply, east of Tyrrhenian)

    # Greek peninsula and Balkans.
    # Balkans start at lon 19 (east of the Adriatic) so the Adriatic
    # carve at lon 13-19 is not overwritten by the restoration.
    paint(36, 41, 20, 24, Terrain.TEMPERATE)     # Greek mainland
    paint(37, 41, 20, 22, Terrain.MOUNTAIN)      # Pindus range (Greek backbone)
    paint(41, 46, 19, 30, Terrain.TEMPERATE)     # Balkan landmass (east of Adriatic)
    paint(42, 45, 19, 22, Terrain.MOUNTAIN)      # Dinaric Alps (W Balkans)
    paint(42, 44, 22, 27, Terrain.MOUNTAIN)      # Balkan Mts + Rhodope (E Balkans)

    # British Isles — islands separated by STRAIT cells, not continuous land.
    paint(51, 58, -4, 2, Terrain.TEMPERATE)      # Britain (south coast at lat 51)
    paint(52, 55, -10, -6, Terrain.TEMPERATE)    # Ireland

    paint(58, 71, 4, 11, Terrain.TEMPERATE)      # Norway
    paint(55, 71, 11, 32, Terrain.FOREST)        # Sweden / Finland

    paint(40, 42, 27, 30, Terrain.TEMPERATE)     # Bosphorus / east Thrace bridge
    paint(36, 42, 28, 45, Terrain.TEMPERATE)     # Anatolia base
    paint(38, 40, 30, 37, Terrain.STEPPE)        # central Anatolian Plateau (steppe)
    paint(39, 41, 32, 42, Terrain.MOUNTAIN)      # Pontic Mts (re-apply over Anatolia base)
    paint(36, 38, 28, 44, Terrain.MOUNTAIN)      # Taurus Mts (re-apply)

    return Geography(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, cell_size, terrain)


__all__ = [
    "EARTH_RADIUS_KM",
    "Terrain",
    "TerrainProps",
    "TERRAIN_PROPS",
    "Geography",
    "haversine_km",
    "stylized_eurasia",
]
