"""Population dynamics, migration, and language transmission on the geography.

A metapopulation model: each grid cell has a scalar population count and,
if populated, a :class:`Lexicon` that evolves under stochastic-Dollo
dynamics with borrowing from populated geographic neighbors. Populations
grow logistically toward terrain-dependent carrying capacity and disperse
to neighboring cells with weight ``1 / friction``.

Design choices for the MVP:

- A cell's population is a single scalar (not multiple coexisting lineages).
- A cell carries at most one lexicon at a time. When migrants arrive in
  an empty habitable cell and push it past ``founding_threshold``, the
  cell inherits a copy of the lexicon of its *dominant* migrant source.
- Migration into already-populated cells contributes mass but does **not**
  switch the lexicon. Linguistic influence at populated cells flows
  through the borrowing channel.
- Borrowing intensity scales with neighbor population × spatial proximity:
  ``w_k = (N_k / K_max) * exp(-d_k / contact_sigma_km)``.

This produces the demic-diffusion + founder-effect dynamics that drive
Indo-European divergence under our framing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pie_sim.cognates import CognateEvolver, Lexicon
from pie_sim.geography import Geography


# 8-neighbour offsets (cardinal + diagonal); fixed order shared throughout.
NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)


@dataclass
class Population:
    """State of the metapopulation across the geography.

    Attributes:
        geo: Underlying terrain grid.
        counts: ``(n_lat, n_lon)`` float array of population counts in
            relative units (``0`` = empty, ``K_max`` = full carrying capacity).
        lexicons: Mapping from cell index ``(i, j)`` to its current lexicon.
            Only populated cells appear here.
    """

    geo: Geography
    counts: np.ndarray
    lexicons: dict[tuple[int, int], Lexicon] = field(default_factory=dict)

    @classmethod
    def empty(cls, geo: Geography) -> "Population":
        return cls(geo=geo, counts=np.zeros(geo.shape, dtype=np.float64))

    def seed(
        self,
        lat: float,
        lon: float,
        count: float,
        lexicon: Lexicon,
    ) -> tuple[int, int]:
        """Place a founding population at ``(lat, lon)`` with ``lexicon``."""
        i, j = self.geo.coord_to_cell(lat, lon)
        if not self.geo.is_habitable(i, j):
            raise ValueError(f"({lat}, {lon}) is not habitable")
        self.counts[i, j] = count
        self.lexicons[(i, j)] = lexicon.copy()
        return i, j

    @property
    def populated_mask(self) -> np.ndarray:
        """Boolean mask of cells that have been founded (carry a lexicon).

        Note this is *stricter* than ``counts > 0``: diffusion can leave
        tiny sub-threshold mass in cells that have not yet been settled.
        """
        mask = np.zeros(self.geo.shape, dtype=bool)
        for (i, j) in self.lexicons:
            mask[i, j] = True
        return mask

    @property
    def total_population(self) -> float:
        return float(self.counts.sum())

    @property
    def n_populated_cells(self) -> int:
        return len(self.lexicons)

    def lineage_locations(self) -> list[tuple[int, int]]:
        return list(self.lexicons.keys())


class MigrationSimulator:
    """Forward-time simulator coupling demography and language change.

    Args:
        geo: Geography grid.
        evolver: Shared cognate evolver (owns the global cognate-ID counter).
        growth_rate: Logistic growth rate per year. Default ``0.005``,
            roughly matching pre-industrial human r ≈ 0.5%/yr.
        K_max: Maximum population per cell at full carrying capacity
            (``capacity = 1``). Default ``1.0`` — everything is in
            relative units.
        migration_rate: Fraction of each cell's population dispersing per
            year. Default ``0.05``.
        contact_sigma_km: Length scale of contact decay; borrowing weight
            from a neighbor at distance ``d`` is ``exp(-d / contact_sigma_km)``.
            Default ``300`` km.
        founding_threshold: Fraction of a cell's *carrying capacity* a
            previously-empty cell must reach to "settle" and acquire a
            lexicon. Default ``0.05`` = 5% of ``K_max * capacity``. Scaling
            by capacity is what lets low-capacity terrain (desert at 5%,
            mountain at 10%) be settled at all — an absolute threshold of
            0.05 would make desert cells unsettleable.
        rng: Optional ``numpy.random.Generator`` (currently unused — the
            model is deterministic at the demography layer; stochasticity
            lives in the cognate evolver).
        isolated_cells: Optional set of ``(i, j)`` cells that are
            *pinned* — they hold a lexicon and evolve via internal
            innovation only. They do not exchange migrants with any
            neighbour, and other cells do not borrow from them. This
            models a small population that broke off from the homeland
            early and stayed put in geographic isolation (e.g., the
            Anatolian breakaway from PIE).
    """

    def __init__(
        self,
        geo: Geography,
        evolver: CognateEvolver,
        *,
        growth_rate: float = 0.005,
        K_max: float = 1.0,
        migration_rate: float = 0.05,
        contact_sigma_km: float = 300.0,
        founding_threshold: float = 0.05,
        rng: np.random.Generator | None = None,
        isolated_cells: set[tuple[int, int]] | None = None,
    ) -> None:
        self.geo = geo
        self.evolver = evolver
        self.r = float(growth_rate)
        self.K_max = float(K_max)
        self.m = float(migration_rate)
        self.contact_sigma = float(contact_sigma_km)
        self.founding_threshold = float(founding_threshold)
        self.rng = rng if rng is not None else np.random.default_rng()

        n_lat, n_lon = geo.shape
        self.isolated_cells: frozenset[tuple[int, int]] = frozenset(
            isolated_cells or ()
        )
        for (i, j) in self.isolated_cells:
            if not (0 <= i < n_lat and 0 <= j < n_lon):
                raise ValueError(f"isolated cell ({i}, {j}) out of bounds")

        # Carrying capacity per cell.
        self.K = self.K_max * geo.capacity

        # Inverse friction with ocean → 0.
        inv_fric = np.where(
            np.isfinite(geo.friction),
            1.0 / np.maximum(geo.friction, 1e-9),
            0.0,
        )
        inv_fric[~geo.land_mask] = 0.0
        self.inv_friction = inv_fric

        # Per-direction outflow weights and normalised version.
        # ``outflow_norm[k, i, j]`` is the fraction of cell (i,j)'s outflow
        # heading in direction k; it sums to 1 over k where the cell has at
        # least one valid land neighbor, and 0 otherwise.
        self.outflow_weights = np.zeros((len(NEIGHBOR_OFFSETS), n_lat, n_lon))
        for k, (di, dj) in enumerate(NEIGHBOR_OFFSETS):
            rolled = np.roll(inv_fric, shift=(-di, -dj), axis=(0, 1))
            mask = self._valid_offset_mask(di, dj, geo.shape)
            self.outflow_weights[k] = np.where(mask, rolled, 0.0)

        # Isolation: zero out outflow *from* isolated cells and outflow
        # *into* isolated cells (latter via the source's outgoing weight
        # in the direction that targets the isolated cell). The remaining
        # weights are then renormalised so non-isolated cells migrate at
        # the intended overall rate, just routed through fewer directions.
        for (ic, jc) in self.isolated_cells:
            self.outflow_weights[:, ic, jc] = 0.0
            for k, (di, dj) in enumerate(NEIGHBOR_OFFSETS):
                si, sj = ic - di, jc - dj
                if 0 <= si < n_lat and 0 <= sj < n_lon:
                    self.outflow_weights[k, si, sj] = 0.0

        total = self.outflow_weights.sum(axis=0)
        self.outflow_norm = np.where(
            total > 0,
            self.outflow_weights / np.where(total > 0, total, 1.0),
            0.0,
        )

    @staticmethod
    def _valid_offset_mask(di: int, dj: int, shape: tuple[int, int]) -> np.ndarray:
        """Mask of cells whose (+di, +dj)-neighbour is in-bounds."""
        n_lat, n_lon = shape
        mask = np.ones(shape, dtype=bool)
        if di > 0:
            mask[-di:, :] = False
        elif di < 0:
            mask[: abs(di), :] = False
        if dj > 0:
            mask[:, -dj:] = False
        elif dj < 0:
            mask[:, : abs(dj)] = False
        return mask

    # ------------------------------------------------------------------ step

    def step(self, pop: Population, dt: float = 1.0) -> Population:
        """Advance ``pop`` by ``dt`` years."""
        if pop.geo is not self.geo:
            raise ValueError("population uses a different geography")
        counts = pop.counts.copy()

        # ---- 1. Logistic growth ----
        K_safe = np.where(self.K > 0, self.K, 1.0)
        growth = self.r * counts * (1.0 - counts / K_safe) * dt
        growth = np.where(self.K > 0, growth, 0.0)
        counts = np.clip(counts + growth, 0.0, self.K + 1e-9)

        # ---- 2. Migration ----
        m_eff = min(self.m * dt, 1.0)
        outflow_per_dir = m_eff * counts[None, :, :] * self.outflow_norm
        total_out = outflow_per_dir.sum(axis=0)

        inflows = np.zeros_like(self.outflow_weights)
        for k, (di, dj) in enumerate(NEIGHBOR_OFFSETS):
            # Migrants leaving (i', j') in direction k arrive at (i'+di, j'+dj),
            # so cell (i, j) receives from (i-di, j-dj). Roll by (+di, +dj).
            rolled = np.roll(outflow_per_dir[k], shift=(di, dj), axis=(0, 1))
            mask = self._valid_offset_mask(-di, -dj, self.geo.shape)
            inflows[k] = np.where(mask, rolled, 0.0)
        total_in = inflows.sum(axis=0)

        new_counts = np.clip(counts - total_out + total_in, 0.0, self.K + 1e-9)

        # ---- 3. Founding events ----
        # An *unfounded* cell (no lexicon yet) that has just crossed the
        # founding threshold inherits the dominant source's lexicon. We
        # check "no lexicon yet" rather than "counts == 0" because
        # diffusion can leave sub-threshold mass in cells before they are
        # actually settled, and we want such cells to remain eligible
        # for founding in later steps.
        unfounded = np.ones(self.geo.shape, dtype=bool)
        for (ui, uj) in pop.lexicons:
            unfounded[ui, uj] = False
        # Pinned isolated cells are managed by the caller; never re-found.
        for (ic, jc) in self.isolated_cells:
            unfounded[ic, jc] = False
        # Threshold scales with cell capacity so deserts/mountains are
        # reachable with proportionally less absolute mass.
        thresh = self.founding_threshold * self.K
        became_pop = unfounded & (new_counts >= thresh) & self.geo.land_mask

        new_lexicons: dict[tuple[int, int], Lexicon] = dict(pop.lexicons)
        if became_pop.any():
            n_lat, n_lon = self.geo.shape
            for ci, cj in zip(*np.where(became_pop)):
                # Pick the populated source that contributed the most
                # inflow this step. Fall through if no neighbour has a
                # lexicon yet — the cell stays unfounded and will retry
                # next step. This matters when migrants arrive via
                # below-threshold transit cells that haven't been
                # founded themselves.
                best_src: tuple[int, int] | None = None
                best_inflow = 0.0
                for k, (di, dj) in enumerate(NEIGHBOR_OFFSETS):
                    si, sj = int(ci) - di, int(cj) - dj
                    if not (0 <= si < n_lat and 0 <= sj < n_lon):
                        continue
                    if (si, sj) not in pop.lexicons:
                        continue
                    if inflows[k, ci, cj] > best_inflow:
                        best_inflow = float(inflows[k, ci, cj])
                        best_src = (si, sj)
                if best_src is not None:
                    new_lexicons[(int(ci), int(cj))] = pop.lexicons[best_src].copy()

        # ---- 4. Cognate evolution ----
        evolved: dict[tuple[int, int], Lexicon] = {}
        for (i, j), lex in new_lexicons.items():
            if new_counts[i, j] <= 0:
                continue  # cell vanished (rare numerical edge case)
            if (i, j) in self.isolated_cells:
                # Pinned isolated cell: pure internal innovation, no contact.
                evolved[(i, j)] = self.evolver.step(lex, dt=dt)
                continue
            neigh_lex: list[Lexicon] = []
            neigh_w: list[float] = []
            for di, dj in NEIGHBOR_OFFSETS:
                ni, nj = i + di, j + dj
                if (ni, nj) in self.isolated_cells:
                    # Isolated cells do not transmit borrowings outward.
                    continue
                src = new_lexicons.get((ni, nj))
                if src is None or new_counts[ni, nj] <= 0:
                    continue
                d_km = self.geo.great_circle_km((i, j), (ni, nj))
                w = (new_counts[ni, nj] / self.K_max) * np.exp(-d_km / self.contact_sigma)
                if w > 0:
                    neigh_lex.append(src)
                    neigh_w.append(float(w))

            if neigh_lex:
                intensity = float(sum(neigh_w))
                evolved[(i, j)] = self.evolver.step(
                    lex,
                    dt=dt,
                    neighbors=neigh_lex,
                    contact_weights=np.asarray(neigh_w),
                    borrow_intensity=intensity,
                )
            else:
                evolved[(i, j)] = self.evolver.step(lex, dt=dt)

        return Population(geo=self.geo, counts=new_counts, lexicons=evolved)

    # ------------------------------------------------------------------ run

    def run(
        self,
        pop: Population,
        n_years: int,
        dt: float = 1.0,
        snapshot_years: list[int] | None = None,
    ) -> list[tuple[int, Population]]:
        """Run the simulation for ``n_years`` years.

        Args:
            pop: Initial population state.
            n_years: Total simulation length in years.
            dt: Step size in years.
            snapshot_years: If provided, snapshots are recorded at the listed
                years (clamped to multiples of ``dt``). Defaults to start
                and end only.

        Returns:
            List of ``(year, Population)`` snapshots in chronological order.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        n_steps = int(round(n_years / dt))
        if abs(n_steps * dt - n_years) > 1e-9:
            raise ValueError(f"n_years={n_years} is not a multiple of dt={dt}")

        if snapshot_years is None:
            snapshot_set: set[int] = {0, n_years}
        else:
            snapshot_set = set(snapshot_years) | {0, n_years}

        snapshots: list[tuple[int, Population]] = []
        if 0 in snapshot_set:
            snapshots.append((0, pop))

        for step in range(1, n_steps + 1):
            pop = self.step(pop, dt=dt)
            year = int(round(step * dt))
            if year in snapshot_set:
                snapshots.append((year, pop))

        return snapshots
