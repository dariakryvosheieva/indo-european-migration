"""Pre-defined Indo-European migration scenarios.

Each :class:`Scenario` describes a candidate Indo-European homeland and the
seed location (lat, lon) where the simulation plants the founding lexicon.
The simulator is otherwise identical across scenarios — what differs is
purely *where* the pin goes — so head-to-head comparison of the resulting
phylogenetic trees is a clean test of how much the geographic origin
shapes the divergence pattern.

Two scenarios are provided:

- :data:`STEPPE` — Yamnaya / Pontic-Caspian origin (Anthony, Reich,
  Chang et al. 2015). Seed at (50 °N, 45 °E), the heart of the Yamnaya
  cultural horizon ~3500 BCE.
- :data:`ANATOLIAN` — Anatolian / farming-dispersal origin (Renfrew 1987,
  partially revived by Heggarty et al. 2023). Seed at (38.5 °N, 35 °E),
  the central Anatolian plateau, ~7000 BCE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import pickle
import time

import numpy as np

from pie_sim.cognates import CognateEvolver
from pie_sim.geography import stylized_eurasia
from pie_sim.population import MigrationSimulator, Population


class IsolatedBreakaway(NamedTuple):
    """A satellite lineage that splits from the homeland and stays isolated.

    Attributes:
        lat: Latitude of the pinned cell (°N).
        lon: Longitude of the pinned cell (°E).
        name: Human-readable label (e.g., ``"Anatolian-early"``).
        pre_evolve_years: Years of *pre-simulation* drift to apply to the
            breakaway lexicon before the main run begins. Models the gap
            between when the breakaway split from the homeland and when
            the main expansion (Yamnaya horizon) starts. With our
            innovation rate, pairwise distances saturate near
            ``1 - e^{-2 μ T_run}`` after the run, so a head start of
            ``pre_evolve_years`` yields a clear basal gap of
            ``e^{-2 μ T_run} - e^{-μ (2 T_run + pre)}`` in distance.
            ``0`` means the breakaway diverges *at* year 0 with no head
            start (which leaves it indistinguishable from late-saturating
            sister branches and is usually not what you want).
    """
    lat: float
    lon: float
    name: str
    pre_evolve_years: int = 0


@dataclass(frozen=True)
class Scenario:
    """A candidate Indo-European homeland and its seed coordinates.

    Attributes:
        name: Short slug, used for filenames (e.g., ``"steppe"``).
        pretty_name: Display name (e.g., ``"Steppe (Yamnaya)"``).
        seed_lat: Latitude of the founding population (°N).
        seed_lon: Longitude of the founding population (°E).
        description: One-sentence summary citing the proponent literature.
        isolated_breakaways: Optional satellite populations that break
            off from the homeland and stay geographically isolated for
            the entire run (no migration, no contact-mediated borrowing).
            Each entry is an :class:`IsolatedBreakaway` carrying both
            its location and the number of pre-simulation drift years
            applied before the main run starts. This encodes the
            empirical fact that some sub-groups (most famously
            Anatolian) split off from PIE before the main expansion and
            so accumulated more lexical drift from the rest of the
            family than later branches did.
    """
    name: str
    pretty_name: str
    seed_lat: float
    seed_lon: float
    description: str
    isolated_breakaways: tuple[IsolatedBreakaway, ...] = ()


STEPPE = Scenario(
    name="steppe",
    pretty_name="Steppe (Yamnaya / Pontic-Caspian)",
    seed_lat=50.0,
    seed_lon=45.0,
    description=(
        "Yamnaya / Pontic-Caspian origin, ~3500 BCE; supported by "
        "ancient-DNA evidence (Anthony 2007; Haak et al. 2015; "
        "Reich 2018; Chang et al. 2015). The Anatolian branch is "
        "modelled as an early breakaway pinned to central Anatolia "
        "with 1500 years of pre-simulation drift, matching the "
        "archaeological gap between proto-Anatolian splitting from "
        "Indo-Anatolian (~5000 BCE) and the Yamnaya horizon (~3500 BCE)."
    ),
    isolated_breakaways=(
        IsolatedBreakaway(
            lat=40.0, lon=33.5, name="Anatolian-early",
            pre_evolve_years=1500,
        ),
    ),
)


ANATOLIAN = Scenario(
    name="anatolian",
    pretty_name="Anatolian (farming dispersal)",
    seed_lat=38.5,
    seed_lon=35.0,
    description=(
        "Anatolian / farming-dispersal origin, ~7000 BCE; revived "
        "by Heggarty et al. 2023 (Renfrew 1987). No early breakaway "
        "is needed in this scenario — Anatolian is the homeland."
    ),
)


SCENARIOS: dict[str, Scenario] = {
    STEPPE.name: STEPPE,
    ANATOLIAN.name: ANATOLIAN,
}


# ---------------------------------------------------------------- defaults


DEFAULT_SIM_PARAMS: dict = dict(
    growth_rate=0.01,
    K_max=1.0,
    migration_rate=0.04,
    contact_sigma_km=300.0,
    founding_threshold=0.05,
)
DEFAULT_COGNATE_PARAMS: dict = dict(
    # ~1500 reconstructed PIE roots is the conservative end of scholarly
    # estimates (Pokorny ~1800; Mallory & Adams ~1500-2000; Watkins
    # ~1000-1500). Going beyond a Swadesh-100/200 list is also
    # statistically valuable — it cuts the standard error on each
    # pairwise cognate distance from ~0.029 (n=200) to ~0.011 (n=1500),
    # which markedly stabilizes the reconstructed tree topology.
    n_meanings=1500,
    innovation_rate=1.5e-4,
    borrowing_rate=2.0e-5,
)
DEFAULT_N_YEARS: int = 5000
DEFAULT_DT: float = 10.0
DEFAULT_SEED: int = 2026
TOCHARIAN_EARLY_BREAKAWAY = IsolatedBreakaway(
    lat=40.0,
    lon=87.0,
    name="Tocharian-early",
    pre_evolve_years=500,
)


# ----------------------------------------------------------------- runner


def run_scenario(
    scenario: Scenario,
    *,
    n_years: int = DEFAULT_N_YEARS,
    dt: float = DEFAULT_DT,
    cell_size: float = 1.0,
    sim_params: dict | None = None,
    cognate_params: dict | None = None,
    rng_seed: int = DEFAULT_SEED,
    pre_evolve_hittite: bool = True,
    pre_evolve_tocharian: bool = False,
    verbose: bool = True,
) -> Population:
    """Run a scenario from seeding to the requested simulation horizon.

    Args:
        scenario: Which homeland to seed.
        n_years: Total simulation length in years (post-seeding).
        dt: Time step in years.
        cell_size: Geography resolution in degrees.
        sim_params: Overrides for :class:`MigrationSimulator` parameters.
        cognate_params: Overrides for :class:`CognateEvolver` parameters.
        rng_seed: Seed for the cognate evolver's RNG (the demography
            layer is deterministic).
        pre_evolve_hittite: In the steppe scenario only, keep the
            Anatolian-early breakaway with its configured pre-simulation
            drift (default) or disable that special treatment and let
            Hittite emerge from the regular expansion dynamics.
        pre_evolve_tocharian: Add an isolated Tocharian-early breakaway
            pinned to the Tarim Basin with
            500 years of pre-simulation drift.
        verbose: Print timing info.

    Returns:
        Final :class:`Population` after ``n_years`` of evolution.
    """
    sp = {**DEFAULT_SIM_PARAMS, **(sim_params or {})}
    cp = {**DEFAULT_COGNATE_PARAMS, **(cognate_params or {})}

    geo = stylized_eurasia(cell_size=cell_size)
    rng = np.random.default_rng(rng_seed)
    evolver = CognateEvolver(rng=rng, **cp)
    breakaways = scenario.isolated_breakaways
    if scenario.name == STEPPE.name and not pre_evolve_hittite:
        # Disable the special Anatolian-early handling so Hittite follows
        # the same demographic/lexical path as other IE groups.
        breakaways = tuple(
            b for b in scenario.isolated_breakaways if b.name != "Anatolian-early"
        )
    if pre_evolve_tocharian:
        breakaways = (*breakaways, TOCHARIAN_EARLY_BREAKAWAY)

    # Resolve isolated breakaway coordinates to grid cells and validate.
    isolated_cells: set[tuple[int, int]] = set()
    for breakaway in breakaways:
        i, j = geo.coord_to_cell(breakaway.lat, breakaway.lon)
        if not geo.is_habitable(i, j):
            raise ValueError(
                f"isolated breakaway '{breakaway.name}' at "
                f"({breakaway.lat}, {breakaway.lon}) is not habitable"
            )
        isolated_cells.add((i, j))

    sim = MigrationSimulator(geo, evolver, isolated_cells=isolated_cells, **sp)

    # All seeds (homeland + breakaways) start from the *same* initial
    # PIE lexicon — they're sister lineages whose common ancestor is
    # this fresh lexicon. Breakaways may be pre-evolved before the
    # main run starts (representing earlier-than-Yamnaya divergence).
    init_lex = evolver.initial_lexicon()

    pop = Population.empty(geo)
    pop.seed(
        lat=scenario.seed_lat,
        lon=scenario.seed_lon,
        count=0.5,
        lexicon=init_lex,
    )

    for breakaway in breakaways:
        breakaway_lex = init_lex.copy()
        if breakaway.pre_evolve_years > 0:
            n_pre_steps = max(1, int(round(breakaway.pre_evolve_years / dt)))
            for _ in range(n_pre_steps):
                # Pure internal innovation, no neighbours.
                breakaway_lex = evolver.step(breakaway_lex, dt=dt)
        i, j = geo.coord_to_cell(breakaway.lat, breakaway.lon)
        # Plant at half the cell's carrying capacity so the lexicon
        # is anchored in a healthy local population.
        pop.counts[i, j] = 0.5 * sim.K[i, j]
        pop.lexicons[(i, j)] = breakaway_lex

    if verbose:
        print(
            f"Running {scenario.pretty_name} for {n_years} years "
            f"(seed {scenario.seed_lat}°N, {scenario.seed_lon}°E"
            + (
                f"; {len(breakaways)} isolated breakaway(s)"
                if breakaways
                else ""
            )
            + ")..."
        )
    t0 = time.time()
    snaps = sim.run(pop, n_years=n_years, dt=dt, snapshot_years=[0, n_years])
    if verbose:
        print(f"  done in {time.time() - t0:.1f}s")

    return snaps[-1][1]


def run_or_load_scenario(
    scenario: Scenario,
    cache_dir: Path,
    *,
    n_years: int = DEFAULT_N_YEARS,
    dt: float = DEFAULT_DT,
    sim_params: dict | None = None,
    cognate_params: dict | None = None,
    rng_seed: int = DEFAULT_SEED,
    pre_evolve_hittite: bool = True,
    pre_evolve_tocharian: bool = False,
    cache_tag: str | None = None,
    force: bool = False,
) -> Population:
    """Cached runner: load ``data/{scenario.name}_final.pkl`` if present.

    Args:
        scenario: Which scenario to run.
        cache_dir: Directory where ``{scenario.name}_final.pkl`` lives.
        pre_evolve_hittite: In the steppe scenario only, enable/disable
            Anatolian-early pre-evolution before the main run.
        pre_evolve_tocharian: Enable/disable the optional Tocharian-early
            pre-evolution before the main run.
        cache_tag: Optional extra identifier appended to the cache
            filename stem (e.g., ``"no_borrowing"``) so alternative
            parameterizations do not collide.
        force: Ignore the cache and re-run.
        Other args: passed to :func:`run_scenario`.
    """
    cache_stem = scenario.name
    if scenario.name == STEPPE.name and not pre_evolve_hittite:
        cache_stem = f"{scenario.name}_no_hittite_pre_evolve"
    if pre_evolve_tocharian:
        cache_stem = f"{cache_stem}_with_tocharian_pre_evolve"
    if cache_tag:
        cache_stem = f"{cache_stem}_{cache_tag}"
    cache_path = cache_dir / f"{cache_stem}_final.pkl"
    if cache_path.exists() and not force:
        print(f"Loading cached {scenario.name} simulation from {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    pop = run_scenario(
        scenario,
        n_years=n_years,
        dt=dt,
        sim_params=sim_params,
        cognate_params=cognate_params,
        rng_seed=rng_seed,
        pre_evolve_hittite=pre_evolve_hittite,
        pre_evolve_tocharian=pre_evolve_tocharian,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(pop, f)
    print(f"Saved cache to {cache_path}")
    return pop
