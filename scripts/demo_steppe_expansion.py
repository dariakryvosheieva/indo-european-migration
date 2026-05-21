"""Visualize the Yamnaya / Steppe Indo-European expansion.

Seed a single population at the Pontic-Caspian steppe (50 °N, 45 °E) carrying
the PIE lexicon; let it grow logistically and disperse across Eurasia under
the friction map for ~5000 years; record snapshots and plot population
density at four timepoints.

This is the demographic engine — language change runs in parallel inside
each populated cell but isn't visualized in this demo. The next demo will
overlay lexical similarity to show branch differentiation.

Saves ``figs/demo_steppe_expansion.png``.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from pie_sim import (
    CognateEvolver,
    MigrationSimulator,
    Population,
    stylized_eurasia,
)


def main() -> None:
    geo = stylized_eurasia(cell_size=1.0)
    rng = np.random.default_rng(2026)
    evolver = CognateEvolver(
        n_meanings=200,
        innovation_rate=1.5e-4,
        borrowing_rate=2.0e-5,
        rng=rng,
    )
    sim = MigrationSimulator(
        geo, evolver,
        growth_rate=0.01,
        K_max=1.0,
        migration_rate=0.04,
        contact_sigma_km=300.0,
        founding_threshold=0.05,
    )

    pop = Population.empty(geo)
    pop.seed(lat=50.0, lon=45.0, count=0.5, lexicon=evolver.initial_lexicon())

    snapshot_years = [0, 1000, 2000, 3000, 4000, 5000]

    print("Running steppe-seeded simulation for 5000 years...")
    t0 = time.time()
    snaps = sim.run(pop, n_years=5000, dt=10.0, snapshot_years=snapshot_years)
    print(f"  done in {time.time()-t0:.1f}s; final populated cells: "
          f"{snaps[-1][1].n_populated_cells}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    extent = (geo.lon_min, geo.lon_max, geo.lat_min, geo.lat_max)

    # Pre-render terrain once for reuse.
    terrain_rgb = np.full((*geo.shape, 3), 0.93)
    terrain_rgb[~geo.land_mask] = mcolors.to_rgb("#9EC9E5")

    im = None
    for ax, (year, snap) in zip(axes.flat, snaps):
        ax.imshow(terrain_rgb, origin="lower", extent=extent, interpolation="nearest")
        density = np.where(snap.populated_mask, snap.counts, np.nan)
        im = ax.imshow(
            density, origin="lower", extent=extent, cmap=cmap, norm=norm,
            interpolation="nearest", alpha=0.95,
        )
        ax.plot(45.0, 50.0, "k*", markersize=12, markeredgecolor="white",
                markeredgewidth=0.6, zorder=10)
        ax.set_xlim(geo.lon_min, geo.lon_max)
        ax.set_ylim(geo.lat_min, geo.lat_max)
        ax.set_aspect("equal")
        ax.set_title(
            f"year {year}      cells: {snap.n_populated_cells}      "
            f"total pop: {snap.total_population:.1f}",
            fontsize=10,
        )
        ax.tick_params(labelsize=8)

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="population density (relative to $K_{\\max}$)")
    fig.suptitle(
        "Steppe scenario: PIE seeded at Pontic-Caspian steppe (50 °N, 45 °E)",
        fontsize=14,
    )

    out_dir = Path(__file__).resolve().parents[1] / "figs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "demo_steppe_expansion.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
