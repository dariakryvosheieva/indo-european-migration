"""Branch-level analysis of an Indo-European migration scenario.

Runs (or loads cached) a 5000-year simulation under a chosen homeland
scenario, samples lexicons at the historical attestation locations of
the major Indo-European branches, and produces:

1. A pairwise cognate-distance heatmap.
2. A UPGMA dendrogram with tip labels colored by family and bands marking
   the empirical IE families.

Usage:
    python scripts/demo_branch_analysis.py --scenario steppe
    python scripts/demo_branch_analysis.py --scenario anatolian
    python scripts/demo_branch_analysis.py --scenario steppe --force
    python scripts/demo_branch_analysis.py --scenario steppe --dt 1 --migration-rate 0.004

Outputs ``figs/{scenario}_branch_similarity.png`` and
``figs/{scenario}_branch_tree.png``; caches the simulation under
``data/{scenario}_final.pkl``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from pie_sim import (
    IE_BRANCH_SAMPLES,
    SCENARIOS,
    Scenario,
    build_upgma_tree,
    cognate_distance_matrix,
    extract_branch_lexicons,
    run_or_load_scenario,
)
from pie_sim.scenarios import DEFAULT_COGNATE_PARAMS, DEFAULT_DT, DEFAULT_SIM_PARAMS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGS_DIR = PROJECT_ROOT / "figs"


# Family ordering for the heatmap — broadly mirrors the empirical IE
# tree, with peripheral / first-out branches first and the
# Northwest-IE / Late-PIE cluster at the end.
FAMILY_ORDER = [
    "Anatolian",
    "Tocharian",
    "Indo-Iranian",
    "Hellenic",
    "Armenian",
    "Albanian",
    "Italic",
    "Celtic",
    "Germanic",
    "Balto-Slavic",
]


def order_by_family(extracted: dict) -> list[str]:
    """Order sample names so that same-family samples are contiguous."""
    return [
        name
        for fam in FAMILY_ORDER
        for name, (s, *_) in extracted.items()
        if s.family == fam
    ]


def plot_heatmap(
    extracted: dict,
    out_path: Path,
    scenario: Scenario,
    n_years: int,
) -> None:
    """Pairwise cognate-distance heatmap."""
    ordered_names = order_by_family(extracted)
    M, names = cognate_distance_matrix(extracted, order=ordered_names)
    n = len(names)
    colors = [extracted[name][0].color for name in names]

    fig, ax_hm = plt.subplots(figsize=(9, 7.5))

    im = ax_hm.imshow(M, cmap="viridis_r", vmin=0.0, vmax=1.0, aspect="equal")
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("cognate distance ($1 - p_{shared}$)")

    ax_hm.set_xticks(range(n))
    ax_hm.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax_hm.set_yticks(range(n))
    ax_hm.set_yticklabels(names, fontsize=9)
    for tick, color in zip(ax_hm.get_xticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    for tick, color in zip(ax_hm.get_yticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("bold")

    for i in range(n):
        for j in range(n):
            ax_hm.text(
                j, i, f"{M[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if M[i, j] > 0.55 else "black",
            )

    boundaries: list[int] = []
    last_fam = None
    for k, name in enumerate(names):
        fam = extracted[name][0].family
        if last_fam is not None and fam != last_fam:
            boundaries.append(k)
        last_fam = fam
    for b in boundaries:
        ax_hm.axhline(b - 0.5, color="white", lw=1.2)
        ax_hm.axvline(b - 0.5, color="white", lw=1.2)

    ax_hm.set_title(f"Pairwise cognate distance after {n_years} years")

    fig.suptitle(
        f"Branch-level lexical similarity — {scenario.pretty_name} "
        f"({len(names)} samples, {len(boundaries) + 1} families)",
        fontsize=14, y=0.99,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_tree(extracted: dict, out_path: Path, scenario: Scenario, n_years: int) -> None:
    """UPGMA dendrogram with empirical IE families as background bands."""
    names = order_by_family(extracted)
    M, names = cognate_distance_matrix(extracted, order=names)
    Z = build_upgma_tree(M, names)
    color_by_name = {n: extracted[n][0].color for n in names}

    fig, ax = plt.subplots(figsize=(13, 7.2))

    dd = dendrogram(
        Z, labels=names, ax=ax,
        leaf_rotation=45, leaf_font_size=10,
        link_color_func=lambda _: "#444444",
        above_threshold_color="#444444",
    )

    for tick in ax.get_xticklabels():
        name = tick.get_text()
        if name in color_by_name:
            tick.set_color(color_by_name[name])
            tick.set_fontweight("bold")

    leaf_order = dd["ivl"]
    pos_x = {name: 5 + 10 * k for k, name in enumerate(leaf_order)}
    family_of = {n: extracted[n][0].family for n in leaf_order}
    family_color = {extracted[n][0].family: extracted[n][0].color for n in leaf_order}

    runs: list[tuple[str, list[str]]] = []
    cur_fam: str | None = None
    cur_members: list[str] = []
    for n in leaf_order:
        if family_of[n] != cur_fam:
            if cur_fam is not None:
                runs.append((cur_fam, cur_members))
            cur_fam = family_of[n]
            cur_members = [n]
        else:
            cur_members.append(n)
    if cur_fam is not None:
        runs.append((cur_fam, cur_members))

    ymax = ax.get_ylim()[1]
    for fam, members in runs:
        x_left = pos_x[members[0]] - 5
        x_right = pos_x[members[-1]] + 5
        ax.axvspan(x_left, x_right, ymin=0, ymax=1.0,
                   color=family_color[fam], alpha=0.10, zorder=0)

    ax.set_ylabel("UPGMA distance")
    ax.set_ylim(0, ymax * 1.03)
    ax.set_title(
        f"Reconstructed IE tree from simulated lexicons — {scenario.pretty_name}\n"
        f"{n_years} years from a single seed at "
        f"({scenario.seed_lat}°N, {scenario.seed_lon}°E); "
        "bands mark empirical IE families",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenario", choices=sorted(SCENARIOS.keys()), default="steppe",
        help="Which homeland to simulate.",
    )
    ap.add_argument(
        "--n-years", type=int, default=5000,
        help="Total simulation length in years.",
    )
    ap.add_argument(
        "--dt", type=float, default=DEFAULT_DT,
        help=f"Simulation timestep in years (default: {DEFAULT_DT:g}).",
    )
    ap.add_argument(
        "--migration-rate",
        type=float,
        default=DEFAULT_SIM_PARAMS["migration_rate"],
        help=(
            "Fraction of each cell's population dispersing per year "
            f"(default: {DEFAULT_SIM_PARAMS['migration_rate']:g})."
        ),
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Ignore cached simulation and re-run.",
    )
    ap.add_argument(
        "--pre-evolve-hittite",
        dest="pre_evolve_hittite",
        action="store_true",
        help="Steppe-only: keep the 1500-year Anatolian pre-evolution (default).",
    )
    ap.add_argument(
        "--pre-evolve-tocharian",
        action="store_true",
        help="Add 500 years of Tocharian pre-evolution.",
    )
    args = ap.parse_args()

    scenario = SCENARIOS[args.scenario]
    FIGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    suffix_parts: list[str] = []
    cache_tag_parts: list[str] = []
    if args.dt != DEFAULT_DT:
        dt_tag = f"dt{args.dt:g}".replace(".", "p")
        suffix_parts.append(dt_tag)
        cache_tag_parts.append(dt_tag)
    if args.migration_rate != DEFAULT_SIM_PARAMS["migration_rate"]:
        migration_tag = f"mig{args.migration_rate:g}".replace(".", "p")
        suffix_parts.append(migration_tag)
        cache_tag_parts.append(migration_tag)

    pop = run_or_load_scenario(
        scenario,
        DATA_DIR,
        n_years=args.n_years,
        dt=args.dt,
        sim_params={"migration_rate": args.migration_rate},
        cognate_params={"borrowing_rate": DEFAULT_COGNATE_PARAMS["borrowing_rate"]},
        pre_evolve_hittite=args.pre_evolve_hittite,
        pre_evolve_tocharian=args.pre_evolve_tocharian,
        cache_tag="_".join(cache_tag_parts) if cache_tag_parts else None,
        force=args.force,
    )
    print(
        f"Final population: {pop.n_populated_cells} cells, "
        f"total mass {pop.total_population:.1f}"
    )

    extracted = extract_branch_lexicons(
        pop,
        IE_BRANCH_SAMPLES,
        max_distance_km=800.0,
    )
    found = list(extracted.keys())
    missing = [s.name for s in IE_BRANCH_SAMPLES if s.name not in extracted]
    print(f"Extracted lexicons for {len(found)} / {len(IE_BRANCH_SAMPLES)} branches")
    if missing:
        print(f"  missing: {missing}")

    if args.pre_evolve_hittite:
        suffix_parts.append("pre_evolve_hittite")
    if args.pre_evolve_tocharian:
        suffix_parts.append("pre_evolve_tocharian")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

    plot_heatmap(
        extracted,
        FIGS_DIR / f"{scenario.name}_branch_similarity{suffix}.png",
        scenario,
        args.n_years,
    )
    plot_tree(
        extracted,
        FIGS_DIR / f"{scenario.name}_branch_tree{suffix}.png",
        scenario,
        args.n_years,
    )


if __name__ == "__main__":
    main()
