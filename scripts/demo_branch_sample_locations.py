"""Standalone branch sample-location map.

Renders the branch attestation sample points used by the branch-level
analysis, with both candidate Indo-European homelands marked as simultaneous
starting locations.

Saves ``figs/demo_branch_sample_locations.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from pie_sim import IE_BRANCH_SAMPLES, SCENARIOS, stylized_eurasia


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = PROJECT_ROOT / "figs"


def main() -> None:
    geo = stylized_eurasia(cell_size=1.0)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    geo.plot(ax=ax, show_legend=True, gridlines=False)

    for sample in IE_BRANCH_SAMPLES:
        ax.plot(
            sample.lon,
            sample.lat,
            "o",
            color=sample.color,
            markersize=11,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=10,
        )
        ax.annotate(
            sample.name,
            xy=(sample.lon, sample.lat),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=8.5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

    start_styles = {
        "steppe": ("red", "Steppe start"),
        "anatolian": ("blue", "Anatolian start"),
    }
    for scenario_name in ("steppe", "anatolian"):
        scenario = SCENARIOS[scenario_name]
        color, label = start_styles[scenario_name]
        ax.plot(
            scenario.seed_lon,
            scenario.seed_lat,
            marker="*",
            color=color,
            markersize=22,
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=12,
        )
        ax.annotate(
            label,
            xy=(scenario.seed_lon, scenario.seed_lat),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
            zorder=13,
        )

    ax.set_xlim(geo.lon_min, geo.lon_max)
    ax.set_ylim(geo.lat_min, geo.lat_max)
    ax.set_title(
        "Branch sample locations with Steppe and Anatolian starts",
        fontsize=13,
    )

    fig.tight_layout()
    FIGS_DIR.mkdir(exist_ok=True)
    out_path = FIGS_DIR / "demo_branch_sample_locations.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
