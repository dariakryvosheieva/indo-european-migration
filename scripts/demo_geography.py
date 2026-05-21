"""Visualize the stylized Eurasian geography.

Renders the terrain map and overlays the two candidate Indo-European
homelands (Pontic-Caspian steppe vs. Anatolian highlands) plus a few
historically important reference locations.

Saves ``figs/demo_geography.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from pie_sim import stylized_eurasia


REFERENCE_LOCATIONS = [
    # (lat, lon, label, color)
    (50.0, 45.0, "Yamnaya / Pontic-Caspian", "red"),
    (38.5, 35.0, "Anatolian Urheimat", "blue"),
    (50.0, 25.0, "Corded Ware", "darkorange"),
    (47.5, 65.0, "Andronovo", "purple"),
    (42.0, 75.0, "Tarim Basin (Tocharian)", "saddlebrown"),
    (28.0, 77.0, "Indus / Indo-Aryan", "darkgreen"),
    (37.5, 23.5, "Mycenae", "black"),
    (52.0, 0.0, "British Isles", "black"),
]


def main() -> None:
    geo = stylized_eurasia(cell_size=1.0)

    fig, ax = plt.subplots(figsize=(15, 7))
    geo.plot(ax=ax, show_legend=True, gridlines=False)

    for lat, lon, label, color in REFERENCE_LOCATIONS:
        ax.plot(lon, lat, marker="*", markersize=14, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=10)
        ax.annotate(
            label, xy=(lon, lat), xytext=(6, 4), textcoords="offset points",
            fontsize=8.5, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    ax.set_title(
        "Stylized Eurasian geography for the IE migration simulation\n"
        f"({geo.shape[0]}×{geo.shape[1]} grid at {geo.cell_size}° resolution)",
        fontsize=13,
    )
    ax.set_xlim(geo.lon_min, geo.lon_max)
    ax.set_ylim(geo.lat_min, geo.lat_max)
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parents[1] / "figs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "demo_geography.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
