"""Analysis utilities: branch sampling, distance matrices, tree building.

Provides the bridge between a finished simulation (a :class:`Population`)
and the numerical artefacts we want to compare against the empirical IE
phylogeny: pairwise cognate distances and a UPGMA tree.

The canonical sample list :data:`IE_BRANCH_SAMPLES` places one point per
major Indo-European branch at approximately its earliest attestation
location. Extraction uses nearest-cell (single-point) sampling only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from pie_sim.cognates import Lexicon, cognate_distance
from pie_sim.geography import EARTH_RADIUS_KM
from pie_sim.population import Population


@dataclass(frozen=True)
class BranchSample:
    """A sampled Indo-European branch at an attestation point.

    Attributes:
        name: Human-readable label (used as plot tip label).
        lat: Latitude of attestation point (degrees N).
        lon: Longitude of attestation point (degrees E).
        family: Higher-level grouping for clustering / plotting.
        color: Hex colour for plot tips.
    """

    name: str
    lat: float
    lon: float
    family: str = ""
    color: str = "#333333"


# Approximate earliest-attestation locations for the major IE branches.
# Coordinates are reasonable — they are not meant as historical claims.
IE_BRANCH_SAMPLES: tuple[BranchSample, ...] = (
    BranchSample("Hittite",   40.0, 33.5, "Anatolian",     "#A6611A"),
    BranchSample("Tocharian", 40.0, 87.0, "Tocharian",     "#DFC27D"),
    BranchSample("Persian",   32.0, 53.0, "Indo-Iranian",  "#B2182B"),
    BranchSample("Sanskrit",  28.0, 77.0, "Indo-Iranian",  "#D6604D"),
    BranchSample("Greek",     38.0, 23.5, "Hellenic",      "#762A83"),
    BranchSample("Armenian",  40.0, 44.5, "Armenian",      "#9970AB"),
    BranchSample("Albanian",  41.0, 20.0, "Albanian",      "#5AAE61"),
    BranchSample("Latin",     42.0, 12.5, "Italic",        "#1B7837"),
    BranchSample("Gaulish",   50.0,  2.0, "Celtic",        "#018571"),
    BranchSample("Gaelic",    53.0, -8.0, "Celtic",        "#80CDC1"),
    BranchSample("Germanic",  55.0, 12.0, "Germanic",      "#2166AC"),
    BranchSample("Slavic",    52.0, 27.0, "Balto-Slavic",  "#4A1486"),
    BranchSample("Baltic",    55.5, 24.0, "Balto-Slavic",  "#807DBA"),
)


def extract_branch_lexicons(
    pop: Population,
    samples: Sequence[BranchSample],
    max_distance_km: float = 500.0,
) -> dict[str, tuple[BranchSample, Lexicon, tuple[int, int]]]:
    """Extract one representative lexicon per branch sample.

    Uses nearest-cell extraction only.

    Args:
        pop: Final population state from a simulation.
        samples: Sample locations to query.
        max_distance_km: Reject samples whose nearest populated cell is
            farther than this. Useful for detecting branches the
            simulation never reached.

    Returns:
        Dict from sample name to ``(sample, lexicon, (i, j))`` where
        ``(i, j)`` is the nearest contributing populated cell.
    """
    out: dict[str, tuple[BranchSample, Lexicon, tuple[int, int]]] = {}
    if not pop.lexicons:
        return out

    populated_cells = list(pop.lexicons.keys())
    centres = np.array(
        [pop.geo.cell_center(i, j) for (i, j) in populated_cells],
        dtype=np.float64,
    )
    cell_lats = centres[:, 0]
    cell_lons = centres[:, 1]

    for sample in samples:
        phi1 = np.radians(sample.lat)
        phi2 = np.radians(cell_lats)
        dphi = phi2 - phi1
        dlam = np.radians(cell_lons - sample.lon)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
        d_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

        idx = int(np.argmin(d_km))
        if d_km[idx] > max_distance_km:
            continue

        cell = populated_cells[idx]
        out[sample.name] = (sample, pop.lexicons[cell], cell)
    return out


def cognate_distance_matrix(
    extracted: dict[str, tuple[BranchSample, Lexicon, tuple[int, int]]],
    order: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Pairwise cognate-distance matrix for extracted samples.

    Args:
        extracted: Output of :func:`extract_branch_lexicons`.
        order: Optional explicit ordering for rows/cols. Defaults to dict order.

    Returns:
        ``(matrix, names)`` where ``matrix[i, j]`` is the cognate distance
        between samples ``names[i]`` and ``names[j]``. Distance is
        ``1 - p_shared`` over Swadesh meaning slots.
    """
    if order is None:
        order = list(extracted.keys())
    missing = [n for n in order if n not in extracted]
    if missing:
        raise KeyError(f"order contains samples not in extracted: {missing}")

    n = len(order)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = cognate_distance(extracted[order[i]][1], extracted[order[j]][1])
            M[i, j] = d
            M[j, i] = d
    return M, list(order)


def build_upgma_tree(
    distance_matrix: np.ndarray,
    names: Sequence[str],
    method: str = "average",
) -> np.ndarray:
    """Hierarchical-clustering linkage matrix from a pairwise distance matrix.

    ``method='average'`` is UPGMA (the standard glottochronology tree
    inference). ``'single'``, ``'complete'``, ``'ward'`` are also accepted.

    Args:
        distance_matrix: Symmetric ``(n, n)`` matrix of pairwise distances.
        names: Tip labels, length ``n`` (used for shape validation only —
            scipy's ``linkage`` ignores them).
        method: scipy linkage method.

    Returns:
        scipy linkage matrix of shape ``(n - 1, 4)``.
    """
    if distance_matrix.shape != (len(names), len(names)):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} != "
            f"({len(names)}, {len(names)})"
        )
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("distance_matrix must be symmetric")
    condensed = squareform(distance_matrix, checks=False)
    return linkage(condensed, method=method)


__all__ = [
    "BranchSample",
    "IE_BRANCH_SAMPLES",
    "extract_branch_lexicons",
    "cognate_distance_matrix",
    "build_upgma_tree",
]
