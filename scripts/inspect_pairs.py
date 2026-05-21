"""Quick check that key cognate-pair distances are stable on the new geography."""

from __future__ import annotations

import pickle
import sys

from pie_sim.analysis import (
    IE_BRANCH_SAMPLES,
    cognate_distance_matrix,
    extract_branch_lexicons,
)
from pie_sim.scenarios import ANATOLIAN, STEPPE


def report(scenario, pickle_path):
    with open(pickle_path, "rb") as fh:
        pop = pickle.load(fh)
    extracted = extract_branch_lexicons(pop, IE_BRANCH_SAMPLES)
    D, names = cognate_distance_matrix(extracted)
    idx = {n: i for i, n in enumerate(names)}

    pairs = [
        ("Gaulish", "Gaelic"),
        ("Latin", "Gaulish"),
        ("Slavic", "Baltic"),
        ("Tocharian", "Sanskrit"),
        ("Persian", "Sanskrit"),
        ("Persian", "Armenian"),
        ("Greek", "Albanian"),
        ("Greek", "Latin"),
        ("Germanic", "Gaulish"),
    ]
    print(f"\n=== {scenario.name} ===")
    for a, b in pairs:
        if a in idx and b in idx:
            print(f"  {a:9s} - {b:9s}: {D[idx[a], idx[b]]:.3f}")
    if "Hittite" in idx:
        print(f"  Hittite mean dist to others: {D[idx['Hittite']].mean():.3f}")


if __name__ == "__main__":
    report(STEPPE, "data/steppe_final.pkl")
    report(ANATOLIAN, "data/anatolian_final.pkl")
