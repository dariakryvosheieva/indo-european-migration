# Modeling the Indo-European Expansion

A computational model of the divergence of Proto-Indo-European (PIE) into the major branches of the Indo-European (IE) language family, driven by
population migration on a Eurasian geography. The same simulator runs under different homeland hypotheses (Steppe vs. Anatolian) to compare and evaluate them.

Final project for **MIT 24.918 Workshop in Linguistic Research**.

## Quick Start

Clone the repo and configure the package:

```bash
git clone https://github.com/dariakryvosheieva/indo-european-migration.git
pip install -e .
pip install -r requirements.txt
```

Run the Steppe scenario:

```bash
python scripts/demo_branch_analysis.py \
  --scenario steppe \
  --n-years 4000 \
  --pre-evolve-hittite \
  --pre-evolve-tocharian
```

Run the Anatolian scenario:

```bash
python scripts/demo_branch_analysis.py \
  --scenario anatolian \
  --n-years 7000
```

Each scenario takes about 1 minute per 1000 years (~4 mins for Steppe, ~7 mins for Anatolian). At the end of the simulation, cache will be saved to the `data/` folder for subsequent reuse (to avoid having to re-run each time). To force cache overwrite, use the `--force` flag.

## What's here

```
indo-european-migration/
├── data/                               # Simulation cache
├── figures/                            # Figures (produced by code in scripts/)
├── pie_sim/
│   ├── analysis.py                     # Lexical distance matrix and dendrogram
│   ├── cognates.py                     # Lexicon, innovation, and borrowing
│   ├── geography.py                    # Eurasia grid
│   ├── population.py                   # Population and migration
│   └── scenarios.py                    # Defaults (starting location, number of years, etc.) for the Steppe and Anatolian scenarios
├── pie-sim.egg-info/                   # Package metadata
├── scripts/
│   ├── demo_branch_analysis.py         # Main simulation script
│   ├── demo_branch_sample_locations.py # Plots sample locations for descendant languages, as well as the Steppe and Anatolian starting locations
│   ├── demo_divergence.py              # Plots lexical similarity as a function of time
│   ├── demo_geography.py               # Plots the stylized Eurasia map
│   └── demo_steppe_expansion.py        # Plots population by cell as a function of time
└── tests/                              # Unit tests
```

## What the comparison shows

Both scenarios produce coherent but distinct topologies:

- **Steppe (Yamnaya / Pontic-Caspian)** — sweeps west and east along the
  steppe corridor, descending into Europe through the Carpathian gap, into
  Iran via the BMAC corridor, into India through Bactria, and into the
  Tarim through Dzungaria. With the Anatolian-early breakaway pinned
  in central Anatolia (1500 years of pre-simulation drift, matching the
  archaeological gap between the Anatolian split ~5000 BCE and the
  Yamnaya horizon ~3500 BCE), and with two attestation locations chosen
  on linguistic rather than naively-geographic grounds —
  **Gaulish** at Pas-de-Calais (50°N, 2°E — northern Gaul, on the same
  northern Yamnaya wave that reaches Britain) and **Slavic** at the
  Pripet / Polesia marshes (52°N, 27°E — the leading Common Slavic
  urheimat candidate, shared forest-belt zone with Baltic) — the
  reconstructed UPGMA tree recovers essentially the full canonical IE
  phylogeny. Joining order from the linkage matrix:

  | Step | Pair                             | Height    | Comment                         |
  | ---- | -------------------------------- | --------- | ------------------------------- |
  | 1    | **Gaulish-Gaelic**               | **0.611** | **Celtic ✓**                    |
  | 2    | Germanic joins (Gaulish, Gaelic) | ≈0.648    | Northwest IE / Germano-Celtic ✓ |
  | 3    | **Greek-Albanian / Greek-Latin** | **0.651** | Paleo-Balkan / Mediterranean ✓  |
  | 4    | Tocharian-Sanskrit               | 0.689     | Andronovo eastern spread        |
  | 5    | **Slavic-Baltic**                | **0.699** | **Balto-Slavic ✓**              |
  | 6    | Persian-Armenian                 | 0.721     | Iranian + adjacent Caucasian    |
  | …    | further internal joinings        | 0.74-0.79 |                                 |
  | last | Hittite joins everything else    | 0.82      | **Anatolian basal ✓**           |

  This recovers, under a single set of plausible assumptions:
  - **Hittite as the deepest outgroup** at the correct relative depth
    (~0.82 vs ~0.78 within-IE ceiling).
  - **Celtic** as a tight clade (Gaulish-Gaelic, the strongest pair in
    the matrix).
  - **Northwest IE** (Germano-Celtic): Germanic joining the Celtic pair
    next.
  - **Mediterranean / Paleo-Balkan** (Greek + Latin + Albanian).
  - **Balto-Slavic** (Slavic + Baltic).
  - An **eastern clade** (Tocharian + Sanskrit) + (Persian + Armenian).

  The one substantively non-canonical signal is that Latin pairs
  closest with Greek rather than with Gaulish-Gaelic — Italo-Celtic
  (Latin + Celtic) is not recovered. In the model this happens
  because Italy is reached primarily _via the Balkans_ (Strait of
  Otranto + Adriatic shipping) under the Steppe wave, so Latin shares
  recent ancestry with the Mediterranean cluster.

- **Anatolian (farming dispersal)** — radiates outward through the
  Bosphorus into the Balkans and across Anatolia into the Levant, Iran,
  and (more slowly) the steppe and India. The reconstructed tree is
  geographic-radial: Hittite, Armenian, Persian, and Tocharian sit
  close to the seed; Mediterranean Europe (Greek, Albanian, Latin) in
  the middle ring; the Northern European group (Slavic, Baltic,
  Germanic, Gaulish, Gaelic) on the outside. Hittite is _not_
  recovered as basal (mean distance to others 0.70 — the _least_
  basal branch). The Celtic pair (Gaulish-Gaelic at 0.655),
  Balto-Slavic (0.691), Indo-Iranian-ish (Tocharian-Sanskrit at 0.673),
  and Paleo-Balkan (Greek-Albanian at 0.705, Greek-Latin at 0.708)
  clades all recover, but at the cost of losing the deep Anatolian split.

### Steppe vs. Anatolian: which sub-pairings does each recover?

| Sub-pairing                         | Steppe                     | Anatolian                          | Empirical status                                |
| ----------------------------------- | -------------------------- | ---------------------------------- | ----------------------------------------------- |
| Hittite basal                       | ✓ (0.82 outgroup)          | ✗ (Hittite mean 0.70, least basal) | Canonical                                       |
| Celtic (Gaulish-Gaelic)             | ✓ (0.611, strongest pair)  | ✓ (0.655, strongest pair)          | Canonical                                       |
| Northwest IE (Germano-Celtic)       | ✓ (Germanic-Gaulish 0.648) | ✓ (0.722)                          | Defensible                                      |
| Balto-Slavic                        | ✓ (0.699)                  | ✓ (0.691)                          | Canonical                                       |
| Paleo-Balkan (Greek-Albanian-Latin) | ✓ (0.651)                  | ✓ (0.705-0.708)                    | Defensible                                      |
| Tocharian-Sanskrit                  | ✓ (0.689)                  | ✓ (0.673)                          | Plausibly Andronovo-shared                      |
| Persian-Armenian                    | (0.721)                    | (0.750)                            | Geographic, not standard                        |
| Italo-Celtic (Latin-Celtic)         | ✗                          | ✗                                  | Canonical, hard to recover from migration alone |

The central empirical contrast holds: the Steppe scenario plus a
short pre-Yamnaya Anatolian breakaway recovers Hittite as basal at
the correct depth and reproduces all the major canonical sub-families
(Celtic, Balto-Slavic, paleo-Balkan, Northwest IE). The Anatolian
scenario without auxiliary assumptions produces a strictly geographic
clustering that fails to recover Hittite as the deepest outgroup —
the classic objection to the Anatolian hypothesis. Only the Italo-
Celtic clade remains elusive in both, since Italy is reached primarily
via Balkan/Mediterranean routes in our migration substrate. Recovering
Italo-Celtic specifically would likely require additional inference
machinery (e.g., explicit sound-change tracking, sampled-ancestor
priors, or a Bell Beaker secondary radiation) on top of the demic
substrate modeled here.
