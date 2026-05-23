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
├── tests/                              # Unit tests
└── 24_918_Final_Paper.pdf              # Project report
```

## What the comparison shows

Both scenarios produce coherent but distinct topologies:

- **Steppe (Yamnaya / Pontic-Caspian)** — sweeps west and east along the
  steppe corridor, descending into Europe through the Carpathian/Balkan gap, into Iran and India through Bactria, and into the Tarim Basin through Dzungaria. With the Anatolian and Tocharian early breakaways (1500 and 500 years of pre-simulation drift respectively), the reconstructed UPGMA tree recovers essentially the full canonical IE phylogeny, except a few small issues.

- **Anatolian (farming dispersal)** — radiates outward through the
  Bosphorus into the Balkans and across Anatolia into the Levant, Iran,
  and (more slowly) the steppe and India. Hittite is not recovered as basal due to its location at the origin of dispersal; the Celtic, Balto-Slavic, and Indo-Iranian clades still recover.

The Anatolian scenario is less accurate overall, mainly due to its inability to recover Hittite as a basal outgroup. Further, language evolution in the Anatolian scenario converges very early resulting in very large cognate distances between descendant languages, which suggests that 7000 years (compared to the Steppe scenario's 4000 years) is an implausibly long time horizon. These findings contribute to resolving the long-standing debate regarding the original homeland of Proto-Indo-Europeans, providing evidence in favor of a steppe homeland.
