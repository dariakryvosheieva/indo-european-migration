# Modeling the Indo-European Divergence

A computational model of the divergence of Proto-Indo-European (PIE) into the
major branches of the Indo-European (IE) language family, driven by
population migration on an explicit Eurasian geography. The same simulator
runs under different homeland hypotheses (Steppe vs. Anatolian) so the two
can be compared head-to-head.

Final project for **24.918 Workshop in Linguistic Research**.

## What's here

- `pie_sim/cognates.py` — stochastic-Dollo lexical evolution with innovation
  and contact-mediated borrowing. Calibrated against glottochronological
  rates (~14% lexical replacement per millennium). Lexicons span 1500
  meaning slots — at the conservative end of scholarly estimates of
  reconstructable PIE roots (Pokorny ~1800; Mallory & Adams ~1500-2000;
  Watkins ~1000-1500). Larger meaning sets reduce the standard error of
  each pairwise distance from ~0.029 (Swadesh-200) to ~0.011, which
  markedly stabilizes the reconstructed tree topology.
- `pie_sim/geography.py` — anatomically-improved Eurasia grid (1° cells)
  encoding the actual IE migration corridors and barriers:
  - **Migration corridors** — Pontic-Caspian + Kazakh steppes, the
    Bactria/BMAC corridor east of the Caspian, the Iranian plateau
    corridor, the Dzungar gap north of the Tien Shan, the Bosphorus
    land bridge, the Anatolian plateau, the Punjab corridor (the
    narrow temperate strip between the Himalayas and the Thar
    Desert that funnels the Indo-Aryan migration south into the
    Gangetic plain).
  - **Mountain barriers** — Caucasus, Alps, Carpathians, Pyrenees,
    Apennines, Pindus (Greek), Dinaric + Balkan/Rhodope (Balkans),
    Hindu Kush, Pamir, Karakoram, Tien Shan, Elburz, Pontic + Taurus,
    Himalayas, Yemen highlands, Western Ghats.
  - **Deserts** — Karakum, Kyzylkum, Taklamakan, Gobi, Iranian central,
    Syrian, Arabian (incl. Yemen lowlands), Thar (NW India).
  - **Coastlines** — Bay of Biscay separating Iberia from France; the
    Italian boot flanked by Tyrrhenian (W) and Adriatic (E); the Greek
    peninsula carved out by the Aegean; the British Isles disconnected
    from the continent. The Arabian peninsula is bounded by the Red
    Sea (extending south to Bab-el-Mandeb), Persian Gulf (extending
    east through the Strait of Hormuz), and Arabian Sea. The Indian
    peninsula has its own west coast (Arabian Sea sits between Arabia
    and India) and east coast (Bay of Bengal reaching the Chennai
    meridian).
  - **Narrow straits (`Terrain.STRAIT`)** — English Channel, Irish Sea,
    Strait of Otranto. Friction is set to 3 (matching desert — a real
    but moderate barrier) and capacity is tiny (0.10), modeling the
    well-attested fact that Bronze and Iron Age peoples crossed these
    waters by boat — slowly, but routinely. The cells are passable
    enough that Britain founds within ~1500 years of the wave reaching
    northern France, and they carry enough population to act as a real
    borrowing channel between the two coasts they connect (Britain ↔
    Brittany, Italy ↔ Albania). Without them, a cleanly-disconnected
    Britain could never be settled and Gaelic could never be sampled.
- `pie_sim/population.py` — metapopulation with logistic growth, friction-
  weighted migration, and capacity-scaled founder events that let
  low-capacity terrain (desert, mountain) be settled at all.
- `pie_sim/scenarios.py` — pre-defined homeland hypotheses (`STEPPE`,
  `ANATOLIAN`) plus a cached runner. Each scenario can attach
  `IsolatedBreakaway` cells: small populations that split off from the
  homeland, sit pinned in their starting cell (no migration, no contact
  borrowing), and optionally pre-evolve their lexicon for a configurable
  number of years before the main run starts. We use this to encode the
  empirical fact that the Anatolian branch left PIE several centuries
  before the Yamnaya horizon, so it should accumulate more lexical
  drift than late-splitting branches do.
- `pie_sim/analysis.py` — extracts lexicons at the historical attestation
  locations of 13 IE branches, builds pairwise cognate-distance matrices,
  and reconstructs UPGMA trees.

## Demos

```bash
python scripts/demo_geography.py            # render the stylized map
python scripts/demo_cognates.py             # cognate divergence on its own
python scripts/demo_steppe_expansion.py     # population sweep snapshots
python scripts/demo_branch_analysis.py --scenario steppe
python scripts/demo_branch_analysis.py --scenario steppe --no-pre-evolve-hittite
python scripts/demo_branch_analysis.py --scenario steppe --pre-evolve-tocharian
python scripts/demo_branch_analysis.py --scenario anatolian --pre-evolve-tocharian
python scripts/demo_branch_analysis.py --scenario anatolian
python scripts/demo_compare_scenarios.py    # side-by-side trees + maps
```

The first run of a scenario takes ~4-5 minutes (a 5000-year simulation at
`dt=10y` over 1500 meaning slots) and is then cached under
`data/{scenario}_final.pkl` (or
`data/steppe_no_hittite_pre_evolve_final.pkl`
when `--no-pre-evolve-hittite` is used, and
`data/{scenario}_with_tocharian_pre_evolve_final.pkl`
when `--pre-evolve-tocharian` is used). Borrowing is always enabled for
`demo_branch_analysis.py`. Cached runs replot the figures in
seconds.

## What the comparison shows

Both scenarios produce coherent but distinct topologies (see
`figs/compare_scenarios.png`):

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

  | Step | Pair | Height | Comment |
  |---|---|---|---|
  | 1 | **Gaulish-Gaelic** | **0.611** | **Celtic ✓** |
  | 2 | Germanic joins (Gaulish, Gaelic) | ≈0.648 | Northwest IE / Germano-Celtic ✓ |
  | 3 | **Greek-Albanian / Greek-Latin** | **0.651** | Paleo-Balkan / Mediterranean ✓ |
  | 4 | Tocharian-Sanskrit | 0.689 | Andronovo eastern spread |
  | 5 | **Slavic-Baltic** | **0.699** | **Balto-Slavic ✓** |
  | 6 | Persian-Armenian | 0.721 | Iranian + adjacent Caucasian |
  | … | further internal joinings | 0.74-0.79 |  |
  | last | Hittite joins everything else | 0.82 | **Anatolian basal ✓** |

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
  because Italy is reached primarily *via the Balkans* (Strait of
  Otranto + Adriatic shipping) under the Steppe wave, so Latin shares
  recent ancestry with the Mediterranean cluster.
- **Anatolian (farming dispersal)** — radiates outward through the
  Bosphorus into the Balkans and across Anatolia into the Levant, Iran,
  and (more slowly) the steppe and India. The reconstructed tree is
  geographic-radial: Hittite, Armenian, Persian, and Tocharian sit
  close to the seed; Mediterranean Europe (Greek, Albanian, Latin) in
  the middle ring; the Northern European group (Slavic, Baltic,
  Germanic, Gaulish, Gaelic) on the outside. Hittite is *not*
  recovered as basal (mean distance to others 0.70 — the *least*
  basal branch). The Celtic pair (Gaulish-Gaelic at 0.655),
  Balto-Slavic (0.691), Indo-Iranian-ish (Tocharian-Sanskrit at 0.673),
  and Paleo-Balkan (Greek-Albanian at 0.705, Greek-Latin at 0.708)
  clades all recover, but at the cost of losing the deep Anatolian split.

### Steppe vs. Anatolian: which sub-pairings does each recover?

| Sub-pairing | Steppe | Anatolian | Empirical status |
|---|---|---|---|
| Hittite basal | ✓ (0.82 outgroup) | ✗ (Hittite mean 0.70, least basal) | Canonical |
| Celtic (Gaulish-Gaelic) | ✓ (0.611, strongest pair) | ✓ (0.655, strongest pair) | Canonical |
| Northwest IE (Germano-Celtic) | ✓ (Germanic-Gaulish 0.648) | ✓ (0.722) | Defensible |
| Balto-Slavic | ✓ (0.699) | ✓ (0.691) | Canonical |
| Paleo-Balkan (Greek-Albanian-Latin) | ✓ (0.651) | ✓ (0.705-0.708) | Defensible |
| Tocharian-Sanskrit | ✓ (0.689) | ✓ (0.673) | Plausibly Andronovo-shared |
| Persian-Armenian | (0.721) | (0.750) | Geographic, not standard |
| Italo-Celtic (Latin-Celtic) | ✗ | ✗ | Canonical, hard to recover from migration alone |

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

## Setup

```bash
pip install -e .
pip install -r requirements.txt
pytest tests/
```

## References

- Anthony (2007). *The Horse, the Wheel, and Language*. Princeton.
- Bouckaert et al. (2012). Mapping the origins and expansion of the
  Indo-European language family. *Science* 337, 957–960.
- Chang, Cathcart, Hall & Garrett (2015). Ancestry-constrained phylogenetic
  analysis supports the Indo-European steppe hypothesis. *Language* 91.
- Haak et al. (2015). Massive migration from the steppe was a source for
  Indo-European languages in Europe. *Nature* 522, 207–211.
- Heggarty et al. (2023). Language trees with sampled ancestors support a
  hybrid model for the origin of Indo-European languages. *Science* 381.
- Nicholls & Gray (2008). Dated ancestral trees from binary trait data and
  their application to the diversification of languages. *JRSS B* 70.
- Reich (2018). *Who We Are and How We Got Here*. Pantheon.
- Renfrew (1987). *Archaeology and Language*. Cambridge.
