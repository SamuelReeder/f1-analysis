# f1-analysis

Separates F1 **driver skill** from **car-package performance** using qualifying lap
times, and produces continuously updatable ratings with honest uncertainty.

Stage 1 scope: **one-lap qualifying pace**, 2010 onward. Race pace, reliability and
driver sub-skills are later stages. The proposed approach for racing and an overall
driver rating is in `docs/racing_approach.md`.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scipy pyarrow requests "jax[cpu]" numpyro arviz

.venv/bin/python -m f1rank.fetch --first 2006   # download (cached; current season refreshed)
.venv/bin/python -m f1rank.build                # tidy Parquet tables in data/processed/
.venv/bin/python -m f1rank.fit --warmup 1500 --samples 1500   # main fit (~40 min)
.venv/bin/python -m f1rank.export               # dashboard-ready outputs in outputs/ratings/
.venv/bin/python -m f1rank.jobs synth-source    # freeze the main fit as the synthetic-truth source
.venv/bin/python -m f1rank.jobs all             # validation, sensitivity, placebo fits (~5 h)
.venv/bin/python -m f1rank.evaluate all         # scores in outputs/validation/
.venv/bin/python -m f1rank.compare              # variants vs main model on identical forecasts
.venv/bin/python -m f1rank.diagnostics          # residual checks on the main fit
.venv/bin/python -m f1rank.report               # outputs/REPORT.md
```

Jolpica has no qualifying times for a few events (currently 2025 Miami). They are
rebuilt from FastF1 lap timing by `extract/quali_fill.py`, which also checks the method
against Jolpica on every other event of that season. It needs FastF1, which runs in its
own environment (see Racing below). Its output, `data/supplements/quali_times_fastf1.json`,
is committed, so `build` does not need FastF1.

After each qualifying session: `fetch`, `build`, `fit`, `export`.

- **Fit metadata.** Every fit has a `<name>.meta.json` sidecar with a fingerprint of its
  model inputs, its data date and training cutoff, and identifiers for every state.
  Draws are only ever matched to the design they were fitted on:
  - `export` refuses a main fit made on older data.
  - `evaluate` scores each batch of validation fits on the data they were fitted on.
  - `jobs all` reruns only fits that are missing or stale.
  - `jobs list` shows the status of each fit.
- **Snapshots.** Each export writes `outputs/snapshots/<event>_<model>_<fit id>.json`
  once and never overwrites it, so what was published for each fit is kept.
- **Failures.** A failed validation job makes `jobs` exit with status 1.

| Module | Role |
|---|---|
| `fetch.py` | Jolpica (Ergast-compatible) API client with caching and rate limiting |
| `build.py` | Qualifying times, entries, drivers, race results as Parquet |
| `lineage.py` | Maps rebrands to one team lineage (e.g. Toro Rosso → AlphaTauri → RB) |
| `design.py` | Pace per segment, driver/car state indices, circuit factors |
| `model.py` | The Bayesian model (NumPyro) |
| `fit.py` | NUTS fitting; fit metadata and checks that draws match their design |
| `ratings.py` | Leaderboards, series, rank distributions |
| `export.py` | Dashboard files and write-once snapshots |
| `simulate.py` | Synthetic truth on the real F1 network, with misspecification scenarios |
| `jobs.py`, `evaluate.py` | Leave-future-out forecasting, synthetic recovery, sensitivity, placebo test |
| `compare.py` | Variants (2006 data window; team-specific effect per spell or per era) vs the main model on identical forecast targets |
| `report.py` | `outputs/REPORT.md` and acceptance gates |
| `diagnostics.py` | Residual checks (every fitted term) |
| `extract/` | FastF1 extraction (own environment): qualifying-time fill, race timing tables |
| `racedata.py` | Combines extracted race tables into `data/processed/race_*.parquet` |
| `timeline.py` | Race event timeline with evidence and cause probabilities; audit |
| `racepace.py`, `racemodel.py`, `racejoint.py` | Race pace and degradation (stage A per race, stage B across races, joint-model check) |
| `reliability.py` | Retirements as competing risks: mechanical reliability, driver error rates |
| `benchmark.py` | Simple results benchmark (rank-ordered logit on finishing orders) |
| `firstlap.py` | Positions gained on lap 1 |
| `battles.py`, `overtaking.py` | Battle episodes; pass model with a synthetic feasibility check |
| `championship.py` | Equal-car championship simulation |

## Method

**Observation.** Each lap time in a qualifying segment (Q1/Q2/Q3) becomes
`y = -100·log(time / segment median)`: percent of lap time, positive = faster. A
free intercept per segment absorbs track evolution and the changing field in Q2/Q3,
so only comparisons between cars in the same segment carry information. Laps more
than 5% off the median are dropped as non-representative. For 2006–09, Q3 was run
on race fuel and is excluded.

**Model.**

```
y = segment intercept
  + car development      persistent random walk per team; heavy-tailed steps within a
                         season; partial carryover between seasons; larger resets at
                         regulation changes (2014, 2017, 2022, 2026)
  + car x circuit        team-season loading x circuit factor (slow/street vs high-speed)
  + car weekend effect   shared by both teammates, one event, does not persist
  + car segment effect   shared by both teammates within one segment (run window, track state)
  + driver skill         portable ability: driver level + slow random walk + experience
                         curve + decline after 32
  + team-specific effect one per driver and team lineage (persistent; see below)
  + driver weekend form  one event, does not persist
  + noise                Student-t, with a scale per segment (wet/chaotic sessions down-weighted)
```

Only relative quantities are identified. Car and driver effects are centred on the
field at each event, and the ratings are **relative to the average driver / car
entered at that event**. Every driver entered in qualifying has a state at that event. A
driver with no valid lap there (no time, or only laps more than 5% off) has no
observations, and their rating is carried forward by the skill walk. Driver-vs-car separation comes from teammates (same car)
and from drivers moving between teams. The transient terms stop one-off weekends
from being read as lasting changes.

There are two driver ratings:

- **Pace in the current car (headline):** portable skill plus the team-specific
  effect. This is what teammate comparisons measure directly.
- **Portable skill (experimental):** the part expected to carry over to another
  team.

A placebo test splits stints within one team. It shows the team-specific effect is
associated with the team: SD about 0.16% at a team change, against about 0.02% within
a team. The test cannot say *why*. Car-handling compatibility, team support, role,
adaptation and selection would all look the same, so the effect is not labelled
compatibility.

The model gives one effect per driver and team lineage, even across separate spells
years apart. Two variants test that choice:
- **Per spell:** a return after 3+ events with other teams starts a new effect.
- **Per regulation era:** 2014, 2017, 2022 and 2026 start new ones.

Both are fitted as sensitivity variants and compared on identical forecasts
(`outputs/REPORT.md`).

The circuit factor is a 1-D rank-1 decomposition of team pace deviations, fitted
beforehand (and, for forecasts, only on data before the cutoff). One end is slow,
traction-limited street circuits (Marina Bay, Monaco); the other is high-speed
circuits (Spa, Silverstone, Monza).

## Results and validation

Full results: `outputs/REPORT.md` (`python -m f1rank.report`). Summary as of the 2026
Azerbaijan GP:

- **Forecasting**, 25 leave-future-out cutoffs 2013–2026: the model beats a static
  two-way model, raw teammate gaps and a zero baseline on every target. Teammate gap
  per pairing: RMSE 0.188 s vs 0.245–0.264 s. New pairings: 0.243 s vs 0.264–0.425 s.
  Session-level 90% intervals cover 94%.
- **Synthetic recovery** on the real F1 network, over 8 independent clean truths (each
  with its own hyperparameters from the posterior):
  - Car ratings: correlation 0.99; 90% intervals cover 88% on average (79–92%).
  - Driver skill: 90% intervals cover 90% on average (82–95%).
  - Misspecification scenarios (one shared truth): portable current-grid rank
    correlations stay within the spread between the clean truths.
- **Pace in the current car (headline)** passes its own gates:
  - Current-grid ranking recovered with rank correlation 0.83 on average (0.74–0.92).
  - Rank correlation 0.92 or more against every sensitivity variant.
- **Limits:**
  - **Portable ranking recovery.** The current-grid ranking by portable skill is
    recovered with rank correlation 0.70 on average (0.51–0.89).
  - **Sensitivity.** The portable ranking moves with structural choices: whether the
    team-specific effect is modelled, the 2006 vs 2010 data window, and whether the
    effect restarts at each regulation era. The windows and the variants forecast
    about equally well (one effect per spell slightly better).
  - **Status.** Seven of the nine acceptance gates pass. The two that fail are portable
    skill's, so portable skill is experimental and pace in the current car is the
    headline.

  Read ranks as ranges.

Outputs for the dashboard are in `outputs/ratings/`, and `outputs/snapshots/` keeps what
was published for each fit. Posterior draws (`outputs/fits/`, several GB) are not
committed.

## Racing (stage 2, in progress)

Approach and build order: `docs/racing_approach.md`. Built so far (race timing from 2018;
retirement outcomes from 2010):

1. **Race data layer.** FastF1 race timing (laps, race-control messages, track status,
   weather, classification) is extracted per race and combined into
   `data/processed/race_*.parquet`, with each race's FastF1 version, retrieval time and
   table hashes in `race_sources.json`.
2. **Event timeline** (`f1rank/timeline.py`): one row per thing that happened in a race
   (neutralisations, retirements, incidents, penalties, off-track moments, pit anomalies,
   suspected damage, possible team orders), with its evidence and, for retirements, cause
   probabilities. Reviewed causes go in `data/overrides/timeline_overrides.csv`. Audit:
   `outputs/timeline/AUDIT.md`.
3. **Race pace and degradation** (`f1rank/racepace.py`, `f1rank/racemodel.py`): per-race
   estimates from clean laps, then a model across races linked to qualifying pace, with a
   held-out test of whether race pace adds anything beyond qualifying. Results in
   `outputs/REPORT.md` (Racing) and `outputs/race/`.
4. **Reliability and errors, results benchmark, first-lap performance, overtaking and an
   equal-car championship** (`reliability.py`, `benchmark.py`, `firstlap.py`,
   `battles.py`/`overtaking.py`, `championship.py`).

**What the gates say so far** (numbers in `outputs/REPORT.md`, Racing):
- **Race pace follows qualifying pace.** A race-specific pace skill, tyre degradation
  differences and driver error rates do not improve held-out predictions.
- **The race-pace shortcut** failed its agreement check against a joint lap-level model, so
  its driver estimates are not published. A held-out test with the joint model gives the same
  conclusion.
- **First-lap gains** improve held-out predictions for drivers who stay in a team, but not
  for drivers in a new team. So no first-lap driver ranking is published.
- **Overtaking and defending** cannot be rated at real sample sizes. The synthetic
  feasibility check fails for defender effects, so overtaking is reported as counts only.
- **Overall rating: the equal-car championship.** It combines qualifying pace (stage 1)
  with a grid-to-finish model and average reliability. No other driver-specific racing
  quality has passed its gates yet.

FastF1 requires pandas < 3, so extraction runs in its own environment:

```bash
python3 -m venv .venv-fastf1 && .venv-fastf1/bin/pip install fastf1 pyarrow
.venv-fastf1/bin/python extract/race_extract.py --first 2018   # ~15 s per race (API limits)
.venv/bin/python -m f1rank.racedata          # combine into data/processed/race_*.parquet
.venv/bin/python -m f1rank.timeline          # event timeline + audit
.venv/bin/python -m f1rank.racepace stage-a  # per-race pace and degradation
.venv/bin/python -m f1rank.racepace stage-b  # across races, held-out test (needs `export`)
.venv/bin/python -m f1rank.racejoint --season 2024   # two-stage vs joint lap-level check
.venv/bin/python -m f1rank.racejoint --heldout       # held-out test with the joint model
.venv/bin/python -m f1rank.reliability       # competing risks, held-out tests
.venv/bin/python -m f1rank.benchmark         # results benchmark
.venv/bin/python -m f1rank.firstlap          # first-lap performance
.venv/bin/python -m f1rank.battles episodes && .venv/bin/python -m f1rank.battles feasibility \
  && .venv/bin/python -m f1rank.battles fit  # overtaking
.venv/bin/python -m f1rank.championship      # equal-car championship
.venv/bin/python -m f1rank.report            # adds the Racing section to outputs/REPORT.md
```

`analysis/race_signal.py` is the earlier reproducible check of the racing signal in the
same data (results in `outputs/analysis/race_signal/`):

```bash
.venv-fastf1/bin/python analysis/race_signal.py --first 2018 --last 2026
```
