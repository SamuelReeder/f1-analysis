# f1-analysis

Separates F1 **driver skill** from **car-package performance** using qualifying lap
times, and produces continuously updatable ratings with honest uncertainty.

Stage 1: **one-lap qualifying pace**, 2010 onward. Stage 2 (see Racing below): racing
qualities, each tested on held-out races, and an overall rating as an equal-car
championship; the approach is in `docs/racing_approach.md`.

## Dashboard

**F1 Analysis** is a local React/TypeScript dashboard in `dashboard/` using real
published model estimates.

The interface uses a pure black background (`#000000`), Formula 1 red (`#e10600`)
and light text.
Team markers use the current colours from [Formula 1’s team directory](https://www.formula1.com/en/teams),
recorded on 2026-09-30 in `dashboard/src/lib.ts` and keyed by team lineage.
Historical entries retain that lineage palette. Chart comparisons use red and
white, with a dashed second series, to stay distinct even for teammates.
The Methodology section in Model health describes the qualifying model, rating
definitions, uncertainty, validation limits and publication process.

The Race pace view reads a separate checked export from `outputs/race_total/`.
Its driver and car tables are gated independently; an absent or failed result is
shown as pending or withheld. The existing qualifying tables and comparisons keep
their own metric and data cutoff.

It provides:

- Driver and car rankings, 90% pace and rank intervals, fastest/top-three
  probabilities, searchable tables, team filters, entry details and CSV downloads.
- Driver and car history, including former drivers and team lineages, with season
  selection, uncertainty bands and optional circuit-adjusted car pace.
- Head-to-head pace differences and a field-wide probability matrix. Differences
  use joint posterior samples, preserving dependence between estimates.
- Model health: data cutoff, publication and convergence checks, refresh status and
  errors, per-quality readiness, clearly marked historical validation, source
  fingerprints, and an archive of as-published estimates.

Start it from the repository root (Node 18+ and the existing Python environment):

```bash
.venv/bin/python -m f1rank.export             # creates the verified export manifest
.venv/bin/python -m f1rank.dashboard publish  # packages existing results, no refit
npm --prefix dashboard ci
npm --prefix dashboard run build
.venv/bin/python -m f1rank.dashboard serve    # http://localhost:4173
```

The local server reads published data directly, so **publishing new results does not
require rebuilding the interface**. For interface development, use
`npm --prefix dashboard run dev`; it reads the same published data. The UI checks for
a new release every 30 seconds, retaining the previous release if the check fails.
It does not fetch new F1 data or start model fitting from the browser.

After a qualifying session, this runs the complete qualifying update (potentially
tens of minutes for the fit):

```bash
.venv/bin/python -m f1rank.dashboard refresh
```

That command runs `fetch`, `build`, `fit`, `export`, then `publish`. Once a race export
exists, it also revalidates and exports race pace against the updated entry and event
tables, reusing unchanged fits. Add `--race-python .venv-gpu/bin/python` to use the
GPU environment for that stage. A process lock prevents overlapping refreshes.
Failed fits or stale exports cannot replace the
dashboard release. It records stage, elapsed time and errors in
`dashboard/public/data/status.json`, a run history in `outputs/dashboard/runs.jsonl`,
and fitting output in `outputs/dashboard/refresh.log`. The local server also detects
a refresh process that exited without recording its final status. Refreshes are
explicit; no model-refresh scheduler has been configured.

After a race, include the new timing data and timeline in the same update:

```bash
.venv/bin/python -m f1rank.dashboard refresh --races \
  --extract-python .venv-fastf1/bin/python --race-python .venv-gpu/bin/python
```

This extracts uncached race timing with FastF1, combines the tables, rebuilds the
incident timeline, then fits and exports both qualifying and race pace before
publishing. Omitting the environment options uses the current Python environment.
Race fitting can take hours; completed, unchanged validation fits are checkpointed.

Dashboard data uses a versioned JSON contract: `latest.json` points to an immutable,
content-addressed file in `dashboard/public/data/releases/`. The pointer changes only
after the complete release has been written. The underlying qualifying export has a
manifest covering its inputs, model/export code, fit metadata and every consumed
output. Partial exports and changed inputs are rejected. The full posterior fit is
checked by `export`, while the browser only receives summaries and pairwise contrasts.
The new data files are generated locally and ignored by Git. A static build includes
the publications available at build time; a static host must receive updated `data/`
files to show subsequent releases. Serve `latest.json` and `status.json` without caching.

The total driver race-pace ranking is published; the car race-pace table is withheld
because it did not pass the prediction and interval-coverage checks. Overall race-result
and equal-car championship standings remain unavailable until the corrected racing
pipeline is rerun.
Portable skill is explicitly experimental. Historical validation files have no run
manifest and are shown as recorded research, not as fresh acceptance of a new fit.

Verification:

```bash
.venv/bin/python -m pytest -q
npm --prefix dashboard run build
cd dashboard
npx playwright install chromium              # once, for browser tests
npm test -- --workers=2
```

Browser tests cover interactions, exports, release adoption and failure handling,
mobile overflow, browser errors, and automated accessibility checks on all five
views. `PLAYWRIGHT_CHROMIUM_EXECUTABLE` optionally selects an existing Chromium
binary. The Python checks cover publication atomicity, corruption, process locking,
missing provenance, stale racing outputs and statistically correct comparisons.

### Publish on push with GitHub Pages

`.github/workflows/dashboard.yml` checks and deploys the dashboard when relevant
changes reach `main`. Pull requests to `main` run the same checks without publishing.
The workflow can also be started manually from the Actions tab on `main`.

One-time setup in [repository Pages settings](https://github.com/SamuelReeder/f1-analysis/settings/pages):
choose **GitHub Actions** under **Build and deployment → Source**. Then merge or push
the dashboard, workflow, checked ratings and supporting files to `main`. The expected
site address is **https://samuelreeder.github.io/f1-analysis/**. Preparing the workflow
locally does not enable Pages or publish the uncommitted dashboard.

Each deployment verifies the committed export manifest, builds a browser dataset,
runs publisher checks, builds the site, and runs browser/accessibility tests against
the static build under `/f1-analysis/`. Only the checked `dashboard/dist/` artifact
is deployed. A failed build or check leaves the previous deployed site in place.
GitHub's built-in deployment token is used; no personal token or hosting secret is
needed. Versions of the official Actions are pinned to verified commit hashes.

**New data and model fitting are separate from website deployment.** Run
`python -m f1rank.dashboard refresh --races` (with the Python environment options above)
on the fitting machine after a race weekend,
then commit the updated `data/processed/` model inputs, `outputs/ratings/` (including
`manifest.json` and `fit_metadata.json`), checked `outputs/race_total/` exports and
validation files, and new `outputs/snapshots/` entries. Once
those reach `main`, the workflow publishes the new rankings. A UI-only push reuses
the existing verified rankings. Changes to model code or data require a matching
export; a stale manifest deliberately blocks deployment.

The portable `outputs/ratings/fit_metadata.json` preserves the metadata of the fit
used by the export. GitHub Actions needs only the lightweight dependencies in
`dashboard/requirements.txt`; it neither refits models nor requires the ignored
`outputs/fits/` posterior directory. The live site's Model health page describes
that published dataset. Deployment failures appear in the repository's Actions tab;
an unsuccessful deployment cannot update the already-live site's status panel.

On a static host, publishing is a deployment. Browser checks still run every 30
seconds, though GitHub Pages caching can delay visibility of a new release.

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
- **Snapshots.** Each export writes `outputs/snapshots/<event>_<model>_<fit id>_export2.json`
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
| `extract/` | FastF1 extraction (own environment): qualifying-time fill, race and sprint timing tables, telemetry summaries |
| `racedata.py` | Combines extracted race tables into `data/processed/race_*.parquet` |
| `timeline.py` | Race event timeline with evidence and cause probabilities; audit |
| `oldlaps.py` | Jolpica lap-by-lap data for 2010-2017 (database dump), with inferred neutralisations and pit stops |
| `powerunits.py`, `conditions.py` | Power-unit suppliers per team-season; wet-race proxy (precipitation) and traffic |
| `racepace.py`, `racemulti.py` | Race pace and degradation: stage A per race; one joint lap-level model across seasons |
| `racemodel.py`, `racejoint.py` | The earlier two-stage path and its joint-model check (not accepted) |
| `reliability.py` | Retirements as competing risks: mechanical reliability (team-season, supplier), driver error rates |
| `pitstops.py` | Pit stops as a team operations rating |
| `consistency.py`, `wetpace.py` | Lap-time consistency; wet-weather pace (experimental) |
| `benchmark.py` | Simple results benchmark (rank-ordered logit on finishing orders) |
| `firstlap.py` | Positions gained on lap 1 |
| `battles.py`, `overtaking.py` | Battle episodes; pass model with a synthetic feasibility check |
| `championship.py` | Equal-car championship simulation and the entry test for racing qualities |
| `racereport.py` | The Racing section of `outputs/REPORT.md` |

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

## Racing (stage 2)

**2026-09-30 review update:** the recorded racing results below describe the earlier
pipeline. Its qualifying features used later qualifying data, and its championship
entry tests tested qualities individually. Those validation results and equal-car
standings need regeneration under the corrected procedure. They are not evidence that
the new gates pass. The qualifying headline remains available; its exports have been
updated. See **Review fixes and regeneration** below.

Approach, decisions fixed before the final runs, and build status:
`docs/racing_approach.md`. Results and gates: `outputs/REPORT.md` (Racing). Race outcomes
run from 2010; lap timing from 2018 (FastF1) and, through a weaker observation model (no
tyre compounds, speed traps or track status), from 2010 (Jolpica).

1. **Race data.** FastF1 race and sprint timing (laps, race-control messages, track status,
   weather, classification) per race, combined into `data/processed/race_*.parquet` and
   `sprint_*.parquet` with each race's FastF1 version, retrieval time and table hashes in
   `race_sources.json`; per-lap lift-and-coast seconds from FastF1 car data
   (`race_telemetry.parquet`). Jolpica lap-by-lap data for 2010-2017 (`oldlaps.py`, from
   Jolpica's delayed CSV dump, sha256 checked), with neutralisations, 2010 pit stops and
   gaps at the line inferred and checked against FastF1. Power-unit suppliers per
   team-season (`powerunits.py`, Wikipedia entry lists), wet races before 2018 from hourly
   precipitation (`conditions.py`, Open-Meteo; a weak proxy), traffic exposure per car.
2. **Event timeline** (`timeline.py`): one row per thing that happened in a race, with its
   evidence and, for retirements, cause probabilities. Reviewed causes go in
   `data/overrides/timeline_overrides.csv`. Audit: `outputs/timeline/AUDIT.md`.
3. **Qualities**, each with a held-out test against a baseline that keeps the car and
   context terms: race pace and degradation (`racepace.py` stage A, `racemulti.py` joint
   model), reliability and errors (`reliability.py`), pit stops as team operations
   (`pitstops.py`), consistency (`consistency.py`), wet pace (`wetpace.py`, experimental),
   first-lap performance (`firstlap.py`), overtaking (`battles.py`, `overtaking.py`).
4. **Overall rating** (`championship.py`): an equal-car championship simulated from
   qualifying, the grid, a race stage fitted on real finishing orders, and retirements. A
   racing quality enters the race stage only if it improves held-out finishing orders (the
   entry test), whether or not it has its own ranking. `benchmark.py` is the simple results
   benchmark.

**What the earlier gates said** (historical results, pending regeneration):
- **Pit stops** pass as a team operations rating.
- **Reliability** (driver error rates, team-season and power-unit supplier-season
  mechanical effects), **consistency** and **wet pace** do not improve held-out prediction.
- **First-lap performance:** with a lasting team term (the model fixed in advance), the
  driver effects do not improve held-out prediction; the team term does. No first-lap
  driver ranking.
- **Overtaking and defending** cannot be rated: attacker and defender effects are not
  recovered at real sample sizes in the synthetic check, so overtaking is reported as
  counts only.
- **Race-specific pace and degradation** both pass (held out 2012-2026): race pace beyond
  what qualifying predicts, and the lasting change in pace with tyre age, improve held-out
  teammate gaps. Split by era (descriptive, after the gate), degradation's improvement comes
  from 2012-2017, when compounds are unknown, and race-specific pace's from 2018 on.
- **Entry into the overall rating** (race stage with vs without the quality, held out
  2012-2026): race-specific pace enters (+0.246 log predictive density per race; 95% interval
  +0.156 to +0.341) and so does first-lap performance (+0.049; +0.017 to +0.086), although
  its standalone ranking fails. Degradation (-0.002; -0.009 to +0.005) and consistency
  (+0.030; -0.002 to +0.062) do not enter; overtaking has no held-out draws.

FastF1 requires pandas < 3, so extraction runs in its own environment; the lap-level
race-pace model runs on a GPU (`.venv-gpu`, JAX with CUDA) and falls back to CPU (slow).
Long fits are best run one or two at a time: the machine this was built on has 19 GB.

```bash
python3 -m venv .venv-fastf1 && .venv-fastf1/bin/pip install fastf1 pyarrow
.venv-fastf1/bin/python extract/race_extract.py --first 2018            # ~15 s per race (API limits)
.venv-fastf1/bin/python extract/race_extract.py --first 2021 --sprint   # sprint races
.venv/bin/python -m f1rank.racedata          # combine into data/processed/race_*.parquet
.venv/bin/python -m f1rank.oldlaps download  # Jolpica's delayed CSV dump (sha256 checked)
.venv/bin/python -m f1rank.oldlaps dump      # check the dump against the API pages
.venv/bin/python -m f1rank.oldlaps           # jolpica_laps / jolpica_pitstops
.venv/bin/python -m f1rank.oldlaps check     # inferences against FastF1
.venv/bin/python -m f1rank.powerunits        # data/reference/power_units.csv
.venv/bin/python -m f1rank.conditions        # wet proxy and traffic
.venv/bin/python -m f1rank.timeline          # event timeline + audit
.venv/bin/python -m f1rank.racepace stage-a && .venv/bin/python -m f1rank.racepace stage-a-old
.venv-fastf1/bin/python extract/telemetry.py # coasting per lap, for stage A's races
.venv/bin/python -m f1rank.racedata          # again, to combine the telemetry
.venv-gpu/bin/python -m f1rank.racemulti --heldout   # race pace: held-out test (decides the gate)
.venv-gpu/bin/python -m f1rank.racemulti             # full fit: summary and driver terms
.venv-gpu/bin/python -m f1rank.racemulti --heldout --fastf1-only      # variants
.venv-gpu/bin/python -m f1rank.racemulti --heldout --telemetry-races
.venv-gpu/bin/python -m f1rank.racemulti --heldout --coast
.venv/bin/python -m f1rank.reliability
.venv/bin/python -m f1rank.pitstops
.venv/bin/python -m f1rank.consistency
.venv/bin/python -m f1rank.wetpace
.venv/bin/python -m f1rank.benchmark
.venv/bin/python -m f1rank.firstlap
.venv/bin/python -m f1rank.battles episodes && .venv/bin/python -m f1rank.battles feasibility \
  && .venv/bin/python -m f1rank.battles fit
.venv/bin/python -m f1rank.championship      # needs the quality modules' draws (not committed)
.venv/bin/python -m f1rank.report            # adds the Racing section to outputs/REPORT.md
```

The earlier two-stage race-pace path (`racepace.py stage-b`, `racejoint.py`) is kept for the
record; its driver estimates were not accepted by the joint-model check.

`analysis/race_signal.py` is the earlier reproducible check of the racing signal in the
same data (results in `outputs/analysis/race_signal/`):

```bash
.venv-fastf1/bin/python analysis/race_signal.py --first 2018 --last 2026
```

## Review fixes and regeneration (2026-09-30)

- **Historical qualifying features:** `qualifying.py` loads a checked qualifying fit
  trained strictly before each held-out season. Both training and test features in
  that fold use that fit. Test features are **season-ahead forecasts conditional on
  entrants and circuits**, not estimates updated after the weekend's qualifying.
  This is a conservative, reproducible validation policy; it does not measure the extra
  benefit of within-season qualifying updates. There is no fallback to the main fit.
  Current/full-data estimates still use the checked main fit.
- **Combined championship:** backward removal tests each quality against the combined
  model with that quality removed, then retests survivors. An outer held-out season
  chooses qualities using at least three earlier inner test seasons. The complete
  selection procedure must improve on the base race model in an outer test before
  racing qualities enter. Intervals resample whole seasons. Qualities from one fit
  retain their shared posterior draw indices when scored together.
- **Race cache:** the fingerprint includes every actual likelihood input, driver
  identifiers and model source. Tyre, compound and traffic corrections invalidate it.
- **Publication diagnostics:** all posterior parameters supplied by a fit must have
  finite draws and R-hat < 1.05 (structurally constant coordinates excluded), with at
  most one divergence per 1,000 draws. Failed fits cannot publish. The qualifying-fold
  preparer and joint race fitter have fixed retries; other failures stop for investigation.
- **Racing provenance:** generated manifests identify the run, input data and code,
  output hashes, and the data cutoff. Downstream models and the racing report reject
  missing, changed or partially written artifacts. Legacy results remain on disk for
  reference; they are excluded from the current report until regenerated. Manifests
  conservatively invalidate on changes anywhere in the racing inputs or model package.
- **Dashboard metric names:** `pairwise_drivers.csv` now refers to the headline
  in-team metric. Explicit `pairwise_drivers_in_team.csv` and
  `pairwise_drivers_portable.csv` are also provided. `current_draws.npz` contains both
  `in_team_s` and `portable_skill_s`; `skill_s` remains a legacy alias for the latter.
  IDs are stored as strings and load without pickle. Export version 2 adds a new
  immutable snapshot without overwriting a version-1 publication.
- **Scope:** the older `racemulti` export `u` remains a lasting career-level effect
  beyond qualifying. The new `race_total_model` estimates total dry-race driver
  and car pace independently of qualifying; its own validation decides whether
  either table can publish. The equal-car scenario and portable skill remain experimental.

### Total race-pace dashboard export

`race_total_model.py` fits clean dry-race laps from 2018 onward in one likelihood.
It uses known tyre compounds, age, lap number and traffic, with Student-t errors
correlated within stints. Race entries come from the race classification rather
than qualifying, so a driver without a qualifying time is retained.

The driver estimate combines a lasting effect with current-season form. The car
estimate is a team-season effect. Independent race-day deviations are integrated
out of both tables. Ratings are at tyre age 10 laps and centred on the eligible
current field; they update when new races are fitted but are season estimates,
not a development trajectory or finishing-order model. At least two clean races
in the current season are required for an entry in a table.

The fixed validation windows end training at 2024 round 10, 2025 round 10 and
2026 round 7. Later dry races of each season are excluded from fitting. The
no-driver and no-car models are refitted with all context terms retained. Driver
validation compares teammate gaps; car validation compares team-average observed
pace, without using the fitted driver rating to construct its target. A table
requires lower squared error (race-block bootstrap 95% interval below zero),
85–95% coverage of 90% prediction intervals, and narrower intervals than its
baseline. These prediction tests do not establish a causal separation of innate
skill, strategy and machinery. Team-priority and fuel-load differences remain
possible confounders. Test-race targets are regenerated from the clean laps and
cached separately. Resampling whole bootstrap rows preserves their shared
measurement uncertainty across drivers and teams.

After refreshing the FastF1 race tables and event timeline, run:

```bash
.venv-gpu/bin/python -m f1rank.race_total --validate --export
.venv-pages/bin/python -m f1rank.dashboard publish
```

This fits and checkpoints the full model and the three versions of each historical
fold. Cache identities include all likelihood arrays, identifier ordering, model
source and sampler settings. All sampled parameters, including nuisance terms,
must pass convergence checks. A failed fit leaves the previous published files
intact. The portable `pace.json` and its manifest contain the checked estimates,
gate evidence and fit identities; the publisher and Pages build verify these
without needing JAX or the ignored posterior caches. `research_estimates.json`
holds values for investigation even when a table fails its publication gate and
is never included in the dashboard payload.

Commit the code, validation outputs and portable export together, then push to
`main` to run the existing checked Pages deployment. Merely polling the website
does not refit the models. The full race fit is a substantial GPU job; historical
fits are reused on later runs when their training data have not changed.

The first export contains 22 current driver estimates from 122,648 clean laps in
139 dry races. All ten real-data fits passed the convergence checks with zero
divergences (maximum R-hat 1.034). The three historical windows contain 29 test races:

| Prediction target | RMSE | Baseline RMSE | 90% interval coverage | Ranking |
| --- | ---: | ---: | ---: | --- |
| Teammate driver pace gaps (262 comparisons) | 0.328s | 0.413s | 90.8% | Published |
| Team-average pace (295 predictions) | 0.346s | 0.341s | 84.1% | Withheld |

Errors are seconds per 90-second lap. Driver improvement is supported by the
race-block bootstrap; car improvement is not. The car model also misses the
85% minimum coverage requirement. Its computed estimates remain in the ignored
research file, not the public ranking. These results are prediction checks, not
proof that all differences in strategy, equipment or team priority have been separated.

The corrected qualifying files can be regenerated without refitting the valid main fit:

```bash
.venv/bin/python -m f1rank.export
.venv/bin/python -m pytest -q
```

Racing regeneration is a long research run. Keep data and code fixed during it. Check
or prepare the qualifying folds first; existing valid folds are reused, missing or
nonconverged ones are refitted with a recorded retry rule. New data extending a test
season requires that season's qualifying fold to cover the added event.

```bash
.venv/bin/python -m f1rank.qualifying --seasons 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
.venv-gpu/bin/python -m f1rank.racemulti --heldout
.venv-gpu/bin/python -m f1rank.racemulti
.venv/bin/python -m f1rank.firstlap
.venv/bin/python -m f1rank.consistency
.venv/bin/python -m f1rank.reliability
.venv/bin/python -m f1rank.pitstops
.venv/bin/python -m f1rank.wetpace
.venv/bin/python -m f1rank.benchmark
.venv/bin/python -m f1rank.championship
.venv/bin/python -m f1rank.report
```

Regenerate optional battle draws with `f1rank.battles fit` if they exist, and rerun race
pace variants with their existing flags if their comparison tables are wanted. Commands
must stop on failure; stale files must not be relabelled as validated. Historical
as-published charts should read snapshots; the series files are revised using all data.
