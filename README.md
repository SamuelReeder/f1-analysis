# f1-analysis

Separates F1 **driver skill** from **car-package performance** using qualifying lap
times, and produces continuously updatable ratings with honest uncertainty.

Stage 1: **one-lap qualifying pace**, 2010 onward. Stage 2 (see Racing below): racing
qualities, each tested on held-out races, and an overall rating as an equal-car
championship; the approach is in `docs/racing_approach.md`.

## Dashboard

**F1 Analysis** is a local React/TypeScript dashboard in `dashboard/` using real
published model estimates.

The default **Briefing** presents complete ranking tables and primary metrics on
black backgrounds, with factual headings and source dates. Its linked sections
(`#briefing/qualifying`, `#briefing/cars`, `#briefing/race`, `#briefing/overall`,
`#briefing/evidence`) cover driver estimates, machinery, dry-race pace, the
experimental equal-car scenario, and predictive evidence. Tables lead each
section; concise conclusions and optional explanatory detail follow the data.
Category links and ordinary next/previous links connect the sections without
covering the results. Nothing advances automatically.

Every result and conclusion is derived from the loaded release. Race driver/car
gates remain separate, unverified equal-car standings remain withheld, and
forecast improvement claims use the recorded comparison interval. Full histories,
comparisons, exports and methodology remain directly accessible, with a return
link from each detailed view to the relevant section.

The interface uses a pure black background (`#000000`), light text and red accents.
There are no decorative hero sections or floating presentation controls. Phone
layouts keep pace and rank ranges visible; full interval detail is available in
the detailed ranking pages.
Team markers use the current colours from [Formula 1’s team directory](https://www.formula1.com/en/teams),
recorded on 2026-09-30 in `dashboard/src/lib.ts` and keyed by team lineage.
Historical entries retain that lineage palette. Chart comparisons use red and
white, with a dashed second series, to stay distinct even for teammates.
The Methodology page opens with a short guide to reading the ratings, then
describes the qualifying and race models, rating definitions, uncertainty,
validation limits and publication process.

The Drivers and Cars pages each switch between qualifying and race pace. Driver
qualifying also offers the experimental portable-skill estimate. Metric links are
shareable (`#drivers/race`, `#cars/race`); the former `#race` link redirects to
driver race pace. Race pace reads a separate checked export from `outputs/race_total/`.
Its driver and car tables are gated independently; an absent or failed result is
shown as pending or withheld. The existing qualifying tables and comparisons keep
their own metric and data cutoff.
Qualifying and race pace share the ranking table, filters, event line, export
control and selected-entry card. Switching metrics retains the selected entry,
search and team filter. Metric definitions, evidence counts and validation remain
specific to the selected metric.

Drivers also offers **Overall (equal car)** (`#drivers/overall`), an experimental
championship scenario with equal average machinery. It shows expected points per
race, title probability, simulated season rank ranges, and a selected driver's
contribution breakdown. The headline retains driver–team effects; it is not a
universal transferable-skill ranking or a forecast of the actual championship.
Contributions change each component to the field average separately and all at
once; correlated contributions must not be added. Wet pace and pit operations
are not separate driver components, and passing, traffic and strategy are not
simulated lap by lap. Driver-specific own-error rates enter retirements only
when their separate gate passes.

The loader reads `outputs/championship/{summary.json,standings.csv,contributions.csv}`
without fitting or simulating. It requires a current manifest covering every file
and `summary.combined_validation.gate` to be exactly true, then rechecks provenance
after reading. Missing, stale, historical or failed evidence never exposes a
standings or contribution table. Current evidence also lists entered qualities,
qualities tested but not admitted, tests not run, and the excluded qualifying-fold
seasons for both combined selection and race-stage validation. Excluded seasons
may still train later folds. These decisions and every displayed result value
come from the verified outputs at publication time; they are not copied from the
historical research notes. Older dashboard releases without this payload show an
unavailable status until a new release is published.

It provides:

- Driver and car rankings, 90% pace and rank intervals, fastest/top-three
  probabilities, searchable tables, team filters, entry details and CSV downloads.
- Driver and car history, including former drivers and team lineages, with season
  selection, uncertainty bands, circuit codes on the axis, marked team changes and
  regulation resets, and optional circuit-adjusted car pace. The revised history
  can also be shown as rank in each event's field (median and 90% range). A toggle
  switches between revised history (fitted on all races) and estimates after each
  race (each point fitted only on races up to that event; see below).
- A car-and-driver chart splitting each driver's expected qualifying pace into
  their car's part and their own in-team part, and a qualifying-against-race-pace
  scatter. Driver race pace also has a season-by-season history since 2018.
- A Track record page scoring forecasts: for each 2026 race, a fit that stops at
  the previous race predicts that race's qualifying teammate gaps and field order.
  From October 2026 it also shows the forecast for the season's next race,
  published before its weekend, and scores those forecasts afterwards.
- Head-to-head pace differences and a field-wide probability matrix. Differences
  use joint posterior samples, preserving dependence between estimates.
- Model health: data cutoff, publication and convergence checks, the latest
  refit and its stage timings, refresh status and errors, per-quality readiness
  (research models collapsed), clearly marked historical validation, source
  fingerprints, and archives of as-published qualifying and race estimates.

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
a refresh process that exited without recording its final status. On GitHub, a
scheduled workflow runs the same refresh after each race (see
[Scheduled refresh after each race](#scheduled-refresh-after-each-race)).

After a race, include the new timing data and timeline in the same update:

```bash
.venv/bin/python -m f1rank.dashboard refresh --races \
  --extract-python .venv-fastf1/bin/python --race-python .venv-gpu/bin/python
```

This extracts uncached race timing with FastF1, combines the tables, rebuilds the
incident timeline, then fits and exports both qualifying and race pace before
publishing. Omitting the environment options uses the current Python environment.
Race fitting takes about 15–27 minutes per fit (ten fits from scratch: the full model and
three versions of each of three validation folds); completed, unchanged fits are
checkpointed and reused, so a refresh with a new race normally refits only the full model.

Dashboard data uses a versioned JSON contract: `latest.json` points to an immutable,
content-addressed file in `dashboard/public/data/releases/`. The pointer changes only
after the complete release has been written. The underlying qualifying export has a
manifest covering its inputs, model/export code, fit metadata and every consumed
output. Partial exports and changed inputs are rejected. The full posterior fit is
checked by `export`, while the browser only receives summaries and pairwise contrasts.
The current publication, pointer and portable receipt are committed. After publishing,
run `.venv/bin/python dashboard/verify_publication.py record` on the fitting machine.
This compares the payload with the fully checked exporter and records source hashes.
CI verifies that receipt and immutable payload with
`python dashboard/verify_publication.py`, then builds the site without ignored
posterior caches. Changed inputs, gate evidence or publication content stop the build.
The scheduled refresh records and commits the new receipt and current publication.
A static host must receive updated `data/` files to show subsequent releases. Serve `latest.json` and `status.json` without caching.

The total driver race-pace ranking is published; the car race-pace table is withheld
because it did not pass the prediction and interval-coverage checks. Equal-car
championship standings remain withheld after regeneration: the combined validation
gate failed. The Overall view shows the current evidence and reason without a ranking.
Portable skill is explicitly experimental. Historical validation files have no run
manifest and are shown as recorded research, not as fresh acceptance of a new fit.

Verification:

```bash
.venv/bin/python dashboard/verify_publication.py
.venv/bin/python -m pytest -q
npm --prefix dashboard run build
cd dashboard
npx playwright install chromium              # once, for browser tests
npm test -- --workers=2
```

Browser tests cover interactions, exports, release adoption and failure handling,
mobile overflow, browser errors, and automated accessibility checks, including
passed and not-established equal-car fixtures. `PLAYWRIGHT_CHROMIUM_EXECUTABLE`
optionally selects an existing Chromium binary. The Python checks cover publication atomicity, corruption, process locking,
missing provenance, stale racing outputs and statistically correct comparisons.

### Estimates after each race and the track record

The revised history in `outputs/ratings/` is refitted on every race, so a point for
round 5 also reflects rounds 6 onward. `python -m f1rank.asof` records what the
data supported at the time instead. The fit for event k uses the same design with
the likelihood cut after event k−1 (the same mechanism as the leave-future-out
validation), at 1,000 warm-up and 1,000 draws per chain. Each record,
`outputs/asof/<event>_<fit id>.json`, holds:

- driver (in-team and portable) and car ratings after event k−1, with 90%
  intervals and rank ranges, shown by the dashboard's *After each race* history;
- a forecast of event k's qualifying from that fit, scored against what happened:
  teammate gaps in each segment (with a fresh one-weekend form draw and the fitted
  session noise), their 90% interval coverage, and the rank correlation between
  the predicted (car + circuit + driver) and actual order of each segment.

Records are write-once and only written for converged fits. A fit that fails the
convergence checks is repeated once at the main fit's chain lengths (1,500 + 1,500)
with a new seed; the record keeps the failed attempt's diagnostics. If the repeat
also fails, the event has no record and the refresh continues. The scheduled refresh
adds one per race (`asof fit --latest`). The 2026 records before October 2026 were
computed retrospectively with the same code; their fit ids are their creation times.
`asof summary` pools established pairs (both drivers with at least 10 earlier
qualifying sessions) into `summary.json`, compared with repeating each pair's gap
from their latest season together and with a zero gap. Each comparison is the
difference in mean squared error (model minus baseline) with a 95% interval from
resampling whole events (4,000 resamples, seed 0). The longer benchmark across
25 historical cutoffs remains the qualifying model's main validation.

As of 2026-15, 14 races are scored (280 teammate gaps; 2026-15 has no record because
both of its fits missed the convergence check). The model's teammate-gap error is
0.476 s, against 0.512 s for last season's gap (difference −0.036 s², 95% interval
−0.071 to −0.008: the model is better) and 0.479 s for no gap (−0.003 s², −0.022 to
+0.014: no clear difference). Its 90% intervals cover 95.7% of the gaps, slightly
wide, and the predicted order has a mean rank correlation of 0.74 with each
segment's actual order. Teammate gaps in a single session are mostly noise, so
predicting them is hard to do better than predicting none.

`python -m f1rank.forecast next` also publishes a forecast for the season's next race
before its weekend (from October 2026, the first being 2026-16), from the published main
fit: `outputs/forecasts/<event>_<fit id>.json`, one per event (a later run, even after a
refit, leaves the first as published), and only if the race is at least two days away. The fit has no states for that event, so it uses the latest
event's states plus the model's one-race-ahead terms: a skill-walk step and a
one-weekend form draw per driver, a within-season car step (Student-t), a one-event car
variation and the next circuit's adjustment, the fitted lap noise for a new session,
and the latest lineups. The experience and age trend's one-event change is left out.
After a season's finale no forecast is made. `forecast score` scores each forecast
against its qualifying once it has happened, with the after-the-race records' rules,
into `outputs/forecasts/scores.json`; the Track record page shows the next forecast
and these scores.

The race-pace export also writes season-by-season estimates since 2018 (each season
centred on its own rated drivers, revised with all data) and a write-once snapshot
of the tables as published by each fit, `outputs/snapshots/race/`. A table that
failed its gate is withheld from both.

The qualifying entries table now also includes drivers who started a race but have
no qualifying result in Jolpica (30 entries across 2011–2026, for example three
2026-01 starters). Their `quali_position` is empty and they add no lap
observation. As in the model's design for a driver without a valid lap, they keep
a driver state at that event (carried forward by the skill walk), count towards
experience, and are part of the field that ratings are relative to; previously
their history had a gap at that event.

### Publish on push with GitHub Pages

`.github/workflows/dashboard.yml` checks and deploys the dashboard when relevant
changes reach `main` or a pull request targets `main` or `automated-refresh`.
Passing pull requests from branches in this repository also publish to the same
public Pages URL, replacing its previous deployment. Fork pull requests run the
checks without publishing. Deployments are serialized across branches.
The workflow can also be started manually from the Actions tab on `main`.

One-time setup in [repository Pages settings](https://github.com/SamuelReeder/f1-analysis/settings/pages):
choose **GitHub Actions** under **Build and deployment → Source**. In the
`github-pages` environment, allow deployment branches `main` and
`refs/pull/*/merge`. Then push the dashboard, workflow, checked ratings and
supporting files to `main` or a same-repository pull request branch. The site
address is **https://samuelreeder.github.io/f1-analysis/**. Preparing the workflow
locally does not enable Pages or publish the uncommitted dashboard.

Each deployment verifies the committed export manifest, builds a browser dataset,
runs publisher checks, builds the site, and runs browser/accessibility tests against
the static build under `/f1-analysis/`. Only the checked `dashboard/dist/` artifact
is deployed. A failed build or check leaves the previous deployed site in place.
GitHub's built-in deployment token is used; no personal token or hosting secret is
needed. Versions of the official Actions are pinned to verified commit hashes.

**New data and model fitting are separate from website deployment.** The scheduled
refresh below does both after each race. To refresh by hand instead, run
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

### Scheduled refresh after each race

`.github/workflows/refresh.yml` runs every Monday and Tuesday at 06:00 UTC on a
standard GitHub-hosted runner. No GPU is needed: every model in the refresh runs on
CPU (JAX's CPU backend, four chains in parallel). Each run:

1. restores the raw-data and race-fit cache from the `refresh-cache` release asset;
2. checks Jolpica for a race newer than the processed data
   (`python -m f1rank.schedule check`). Without one, the run stops here. A race whose
   FastF1 timing was not yet available is retried by later runs for up to 10 days;
3. runs `python -m f1rank.dashboard refresh --races`: fetch and build, FastF1 race
   and sprint timing, race tables and timeline, the qualifying fit with the
   published sampler settings (1,500 warm-up and 1,500 draws per chain, 4 chains),
   export, a forecast of the season's next race (`forecast next`) and the scores of
   earlier ones (`forecast score`), an after-the-race record (`asof fit --latest`) and
   its summary, then race
   pace validation and export. Unchanged race validation fits are reused from the
   cache, so a normal week refits only the full race model;
4. runs the Python tests, commits the changed results to `main` as
   `github-actions[bot]` and starts the Pages deployment. The push fails, and
   nothing is published, if `main` changed during the run; a failed fit, gate or
   test likewise stops before the commit;
5. uploads the updated cache and keeps the refresh log as a run artifact for 30
   days.

It can also be started from the Actions tab (**Refresh after race weekends → Run
workflow**), optionally with *force* to refit without a new race.

The refresh does not refit the racing qualities (qualifying-adjusted race pace and tyre
management, starts, consistency, reliability, pit stops, wet pace, overtaking) or the
equal-car championship: their held-out fits take GPU hours. Their manifests hash every
processed table and module, so after the first refresh that adds a race, Model health
lists them as stale until the racing regeneration (see **Review fixes and
regeneration**) is rerun on the fitting machine. None of them feeds a published
ranking; the published race-pace tables come from the race model above, which the
refresh does refit.

One-time setup, from the fitting machine with the GitHub CLI signed in, seeds the
cache so the first run does not download every season again:

```bash
.venv/bin/python -m f1rank.cachestore pack refresh-cache.tar.gz
gh release create refresh-cache refresh-cache.tar.gz --prerelease --latest=false \
  --title "Refresh cache" --notes "Raw data and race-fit checkpoints for the scheduled refresh"
```

The workflow pins its Python dependencies (`requirements/fitting.txt`,
`requirements/fastf1.txt`) and requests only `contents: write` and `actions: write`.
GitHub disables schedules in public repositories after 60 days without activity;
scheduled runs re-enable the workflow through the API to cover the winter break.
Runner time and memory for a full refit have not yet been measured on GitHub's
runners; the first scheduled run will record them (Model health shows each stage's
duration). On the fitting machine (WSL2, 16 cores, 19.5 GB, CPU only for these
steps, with other fits running alongside), measured on 2026-10-01:

| Step | Time | Peak memory |
|---|---|---|
| Qualifying fit, 4 × (1,500 + 1,500) | 27.8 min sampling | over 3.1 GB (not captured at the end) |
| Qualifying export | 11 s | — |
| After-the-race fit, 4 × (1,000 + 1,000) | 15–21 min | 5.3 GB, briefly, when samples are gathered (2.1 GB while sampling) |
| Race validation and export, all fits reused | 10 s | — |

A week with a new race also refits the full race model; its cached fits took
15.0–26.8 min each on CPU or GPU (`outputs/race_total/cache/*.npz` metadata). A
standard public-repository runner has 4 CPUs and 16 GB, so the sequential refresh
should fit within the job's 345-minute limit, but this is an estimate until the
first run.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scipy pyarrow requests "jax[cpu]" numpyro arviz

.venv/bin/python -m f1rank.fetch --first 2006   # download (cached; current season refreshed)
.venv/bin/python -m f1rank.build                # tidy Parquet tables in data/processed/
.venv/bin/python -m f1rank.fit --warmup 1500 --samples 1500   # main fit (~40 min)
.venv/bin/python -m f1rank.export               # dashboard-ready outputs in outputs/ratings/
.venv/bin/python -m f1rank.jobs synth-source    # freeze the main fit as the synthetic-truth source
.venv/bin/python -m f1rank.jobs all             # validation, sensitivity, placebo fits (hours)
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
  - `evaluate` scores each validation fit on the design its job builds from the
    current data, so a stale fit is rejected.
  - `jobs all` reruns only fits that are missing or stale.
  - `jobs list` shows the status of each fit.
- **Converged validation fits.** Every validation, sensitivity and synthetic fit uses a
  fixed retry rule (`jobs.ATTEMPTS`, shared with the racing folds): 4 chains × (700
  warm-up + 400 draws); if the fit fails the publication convergence checks (R-hat
  below 1.05 for every parameter, at most one divergence per 1,000 draws), again with
  longer chains and a new seed (1,400 + 800, then 2,100 + 1,600, then 2,100 + 1,600
  with target acceptance 0.98). A job whose attempts all fail keeps no fit and is listed
  as excluded in `lfo_summary.json`. Each leave-future-out design ends with its test
  season. Before 2026-10 the fits used one attempt and kept every later season in the
  design; 33 of the 34 earlier leave-future-out and sensitivity fits failed the main
  fit's convergence check (`analysis/validation_convergence.py`, recorded in
  `outputs/analysis/validation_convergence/before_retry_rule.json`). In the 24
  leave-future-out fits the worst car R-hat was 2.37 for later seasons without data,
  1.45 for the test season and 1.09 for the training period; driver states, which the
  teammate forecasts use, reached 1.06. Under the retry rule, three leave-future-out
  fits still failed every attempt (`lfo_end2013`, `lfo_end2020`, `lfo_mid2015`; recorded
  in `outputs/fits/*.failed.json`): R-hat stayed above 1.05 on all four attempts of each,
  and `lfo_end2013` also had 46–230 divergences per attempt. They are left out of the
  forecasting scores, and the racing test seasons 2014 and 2021 share the first two
  fits and are left out of the racing tests. All sensitivity, placebo, synthetic and
  sprint-test fits converged.
- **Shared fits.** The validation fit `lfo_end<year>` and the racing fold
  `quali_fold<year+1>` have the same design and retry rule, so whichever is fitted
  second is copied from the first after its fingerprint and retry history are checked.
  If the first failed every attempt, the second is recorded as failed too
  (`<name>.failed.json` with `same_as`) instead of repeating the same seeds.
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

- **Forecasting**, 22 leave-future-out cutoffs 2013–2026 (three cutoffs, `lfo_end2013`,
  `lfo_end2020` and `lfo_mid2015`, failed every attempt of the retry rule and are
  excluded, so they are not scored): the model beats a static two-way model, raw
  teammate gaps and a zero baseline on the pairing-level and segment-level targets.
  Teammate gap per pairing: RMSE 0.192 s vs 0.234–0.265 s. New pairings: 0.265 s vs
  0.266 s for the zero baseline (a margin of 0.001 s) and 0.406–0.458 s for the other
  two. Session-level 90% intervals cover 94.2%.
- **Sprint qualifying** (pre-registered test, `docs/sprint_qualifying.md`): the gate
  failed, so the published model does not use sprint-qualifying sessions.
- **Synthetic recovery** on the real F1 network, over 8 independent clean truths (each
  with its own hyperparameters from the posterior):
  - Car ratings: correlation 0.99; 90% intervals cover 89% on average (87–92%).
  - Driver skill: 90% intervals cover 89% on average (86–92%).
  - Misspecification scenarios (one shared truth): portable current-grid rank
    correlations stay within the spread between the clean truths.
- **Pace in the current car (headline)** passes its own gates:
  - Current-grid ranking recovered with rank correlation 0.83 on average (0.72–0.93).
  - Rank correlation 0.95 or more against every sensitivity variant.
- **Limits:**
  - **Portable ranking recovery.** The current-grid ranking by portable skill is
    recovered with rank correlation 0.65 on average (0.53–0.84).
  - **Sensitivity.** The portable ranking moves with structural choices: whether the
    team-specific effect is modelled, the 2006 vs 2010 data window, and whether the
    effect restarts at each regulation era (rank correlation with the main ranking as
    low as 0.81, 0.85 and 0.88 for those variants). The windows and the variants forecast
    about equally well: the effect per spell and per era have slightly lower pairing RMSE,
    and the 95% interval of each MSE difference includes zero.
  - **Status.** Seven of the nine acceptance gates pass. The two that fail are portable
    skill's, so portable skill is experimental and pace in the current car is the
    headline.

  Read ranks as ranges.

Outputs for the dashboard are in `outputs/ratings/`, and `outputs/snapshots/` keeps what
was published for each fit. Posterior draws (`outputs/fits/`, several GB) are not
committed. The qualifying validation's fits are `lfo_end<year>.npz` and
`lfo_mid<year>.npz`; the racing folds' qualifying fits (trained through the season
before the one they forecast, design ending with that season) are
`quali_fold<season>.npz`, so regenerating either never replaces the other's inputs.

## Racing (stage 2)

**2026-10-05 regeneration update:** the canonical racing outputs now use qualifying
features frozen before each test season, nested conditional entry tests, current
provenance and the fixed convergence retry rules. The equal-car championship ran,
but its combined validation gate failed, so an overall best-driver ranking cannot
be published. These results replace the earlier racing gate claims below.

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

**Regenerated standalone gates** (sources: the corresponding summaries in
`outputs/`, and `outputs/race/multi_heldout.json`; intervals are 95%):

- **Race-specific pace** passes: teammate-gap MSE difference −0.00505 percent²
  (−0.01151 to −0.00022; 113 pair-seasons). **Degradation** fails: −0.0000262
  (−0.0000566 to +0.000000067). The descriptive era splits do not replace these gates.
- **First-lap performance** passes with the lasting team term fixed in advance:
  driver log predictive density improves by +0.106 per race or sprint (+0.050 to
  +0.161). The transfer interval is −0.044 to +0.016, satisfying the fixed rule
  that it must not be entirely below zero. The lasting team term improves by +0.538
  (+0.403 to +0.674).
- **Pit stops** pass as team operations: +1.609 log predictive density per race
  (+0.860 to +2.307).
- **Reliability** fails for driver errors (+0.003; −0.002 to +0.010), team-season
  mechanical effects (−0.003; −0.024 to +0.019), and supplier-season effects
  (−0.015; −0.055 to +0.023), in log predictive density per race.
- **Consistency** fails: MSE difference −0.000221 (−0.002717 to +0.002377) on the
  log ratio of teammate lap-time spreads. **Wet pace** fails: squared-error
  difference −0.01856 per race (−0.11546 to +0.07501; 11 wet races).
- **Overtaking and defending** fail synthetic feasibility: attacker recovery
  correlation 0.68 and defender correlation 0.50; both have 87% interval coverage.
  They remain opportunity counts, with no held-out draws or championship entry test.
- **Results benchmark:** the unexplained driver effect does not establish an
  improvement (+0.113; −0.022 to +0.237 log predictive density per race).

**Entry into the overall rating** (`outputs/championship/summary.json`): conditional
backward removal selects race-specific pace (+0.199; +0.064 to +0.338 per race),
but excludes first-lap performance (+0.007; −0.030 to +0.037), degradation
(+0.001; −0.015 to +0.017), and consistency (−0.030; −0.061 to −0.004).
The outer test of the complete selection procedure fails: +0.125 per race,
interval −0.030 to +0.293, across 207 races. Consequently **no racing quality
enters**, driver error rates are averaged, and the experimental equal-car standings
are withheld. Wet pace is not tested in the dry-race scenario; team operations and
mechanical risk are equalised.

The qualifying folds for **2014 and 2021** failed every prescribed attempt, so
those seasons are excluded wherever season-ahead qualifying features are needed.
They can still train later folds. Wet pace excludes 2021; its test has no 2014 races.
The dated amendments in `docs/racing_approach.md` fixed this exclusion rule and the
racing retry rule before scores were inspected.

All racing lanes completed. First-lap fits needed 17 retries among 92 fits, each
converging on the second attempt with zero divergences. The reused race-pace fits
for 2013, 2015 and 2023 also retain their earlier second attempts. No other racing
quality or championship fit needed a retry. Every retried fit's diagnostics and
the stage runtimes are recorded in `outputs/analysis/racing_regeneration/run.json`
and the dated completion note in `docs/racing_approach.md`.

The three older FastF1-only race-pace sensitivity outputs remain stale and are
excluded from `outputs/REPORT.md`. Its stale-output note names those files;
the canonical results above and their championship provenance are current. Those
secondary comparisons provide no evidence for these decisions.

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
  preparer and joint race fitter have fixed retries. Racing-quality fits use the
  fixed `artifacts.RETRY` ladder from the 2026-10-04 amendment: the original fit,
  then doubled warm-up and draws, then triple warm-up and quadruple draws, then
  those lengths at target acceptance 0.98; each retry uses a new seed. Exhausting
  the ladder stops publication without retuning.
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

The car table failed that test (it predicted later team pace no better than the
driver-only model, and its intervals under-covered). A pre-registered follow-up,
[`docs/race_car_state.md`](docs/race_car_state.md), let each car's pace change from
race to race within a season (`f1rank/race_car_state_model.py`, validated with
`python -m f1rank.race_car_state --validate` on the same races, baseline and gate).
It brought coverage to 88.5% but was again no more accurate than the driver-only
model (MSE difference 95% CI −0.0118 to +0.0104 percent²) and its intervals were
wider, so under the rule fixed beforehand the car table remains withheld. The
dashboard shows that recorded result beside the v1 check.

A second pre-registered follow-up, [`docs/race_features.md`](docs/race_features.md),
tested information published before each race as additions to v1 on the same races
and gate: race-fuel pace in practice (`longrun`, team and teammate), practice
speed-trap speed (`traps`) and the number of performance upgrades declared to the FIA
(`upgrades`, parsed from the FIA's Car Presentation Submissions by
`extract/fia_upgrades.py`). Practice laps come from `extract/practice.py` and
`python -m f1rank.race_features build`; each test runs with
`.venv-gpu/bin/python -m f1rank.race_features validate FEATURE`. The practice
definitions were amended before their fits, after their covariates showed values no
car could produce; the amendment is dated in the document. All four tests failed:
long-run pace came closest (team MSE difference −0.0024 percent², 95% CI −0.0075 to
+0.0032), speed traps made team predictions worse (+0.0044, CI +0.0001 to +0.0098),
and upgrades and the teammate split changed nothing measurable. No feature is used;
the dashboard lists the four results on the race views. Sector times were not tested:
as a weekend covariate they would repeat the practice pace, and a straight-line versus
cornering split needs a sector-level lap model.

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

The separate synthetic experiment (`analysis/race_total_recovery.py`) is incomplete.
Two of three planned truths produced converged fits; driver rank correlations were
0.91 and 0.75, and car correlations were 1.00 and 0.96. Seed 0 failed convergence
twice and was stopped during its third attempt after over an hour of total runtime.
`outputs/race_total/recovery.json` retains both completed cases and the incomplete
case. These results do not establish that the full recovery experiment passed and
do not override the real-data publication gates above.

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
