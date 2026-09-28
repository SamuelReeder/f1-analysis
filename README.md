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
.venv/bin/python -m f1rank.jobs all             # validation, sensitivity, placebo fits (~3 h)
.venv/bin/python -m f1rank.evaluate all         # scores in outputs/validation/
.venv/bin/python -m f1rank.window_compare       # 2010 vs 2006 data window
.venv/bin/python -m f1rank.report               # outputs/REPORT.md
```

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
| `window_compare.py` | 2010 vs 2006 data window on identical forecast targets |
| `report.py` | `outputs/REPORT.md` and acceptance gates |
| `diagnostics.py` | Residual checks (every fitted term) |

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
entered at that event**. Driver-vs-car separation comes from teammates (same car)
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
years apart (8 cases).

The circuit factor is a 1-D rank-1 decomposition of team pace deviations, fitted
beforehand (and, for forecasts, only on data before the cutoff). One end is slow,
traction-limited street circuits (Marina Bay, Monaco); the other is high-speed
circuits (Spa, Silverstone, Monza).

## Results and validation

Full results: `outputs/REPORT.md` (`python -m f1rank.report`). Summary as of the 2026
Azerbaijan GP:

- **Forecasting**, 25 leave-future-out cutoffs 2013–2026: the model beats a static
  two-way model, raw teammate gaps and a zero baseline on every target. Teammate gap
  per pairing: RMSE 0.188 s vs 0.246–0.265 s. New pairings: 0.244 s vs 0.264–0.425 s.
  Session-level 90% intervals cover 94%.
- **Synthetic recovery** on the real F1 network (one simulated truth, several
  misspecifications): car ratings correlation ≥ 0.99. Driver-skill 90% intervals cover
  about 88–90% of true values, also under unequal upgrades, persistent form, extra
  team-specific effects and endogenous transfers.
- **Limits:**
  - **Portable ranking recovery.** In simulation, the current-grid ranking by portable
    skill is recovered with rank correlation about 0.75 (pace in the current car: about
    0.91).
  - **Sensitivity.** The portable ranking moves with two structural choices: whether
    the team-specific effect is modelled, and the 2006 vs 2010 data window. The two
    windows forecast equally well.
  - **Status.** Two of the seven acceptance gates fail for this reason, so portable
    skill is experimental and pace in the current car is the headline.
  - **Simulation evidence.** All synthetic scenarios share one simulated truth. They
    check recovery; they don't yet establish calibration.

  Read ranks as ranges.

Outputs for the dashboard are in `outputs/ratings/`, and `outputs/snapshots/` keeps what
was published for each fit. Posterior draws (`outputs/fits/`, several GB) are not
committed.

## Racing (stage 2, proposed)

- **Approach:** `docs/racing_approach.md`.
- **Data check:** `analysis/race_signal.py` is a reproducible check of the racing
  signal in 2018+ timing data. Its results go to `outputs/analysis/race_signal/`.
- **Separate environment:** the check needs FastF1, which requires pandas < 3, so it
  runs outside the main environment:

  ```bash
  python3 -m venv .venv-fastf1 && .venv-fastf1/bin/pip install fastf1 pyarrow
  .venv-fastf1/bin/python analysis/race_signal.py --first 2018 --last 2026
  ```
