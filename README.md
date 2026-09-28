# f1-analysis

Separates F1 **driver skill** from **car-package performance** using qualifying lap
times, and produces continuously updatable ratings with honest uncertainty.

Stage 1 scope: **one-lap qualifying pace**, 2010 onward. Race pace, reliability and
driver sub-skills are later stages.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scipy pyarrow requests "jax[cpu]" numpyro arviz

.venv/bin/python -m f1rank.fetch --first 2006   # download (cached; current season refreshed)
.venv/bin/python -m f1rank.build                # tidy Parquet tables in data/processed/
.venv/bin/python -m f1rank.fit                  # main fit (~15 min on 16 cores)
.venv/bin/python -m f1rank.export               # dashboard-ready outputs in outputs/ratings/
.venv/bin/python -m f1rank.jobs all             # validation + sensitivity fits (~2 h)
.venv/bin/python -m f1rank.evaluate all         # scores in outputs/validation/
```

After each qualifying session: `fetch`, `build`, `fit`, `export`. Each export writes an
append-only snapshot to `outputs/snapshots/`, so what was published at each point
is kept.

| Module | Role |
|---|---|
| `fetch.py` | Jolpica (Ergast-compatible) API client with caching and rate limiting |
| `build.py` | Qualifying times, entries, drivers, race results as Parquet |
| `lineage.py` | Maps rebrands to one team lineage (e.g. Toro Rosso → AlphaTauri → RB) |
| `design.py` | Pace per segment, driver/car state indices, circuit factors |
| `model.py` | The Bayesian model (NumPyro) |
| `fit.py` | NUTS fitting |
| `ratings.py` | Leaderboards, series, rank distributions |
| `export.py` | Dashboard files and snapshots |
| `simulate.py` | Synthetic truth on the real F1 network, with misspecification scenarios |
| `jobs.py`, `evaluate.py` | Leave-future-out forecasting, synthetic recovery, sensitivity |
| `diagnostics.py` | Residual checks |

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
  + team fit             driver x team compatibility, fixed while the driver stays in that team
  + driver weekend form  one event, does not persist
  + noise                Student-t, with a scale per segment (wet/chaotic sessions down-weighted)
```

Only relative quantities are identified. Car and driver effects are centred on the
field at each event, and the ratings are **relative to the average driver / car
entered at that event**. Driver-vs-car separation comes from teammates (same car)
and from drivers moving between teams. The transient terms stop one-off weekends
from being read as lasting changes.

The headline driver rating is **portable skill**, excluding team fit. A placebo test
(splitting stints within one team) shows the compatibility effect is team-specific:
SD about 0.16% at a team change vs about 0.02% within a team. So a driver's pace
relative to teammates carries a large team-specific part. For drivers who have only
raced for one team, portable skill is therefore uncertain and pulled towards average.

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
- **Synthetic recovery** on the real F1 network: car ratings correlation ≥ 0.99;
  driver-skill intervals cover about 88–90%, also under unequal upgrades, persistent
  form, extra compatibility and endogenous transfers.
- **Limits:** the current-grid ranking by portable skill is recovered with rank
  correlation about 0.75 (the in-current-car ranking about 0.91), and it moves with two
  structural choices: whether team fit is modelled, and the 2006 vs 2010 data window.
  The window choice is predictively indistinguishable. Read ranks as ranges.

Outputs for the dashboard are in `outputs/ratings/`; `outputs/snapshots/` keeps what
was published after each event. Posterior draws (`outputs/fits/`, several GB) are not
committed.
