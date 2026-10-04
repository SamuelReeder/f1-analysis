# Sprint qualifying as qualifying data: pre-registration

Written 2026-10-01, before any fit that uses sprint qualifying was run. The decisions
below are fixed. Results are recorded in `outputs/validation/compare.json` (key
`lfosprint`) and in a dated section appended to this file; nothing above that section
is edited after results exist.

## Why

The qualifying model uses one-lap times from Q1, Q2 and Q3. Since the 2023 Azerbaijan
GP, sprint weekends also have a one-lap sprint qualifying session in three parts (the
"Sprint Shootout" in 2023, "Sprint Qualifying" since 2024): 23 events up to 2026-12,
about one more qualifying session per four race weekends. Its teammate gaps and car
order measure the same thing as qualifying, so they may sharpen the driver and car
estimates. They may also add noise: the session has its own tyre rules, run plans and
parc fermé timing, which have changed between seasons.

## Data

`extract/sprint_quali.py` (FastF1 environment) rebuilds each driver's best lap in
SQ1, SQ2 and SQ3 from FastF1 lap timing, with the method of `extract/quali_fill.py`
(laps FastF1 flags as deleted are excluded; on the 2025 reference events that method
matched 1,014 of 1,015 Jolpica qualifying times). Neither Jolpica nor FastF1's results
carry sprint qualifying times, so there is no official table to check against.
Additionally, a lap named in a race-control "LAP DELETED" message is excluded unless
that car has a "REINSTATED" message in the session: the first extraction kept one such
lap as a best (2025-02, HUL, SQ1 lap 2, 1:32.675, deleted for track limits; his valid
best was 1:32.855). Seven laps are excluded only by their message; all are listed in
the supplement. Result: 1,035 times from 23 events (2023-04 to 2026-12),
`data/supplements/sprint_quali_times.json`, built into
`data/processed/sprint_quali_times.parquet`. Drivers are matched by abbreviation to
that weekend's sprint results (Jolpica ids).

## The variant (`lfosprint`)

`build_design(..., sprint_quali_until=E)` adds the SQ1/SQ2/SQ3 times of events up to
and including E as further segments of their weekend, processed exactly like Q1-Q3:
the same pace transform against the segment median, the same 5% outlier cut and the
same minimum of 4 times per segment. Each segment has its own intercept and noise
scale, the car has its usual effect per segment, and the weekend's car and driver
states are shared with that weekend's qualifying. The model and its priors are
unchanged. Sprint qualifying after E is left out of the design entirely, so the
forecast targets are exactly those of the published model.

## Test

- Cutoffs: the seven leave-future-out cutoffs after the first sprint qualifying
  session: mid2023, end2023, mid2024, end2024, mid2025, end2025, mid2026
  (`jobs.SPRINT_CUTS`). At each, `lfosprint_<cut>` uses sprint qualifying up to the
  cutoff and `lfo_<cut>` (the published model's validation fit) does not. Both are
  fitted on the same current data, with the same design end (the test season), the same
  fixed retry rule (`jobs.ATTEMPTS`) and the same convergence requirement.
- Targets and metrics: `python -m f1rank.compare`, which forecasts the same teammate
  pairings and segments with both fits (`evaluate.evaluate_cutoff`):
  - pairing level: mean squared error difference (variant − main, s²) over matched
    pairings, with a 2,000-resample bootstrap 95% interval over pairings
    (`compare.paired`); its RMSE at each cutoff (`by_cutoff`);
  - session level, averaged over the cutoffs: 90% interval coverage and CRPS.

## Gate (all must hold)

1. All 14 fits converge under the retry rule. A fit that fails every attempt ends the
   test as a failure; nothing is retuned.
2. The 95% interval of the pairing-level MSE difference lies entirely below 0.
3. The variant's pairing RMSE is lower at 5 or more of the 7 cutoffs. (Pairings at one
   cutoff share a fit, so the pairing bootstrap is somewhat too narrow; this guards
   against one cutoff carrying the result.)
4. The variant's session-level 90% coverage is within 85–97% (the published
   forecast-calibration gate).
5. The variant's session-level CRPS is no higher than the main model's.

## Decision rule

- One variant, one run. No data rule, design choice, cutoff, metric or gate is changed
  after results are seen.
- If the gate passes, sprint qualifying becomes part of the published qualifying model
  as a new model version: the default design includes it, the scheduled refresh
  extracts it for new sprint weekends, and the main fit, the full validation, the 2026
  after-the-race records and the racing folds are regenerated before anything is
  published with it.
- If the gate fails, the published model is unchanged and the result is recorded here
  and in the README.

## Result (2026-10-04): the gate failed; the published model is unchanged

Source: `outputs/validation/compare.json`, `lfosprint` and `sprint_qualifying_gate`,
produced by `f1rank.compare` after all 14 fits finished.

| Gate item | Result | Holds |
|---|---|---|
| 1. All 14 fits converge | all converged under the retry rule | yes |
| 2. Pairing MSE difference below 0 (95% interval) | +0.00061 s²; interval -0.00105 to +0.00218 (62 pairings) | no |
| 3. Lower pairing RMSE at 5 or more of 7 cutoffs | lower at 4 of 7 (end2024, end2025, mid2023, mid2025) | no |
| 4. Session 90% coverage within 85–97% | 95.2% (main model 95.0%) | yes |
| 5. Session CRPS no higher | 0.23647 s against 0.23501 s | no |

Pairing RMSE overall: 0.1620 s with sprint qualifying against 0.1601 s without. The
variant was better at four cutoffs and worse at three, and the interval on the
difference includes zero, so these data show no improvement from adding sprint
qualifying. Per the decision rule above, the published model, the main fit and the
2026 records are not regenerated for it, and nothing else about the design was changed
after seeing these results.
