# Car race pace with a within-season car state: pre-registration

Written 2026-10-01, before any fit of this variant was run. The decisions below are
fixed. Results are recorded in `outputs/race_car_state/validation.json` and in a
dated section appended to this file; nothing above that section is edited after
results exist.

## Why

The published race model (`total-dry-race-pace-v1`) gives each team one car effect
per season. Its car table is withheld: on later races of the same season it did
not beat the driver-only ablation (MSE difference +0.0040, 95% CI −0.0061 to
+0.0133 percent²) and its 90% intervals covered 84.1% of held-out team results
(`outputs/race_total/validation.json`). The driver-only ablation is strong because
each driver-season form term already absorbs a constant season-long car level. The
v1 car term therefore adds information only if team pace changes during a season,
which a constant season effect cannot follow. Both shortfalls are consistent with
in-season development: a constant level is wrong late in the season, and its
interval ignores how far the car may have moved.

## The variant (`total-dry-race-pace-carstate-v1`)

`f1rank/race_car_state_model.py`, a separate file so the cached v1 fits (keyed on
the v1 model source) remain valid. Everything is identical to v1 except the car:

- car pace at race r for team t in season s = `package[s, t] + drift[s, t, r]`;
- `drift` is a Gaussian random walk over that team's dry races in the season used
  by the model: 0 at the team's first such race, then
  `drift[r] = drift[r−1] + sd_drift · z[r]`, `z ~ Normal(0, 1)`;
- `sd_drift ~ HalfNormal(0.1)` percent of lap time per race;
- the drift is centred across teams at every race, so it changes relative pace only;
- `package`, its zero-sum prior per season, `car_day` and all driver, tyre, lap,
  traffic and residual terms are unchanged, as are the sampler settings, retry
  rule (3 attempts with longer chains) and convergence requirements.

A prediction k model races after the last training race uses the team's last
fitted state plus `sqrt(k) · sd_drift · Normal(0, 1)`, drawn once per team per
race so teammates share it, plus the v1 race-day terms. k counts the season's dry
races in the model data after the training cutoff up to and including the target.
A team-season without training races gets a prior package draw (scale 1.5, as in
v1) plus the same k-step walk.

## Test (unchanged from v1)

- The v1 outer folds: train through 2024-10, 2025-10 and 2026-07; predict the
  remaining dry races of that season. Same lap selection, same Stage-A targets
  (`f1rank/race_targets.py`), same team-level rows and race-block bootstrap
  (`race_total.improvement`, 4,000 resamples, seed 0).
- Baseline: the v1 driver-only ablation (`<fold>_no_car`). With the car term
  removed the two models are identical, so the cached v1 fits are reused (the fit
  cache verifies their keys).
- Gate, as for v1: at least 12 held-out races; the 95% CI of the race-level MSE
  difference entirely below zero; 90% interval coverage within 85–95%; mean
  interval width below the baseline's.

## Decision rule

- One variant, one run. No prior, step definition, fold or gate is changed after
  results are seen. A fit that fails convergence after the third attempt is
  reported as a failure, not retuned.
- If the gate passes, the car race-pace table is published from this variant,
  labelled as car pace at the latest dry race (not a season average), with its own
  validation. Driver race pace continues to come from v1, whose driver gate was
  run separately; this variant's driver terms are not published.
- If the gate fails, the car table stays withheld and the result is reported here
  and on Model health in the same terms as the v1 result.
