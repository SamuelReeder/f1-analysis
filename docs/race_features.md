# Weekend information for race pace: pre-registration

Written 2026-10-02, before any fit that uses these features was run and before the
features were compared with any race result. The decisions below are fixed. Results are
recorded in `outputs/race_features/<feature>/validation.json` and in a dated section
appended to this file; nothing above that section is edited after results exist.

## Why

The race model (`total-dry-race-pace-v1`, `f1rank/race_total_model.py`) predicts a
team's pace at a later race from a constant team-season effect, and its car table is
withheld: on later races of the same season it was no better than the driver-only
ablation (`outputs/race_total/validation.json`). A within-season car state did not fix
that (`docs/race_car_state.md`). Information published before each race weekend's race
may: how fast each team ran on race fuel in practice, how fast each car was on the
straights there, and which upgrades it brought. Each is tested here as one addition to
v1 on v1's held-out races. (Sprint qualifying, the fourth feature proposed with these,
has its own test: `docs/sprint_qualifying.md`.)

Sector times are not tested. They are parts of the lap times the model already fits, so
as a weekend covariate they would repeat the practice pace; splitting the car term into
straight-line and cornering parts would need a sector-level lap model, which this test
does not build. The speed-trap feature below is the straight-line part.

## Data

1. **Practice laps** (`extract/practice.py`, FastF1 environment): one session per race
   weekend from 2022, the one in which teams do their race-fuel runs: the second
   practice on a conventional weekend, the only practice on a sprint weekend.
   `python -m f1rank.race_features build` combines them into
   `data/processed/practice_laps.parquet`, with driver ids from the abbreviations in that
   weekend's race laps (`race_laps.parquet`); a weekend whose extraction failed has no
   rows.
2. **Upgrades** (`extract/fia_upgrades.py`): the FIA's "Car Presentation Submissions"
   document for each event, published on the Thursday or Friday of the weekend before
   any running. The first is 2024 round 2; none was published before. Each team's
   table is read with pdfplumber into its declared components with their primary
   reason (a reason cell merged over several components applies to each); a component
   counts if its primary reason is "Performance", including a Performance subcategory
   written without the prefix ("Local Load", "Flow Conditioning", "Drag Reduction"), and
   not "Circuit specific" or any other reason. `data/supplements/fia_upgrades.json`
   keeps each document's URL and sha256, every team's parsed components, and the
   extractor's count check (numbered rows with text against parsed components).
   *Amended 2026-10-02, before any fit: this item first said the sections were parsed
   with pdftotext in layout mode; that could not separate merged and wrapped table
   cells, so the extractor reads the tables instead, and the subcategory rule was
   written down when teams were found to use it.*

## Features

Each feature is a separate variant: v1 plus that feature's covariates, nothing else.
Every covariate is known before the race. Where its source does not exist (practice
before 2022, upgrades before 2024 round 2, a failed extraction, a team or driver
without data) the covariate is 0. Each covariate x has its own coefficient,
β ~ Normal(0, 1). A car covariate adds β·x to the mean of that team's laps in that race
(next to the race-day car term), a driver covariate adds β·x to that driver's laps in
that race. All of v1's terms, priors, sampler settings (800 warm-up and 800 draws per
chain, 4 chains, three attempts with longer chains) and convergence checks are
unchanged.

**`longrun`: race-fuel pace in practice.** Practice laps with a lap time that are not
in- or out-laps, not deleted, marked accurate by FastF1, set entirely under green
(track status "1") on SOFT, MEDIUM or HARD tyres. A run is a driver's stint with at
least 5 such laps after laps more than 7% slower than the stint's median are removed;
its pace is the mean of those laps. For each compound with runs from at least 3
drivers, a run's relative pace is 100 × (compound median − run pace) / compound median
(percent, positive = faster), where the median is over that compound's runs. A driver's
value is the mean of their runs' relative paces; a team's value is the mean of its
drivers' values, centred across the teams with values at that event.
- car covariate: the team's value;
- driver covariate: the driver's value minus the team's, when both teammates have
  values (otherwise 0).

**`traps`: straight-line speed in practice.** The same session's laps with a lap time
and a finish-straight speed trap (`SpeedST`), excluding in- and out-laps. A team's value
is the 90th percentile of its two drivers' speed-trap readings, in km/h, minus the
median of the teams' values at that event, divided by 10. Car covariate only.

**`upgrades`: declared performance upgrades.** For each team and race: the number of
"Performance" components it declared at that season's events up to and including this
one, centred across the teams at the race, divided by 10. Car covariate only.

## Test

- v1's outer folds and rows (`f1rank/race_total.py`): train through 2024-10, 2025-10
  and 2026-07 and predict the remaining dry races of that season, with the same lap
  selection, the same Stage-A targets (`f1rank/race_targets.py`), the same team and
  teammate rows, and the same race-block bootstrap (`race_total.improvement`, 4,000
  resamples, seed 0).
- Baseline: v1's full model on the same training laps (its cached fits are reused when
  their keys match). The variant's predictions are v1's construction plus β·x for the
  target race, with β's posterior draws.
- Tables: team rows for every feature; teammate rows for `longrun` only (a car
  covariate is shared by teammates, so the other features cannot change a teammate
  gap).
- Gate, per feature and table: at least 12 held-out races; the 95% interval of the
  race-level MSE difference (variant − v1) lies entirely below 0; the variant's 90%
  intervals cover 85–95% of held-out results. The mean interval widths are reported.
- Four tests are run (`longrun` team and teammate, `traps` team, `upgrades` team), each
  with the published one-sided 2.5% error rate; no correction is applied, and this is
  stated with the results.

## Decision rule

- One run per feature. No definition, prior, fold or gate is changed after results are
  seen. A fit that fails convergence after its third attempt fails the test; nothing is
  retuned.
- No published table changes as a direct result. A feature that passes for team rows
  becomes a candidate for the car race-pace model, and one that passes for teammate rows
  a candidate for the driver race-pace model. Publishing either needs that table's own
  gate (`race_total.py`) under a new, separately pre-registered model version that
  includes every passing feature, and the scheduled refresh must then extract the
  feature's data for each new weekend.
- A feature that fails is recorded here and in the README and is not used.

## Amendment (2026-10-02, before the `longrun` and `traps` fits)

Made after the practice laps were extracted and the two practice covariates computed,
before any fit with either of them and without looking at any race result. The
`upgrades` test had already run under the text above, which this amendment does not
change for it.

The covariates' distributions contained values no car could produce: a team 19.5%
off the race-fuel median, and a team 60 km/h faster on the straight than the median
team. The laps behind the most extreme values (no race results were looked at) showed
two failures of the definitions above, not of the code:

- `longrun`: when cool-down laps are the majority of a stint (push, cool, push,
  cool), the stint's median is a cool-down lap, so the 7% rule kept them (2023-22,
  Zhou: a "run" averaging 110 s from laps starting at 85 s). **Amended:** laps more
  than 7% slower than the stint's *fastest* such lap are removed. Nothing else in the
  feature changes.
- `traps`: in a wet session teams set one or two laps on slicks (2024-04, second
  practice), so a team's 90th percentile came from one or two readings. **Amended:**
  the laps are the ones `longrun` uses before its run rule (timed, not in- or
  out-laps, not deleted, marked accurate, under green, on SOFT, MEDIUM or HARD tyres),
  and a team needs at least 10 of them for a value; otherwise its covariate is 0.

After the amendment the `longrun` team values range from −3.5% to +2.9% (sd 0.82, 796
team-weekends in 90 weekends), the teammate split from −2.5% to +2.5% (sd 0.46), and
`traps` from −2.0 to +1.3 per 10 km/h (sd 0.32, 989 team-weekends in 102 weekends).
The rules were chosen to remove these failures, not to improve any result. The test,
gate and decision rule are unchanged, and each feature is still run once.
