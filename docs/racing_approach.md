# Stage 2: racing ratings and an overall driver rating

## Total race-pace dashboard model (2026-09-30)

The dashboard extension uses a separate joint lap model, `race_total_model.py`,
to estimate total driver and car pace. It does not reinterpret the older
qualifying-adjusted `u` as total driver skill or use its historical gate results.
The model and these decisions were fixed before inspecting the new validation:

- Known-compound dry races from 2018 onward; the same clean-lap and incident-window
  rules as the race-pace work. Race entries supply team lineage even when a driver
  has no qualifying entry.
- Driver = lasting contribution plus a season-specific form deviation. Car =
  team-season package. Driver/car race-day deviations, race lap trend, compounds,
  degradation and traffic remain in the joint lap likelihood. AR(1) Student-t errors
  account for serial dependence and unusually slow laps.
  Driver and car pace are centred within season, and temporary effects within
  race. These identification constraints were added after a full fit failed to
  converge, before the held-out scores were inspected.
- No qualifying features or qualifying posterior point estimates enter this model.
  The driver prior has scale 0.5 percentage points and the car-season prior 1.5;
  season-form and race-day scales are estimated. These priors participate in the
  separation when team-switch evidence is weak.
- Headline is current-season pace at tyre age 10 laps, excluding temporary race-day
  effects. It is neither a pure innate-skill claim nor an equal-car championship.
- Train through 2024-10, 2025-10 and 2026-07; predict later dry races in the same
  season. Refit full, no-driver and no-car variants on the same training laps.
  Driver targets are Stage-A teammate gaps. Car targets are observed team means,
  not observations with this model's driver estimate subtracted.
- A table requires at least 12 held-out races, a race-bootstrap 95% interval for
  the squared-error difference entirely below zero, 85–95% coverage of the 90%
  predictive intervals, and sharper intervals than the corresponding ablation.
  Calibration includes race-day variability and the full shared uncertainty of
  freshly fitted test-race targets, resampling whole Stage-A bootstrap rows.
- Driver and car gates decide independently. Failed tables remain withheld, and
  stale artifacts stop publication. These mid-season tests establish neither
  transfer to an unseen team nor recovery of an exactly known causal skill.

This is a narrower dashboard addition than the full sequential championship plan
below. Degradation, consistency, starts, reliability, pit stops and overtaking do
not enter the new headline. Their existing research paths remain separate.

## Review correction (2026-09-30)

The build status and numerical gate results dated 2026-09-29 below are a historical
record. They need regeneration under these corrections:

1. Held-out season S uses qualifying fits trained before S for both training and
   test features. These are season-ahead forecasts conditional on the actual entrants
   and circuits; later qualifying observations cannot affect the historical test.
   Current full-data fits still use the main qualifying model.
2. Championship entry uses backward conditional ablation of the combined model,
   retesting after each removal. At least three inner test seasons are required.
   Each outer held-out season selects its qualities using earlier seasons only. The
   whole selection procedure must beat the base model before any racing qualities
   enter. Confidence intervals resample seasons, and qualities from the same fit
   share draw indices during predictive integration.
3. Cache fingerprints cover the entire race likelihood input. Fits must pass
   convergence and divergence checks before writing ratings. Provenance manifests
   connect inputs, code, output hashes and data cutoffs; stale or unversioned racing
   artifacts cannot be consumed by the championship or advertised by the report.

These are revised methods, not a new claim that the old positive results survive.
The README's regeneration instructions describe the required reruns. Race-specific
pace remains a lasting career-level effect beyond qualifying, and the equal-car
scenario remains experimental.

Status: every step built and run on data from 2010 (final runs 2026-09-28 and 2026-09-29;
see Build status below); results and gates are in `outputs/REPORT.md` (Racing). The approach was revised on 2026-09-27 after an
external review. Stage 1 (qualifying pace) is in `f1rank/` and described in the README.

## Summary

1. **Rate each racing quality separately, in the order the data can support it.**
   - Start with race pace and tyre degradation from 2018, estimated jointly.
   - Then reliability and errors, first-lap performance, and overtaking/defending.
   - Pit stops are rated as team operations, separate from car performance.
2. **Record what happened in each race as an event timeline with uncertain causes.**
   - This replaces one state per driver-race.
   - Every event keeps its laps, its evidence and how confident its cause is.
   - Each model derives its own exclusions from the timeline.
   - The uncertainty about causes flows into every downstream model.
3. **Build the overall rating as a sequential model of a race weekend:** qualifying →
   grid → race performance given the grid → incidents and retirements → classification
   and points. Simulating that sequence with equal machinery gives the headline
   **equal-car championship**. A simple results model is built early, as a benchmark
   every later part must beat.
4. **Test each quality in two ways.**
   - It gets a standalone ranking only if it beats a baseline that keeps the same car and
     context terms. The test uses held-out races, with the uncertainty of the
     improvement measured, plus transfer tests.
   - It enters the overall rating only if the full model predicts held-out races better
     with it than without it.
5. **Work at lap level.** Telemetry is deferred, but with named uses to test (see
   Granularity), not ruled out.

## What changed after the review

| Review point | Change |
|---|---|
| Fuel trend, tyre compounds and degradation are confounded | Start with 2018+ (known compounds). Estimate pace and degradation jointly with explicit identification constraints. Keep the pace/slope covariance. Model correlated laps. Check the two-stage shortcut against a joint model. Older data comes later, through a weaker observation model. |
| One state per driver-race is too coarse; detection rules too strong | Event timeline with evidence and cause probabilities, including an unknown category. The rules produce evidence, not verdicts. |
| Finishing-order weights are not causal values of skills | Sequential weekend model (qualifying → grid → race → incidents → points), simulated with equal machinery. The finishing-order model is demoted to a benchmark. |
| Driver errors could be counted twice | The race-performance stage excludes incident-affected periods; incidents come only from the incident model. |
| Residual term is not a completeness test | Kept as an "unexplained predictive contribution", included only if it improves held-out prediction. |
| Weak parts entering the overall rating | Nested with/without comparison on held-out races, carrying the part's estimation uncertainty. |
| Standalone gates too weak | Baselines keep car and context terms; uncertainty of the improvement; transfer tests; separate car checks; calibration and sharpness. |
| Qualities overclaimed | Renamed: first-lap performance; pit stops as team operations. Consistency includes car and context effects. Overtaking modelled by battle episodes. |
| Telemetry dismissed without evidence | Deferred with specific uses to test, and measured costs. |
| Evidence not reproducible; 2023 statuses misdescribed | Data check committed as `analysis/race_signal.py`, with robust statistics and intervals. Outlier-sensitive figures withdrawn. Status coverage restated. |

## Evidence from the data

Counts from the committed data (`data/processed/`):

- **Size:** 344 races and 7,241 driver-races since 2010.
- **Compromised races are common:** 31.0% of two-car team-races include at least one
  non-finisher.
- **Qualifying explains much of the race:** among 2,469 teammate pairs where both
  finished from grid slots, the one who started ahead finished ahead 70.1% of the time.
- **Retirement causes, as stored in Jolpica:**
  - Coded through 2022 ("Engine", "Collision", "Accident", ...).
  - 2023: 53 of 59 retirements are just "Retired"; 6 are coded.
  - 2024 onward: every retirement is "Retired".

  Status coding has already changed once (2023). Builds should therefore keep dated
  copies of the source responses and treat statuses as revisable, rather than assume a
  fixed boundary year.

From `analysis/race_signal.py` (FastF1, 2018+; definitions in the script; results in
`outputs/analysis/race_signal/`):

- **Coverage:** 187 races, 2018 to 2026 round 15. The 2018 Italian GP has no FastF1
  timing. Races with any rain or intermediate/wet tyres (48) are excluded from the pace
  comparisons.
- **Signal volume per full-season driver** (median):

  | Quantity | Median |
  |---|---|
  | Clean race laps | 932 |
  | Clean-air laps | 534 |
  | Battle laps (within 1 s) | 175 |
  | On-track passes made | 27 |
  | On-track passes suffered | 24 |

  - Race pace has far more data than qualifying (about 60 laps per season).
  - Overtaking rests on about 25–30 events per season.
- **Teammates are usually comparable:** in 85% of team-races both drivers have at least
  20 clean laps, and in 73% both have at least 10 clean-air laps.
- **Teammate race pace vs qualifying.** 69 pair-seasons with at least 8 dry races (993
  teammate race comparisons); medians per pair-season; bootstrap 95% intervals.

  | Statistic | Estimate | 95% interval |
  |---|---|---|
  | Race-pace gap, reliability over a season (split-half, Spearman–Brown) | 0.73 | 0.57–0.85 |
  | Qualifying gap, same | 0.81 | 0.74–0.86 |
  | Race-pace gap vs qualifying gap, correlation | 0.74 | 0.62–0.84 |
  | Race-minus-qualifying gap, reliability | 0.33 | 0.00–0.70 |

  So race pace is measurable and stable. Most of it is the same ability qualifying
  measures. A distinct race-specific skill is not established by this check: its
  stability is weak, with an interval reaching zero.

  For the build this means:
  - Race skill is linked to qualifying skill by an estimated coefficient.
  - The race-specific part is shrunk towards zero.
  - It gets its own ranking only if it improves held-out prediction.

**Withdrawn.** Earlier figures (race vs qualifying correlation 0.25; race-minus-qualifying
stability 0.63, from 17 pair-seasons) averaged per-race gaps with plain means. One
compromised session (Norris, 2023 round 19, −3.7 s) flipped a pair's qualifying gap.

The robust re-analysis above reverses their conclusion. Race and qualifying gaps are
strongly related, and the race-specific part is weakly stable at best. Even a stable
race-specific part would not by itself show a driver skill, because stable gaps can also
come from stable team strategy or roles.

## Data layer

- **Sources and versions.**
  - Jolpica (results, lap times and positions, pit stops, sprints).
  - FastF1 (2018+: tyres, track status, race-control messages, weather, sector times,
    speed traps).
  - Raw responses are cached with retrieval dates and a content hash. Each build
    records the source versions it used, as fits already record the data they were
    fitted on.
- **FastF1 runs as its own extraction step** in its own environment (it requires
  pandas < 3) and writes Parquet. Its API limit (500 calls per hour, about 50 races) makes
  the one-time 2018+ backfill about 4 hours; after that it is incremental.
- **Tables:**
  - race laps: time, position, gap ahead, compound, tyre age, stint, track status
  - stints, pit stops, race-control messages, results with status, weather
  - the event timeline

### Event timeline with uncertain causes

One row per event.

| Field | Content |
|---|---|
| race, driver, laps | the affected lap range (start, end or open) |
| kind | stoppage, retirement, suspected damage, off-track, collision noted, penalty, unexplained pace loss, pit anomaly, track status (SC/VSC/red/yellow), weather change, teammate swap |
| evidence | source, status code, race-control message text and time, stewards' document, reviewer note |
| cause probabilities | mechanical, own error, other driver, external, unknown |
| attribution confidence, reviewed | how firm the cause is; whether a person has checked it |
| source version | which snapshot of the source data it came from |

**Rules produce evidence, not verdicts:**
- **STOPPED message:** evidence of a stoppage, cause unknown. A solo spin also produces
  only STOPPED, so the rule "STOPPED with no incident message means mechanical" is wrong.
- **Stewards' decisions:** they judge infringements, not necessarily who caused the
  outcome. A penalty raises the probability of fault; it does not settle it. No penalty
  does not split fault 50/50: the attribution stays uncertain.
- **Teammates within 2 s, or swapping places:** flagged as a possible team order, for
  sensitivity analysis only.
- **Lasting pace loss:** recorded as "unexplained pace loss". It could be damage,
  tyres, fuel or energy saving, or an engine mode.

**Using it downstream:**
- Each model derives its exclusions from the timeline.
- Uncertain causes are propagated. Each posterior draw or simulation run samples causes
  from their probabilities. Strict and lenient sensitivity sets are also run.
- The unknown category stays.
- Reviewed entries live in an override file with evidence notes, and are never
  re-inferred.
- An audit report per build lists the cause assignments and what changed since the last
  build.

## The qualities

| Quality | Treatment |
|---|---|
| Race pace | Build first. Rated under stated standard conditions: green flag, clean air, dry, slicks, reference tyre age, relative to the field at that race. |
| Tyre management | Estimated jointly with pace, since going slower early can buy lower degradation. Report pace and degradation together, with their correlation. |
| Consistency | Lap-time spread explained by race, car (team-season), context (traffic, tyre age, track status) and driver. The driver part is reported only if it passes the gates. |
| First-lap performance | Positions gained from the grid slot to the end of lap 1. Includes launch, corner fighting and incidents; named as such, not "starts". Telemetry could later separate the launch. |
| Overtaking and defending | Battle episodes with opportunity counts, accounting for repeated laps against the same opponent. Must first pass a synthetic feasibility check. |
| Errors and reliability | Competing risks with exposure (laps at risk) and uncertain causes. Time at risk before another kind of retirement is kept. |
| Pit stops | Team operations rating, separate from car performance and driver ratings. |
| Wet weather | Driver × wet interaction. Experimental: few wet races. |

### Race pace and degradation (first build, 2018+)

Per clean lap (log ratio to the race reference):

```
lap = race lap trend (fuel burn + track evolution, common to all cars)
    + compound (race)
    + [car degradation + driver degradation] x tyre age
    + car race pace + driver race pace + team-specific effect
    + dirty-air term (gap to the car ahead)
    + noise (heavy-tailed, AR(1) within a stint)

driver race pace = gamma x qualifying skill (estimated)
                 + race-specific part (shrunk towards 0)
```

The data check found race and qualifying gaps strongly related (0.74), with a
race-specific part that is weakly stable at best. The race-specific part is therefore
published only if it improves held-out prediction over the qualifying link alone.

**Identification, made explicit:**
- **Fuel vs common degradation.** Within a stint, tyre age and lap number rise together.
  The common fuel trend and the common part of degradation can only be told apart
  through stint resets and compound changes.
  - Constrain the fuel effect: common per race, with an informative prior.
  - Report degradation relative to the race's common slope, which is identified.
  - Sensitivity: fuel effect fixed vs free.
- **Fuel differences between cars** (starting loads, consumption) are not removed by a
  shared trend. They fall into car effects, which teammates share; sensitivity checks.
- **Compound choice is not random.** It depends on strategy, grid position and the car.
  - Compounds are known from 2018 and enter as fixed effects.
  - Strategy and grid covariates are added.
  - A check compares drivers on the same compound at similar tyre age.
- **Drivers not pushing:**
  - an unpressured covariate (large gap both ahead and behind)
  - a sensitivity variant without the final third of each race
  - later, a coasting flag from telemetry

**Scale.** A two-stage shortcut summarises each stint:
- pace at a reference tyre age and degradation slope, with their full 2×2 covariance
- errors from an AR(1) or block-bootstrap model, so correlated laps are not treated as
  independent
- shared race-day conditions as race-level random effects in stage 2

It is accepted only after comparison with the joint lap-level model on one or two
seasons. The test: driver and car posteriors agree within a quarter of a posterior SD.

**Before 2018** (later), unknown compounds get a weaker observation model: latent or
stint-level effects with larger variance, calibrated and validated separately. Equal
measurement quality across the whole window is not assumed.

### Reliability and errors

- **Model:** a discrete-time competing-risks hazard per lap. The risks are mechanical
  failure, own error, other driver, and unknown.
- **Uncertain causes:** each retirement contributes a mixture over its timeline cause
  probabilities.
- **Censoring:** a retirement from one cause censors the others, and the laps at risk
  before it are kept.
- **Mechanical risk:** team-season, power-unit supplier (shared by customer teams) and
  era.
- **Own-error risk:** a driver effect (slow random walk), plus team, wet and traffic
  density.

### First-lap performance

```
positions gained on lap 1 = grid-slot expectation (slot, clean/dirty side, start compound)
                          + car (team-season) + driver + heavy-tailed noise
```

- Incidents enter through the timeline, with their uncertainty.
- Pit-lane starts are excluded.

### Overtaking and defending

- **Episodes.**
  - A battle episode starts when a car gets within 1.0 s of the car ahead, under green
    flag and with neither car pitting.
  - It ends with a pass, the gap growing beyond about 2 s, a pit stop or a neutralisation.
  - Excluded: teammates, lapping, and passes made through pit stops.
- **Model:** the per-lap chance of a pass within an episode, depending on:
  - the pace difference (from the race-pace model)
  - the tyre-age and compound gap
  - the overtaking-aid state (DRS to 2025, the 2026 overtake mode)
  - circuit difficulty and car straight-line speed
  - attacker and defender effects
  - a pair effect for repeated episodes between the same two cars
- **Opportunity:** reported separately as the number and length of episodes (mostly a
  consequence of pace).
- **Feasibility first:** a synthetic check at real sample sizes (about 30 passes per
  driver per season) comes before any real fit.

## The overall rating: a sequential weekend model

1. **Qualifying performance** (stage 1) → **grid**. Grid penalties come from the
   timeline. In equal-machinery simulation, power-unit penalties are equalised.
2. **Race performance given the grid:**
   - first lap
   - stint pace and degradation
   - traffic and overtaking/defending
   - pit stops (team operations)
   - strategy (a team decision, held at typical strategies in simulation)
3. **Incidents and retirements** from the competing-risks model.
4. **Classification and points.**

Each stage is fitted on its own data with its own validation. The whole chain is then
validated on held-out races (finishing order, teammate head-to-heads, points) against the
simple results benchmark.

- **No double counting.** The race-performance stage is fitted on laps and positions
  outside incident-affected periods, and incidents enter only through the incident
  model.
- **Mechanical failures are not simply deleted.** Pre-failure laps and exposure stay in
  every stage, and classification handles censoring. Sensitivity: retirement treated
  as uninformative vs related to performance.
- **Equal-car championship.**
  - Every driver gets the same average car and team operations (reliability, pit stops).
  - The full sequence is simulated, so grids, traffic and passing opportunities change
    consistently.
  - Outputs, from posterior draws: expected points per race, P(title), rank ranges.
- **Contribution breakdowns are conditional model estimates.** Example: the change in
  expected points when one quality is set to the field average, with the others at the
  driver's values. Correlated qualities have no unique allocation of credit. The
  breakdown reports both one-at-a-time and all-at-once differences and says so.
- **The unexplained predictive contribution** (the residual driver term) is kept only if
  it improves held-out prediction. It can absorb strategy, team support, missed grid
  effects or misclassified incidents, and shrinkage can make it small even when
  components are missing. So its size is not read as completeness.

**Simple results benchmark (built early).** A rank-ordered logit on finishing orders, with
car strength and driver terms (stage-1 ratings plus a driver results effect). It gives
every later part a concrete target: does adding it improve held-out results?

## Validation and gates

**Standalone ranking for a quality:**
- The baseline keeps the same car and context terms, and removes only the driver terms.
- The improvement is measured on outer held-out races, as paired differences per race.
  Its uncertainty comes from a block bootstrap over races, and the interval must exclude
  zero.
- Transfer tests: drivers who change team, and new pairings.
- Car predictions are evaluated separately.
- Calibration (90% intervals cover 85–95%) and sharpness: intervals must be narrower than
  the baseline's. Wide intervals can pass a coverage test without being useful.
- Split-half stability is reported, but it is not sufficient: stable confounding is also
  stable.
- Synthetic recovery across several independent seeds and parameter settings, not one
  shared truth.

**Entering the overall rating:** a nested comparison of the complete model with and
without the quality, on outer held-out races. The comparison uses the quality's
posterior draws, not point estimates, so its estimation uncertainty is carried through.
A quality can be too uncertain for its own ranking and still help here. That is fine,
but it has to show it.

## Granularity

| Level | Decision | Reason |
|---|---|---|
| Lap | core | Smallest unit where teammates face comparable conditions; public back to 2010. |
| Race events | via the timeline | First lap, incidents, reliability, pit stops, exclusions |
| Sector times, speed traps (2018+) | optional inputs | Speed traps: car straight-line speed for overtaking; sectors: traffic detection |
| Telemetry (2018+) | deferred, with uses to test | See below |

**Telemetry.**

Measured on the 2024 Bahrain race and qualifying:
- about 7 s to load a session
- about 700k car-data rows per race, sampled every 0.24 s
- braking recorded only as on/off
- about 70 MB of raw cache per session, so about 30 GB for all 2018+ race, qualifying
  and sprint sessions

It would be reduced once to a few numbers per lap, so model size does not change. Compute
is not the obstacle.

Uses to test, each by whether it improves held-out prediction:
1. **A coasting / not-pushing flag for race pace.** In the Bahrain test, Sainz coasted
   about 0.6 s per lap and Leclerc about 0.1 s. The signal is there, but its meaning is
   ambiguous, so it enters as a covariate, not a skill.
2. **Compromised qualifying laps** (yellow-flag lifts, mistakes, traffic), in place of
   the blunt 5% cut.
3. **Battle states within a lap:** failed attempts, passes undone within the lap, and
   overtaking-aid use.
4. **Launch vs first-lap fighting.**
5. **The gap to the car ahead through corners**, for dirty air.

Not for rankings: braking points and corner speeds. They are dominated by the car and
setup, and too coarse at about 4 samples per second. There is also no steering or
energy-deployment channel, which matters under the 2026 rules.

## Build order

1. **Update and reproducibility fixes.**
   - Stage 1: fit metadata and checks, explicit synthetic source, placebo job, write-once
     snapshots, failures that stop the run; entered drivers without a time kept; team-effect
     variants per spell and per regulation era; synthetic recovery over 8 independent truths.
   - Sources: raw Jolpica responses versioned (changed files archived with their retrieval
     time; `data/processed/sources.json`); FastF1 tables with retrieval time and hashes.
2. **Event timeline and reproducible audit.**
3. **2018+ race pace and degradation**, including the two-stage vs joint check.
4. **Reliability and errors.**
5. **Simple overall results benchmark.**
6. **First-lap performance.**
7. **Overtaking and defending**, after the synthetic feasibility check.
8. **Equal-car championship** (sequential simulation).

Each step adds its validation section to `REPORT.md` and has to pass its gates.

## Build status (2026-09-29)

Every step is built and every quality has had its final run on the data from 2010. Numbers
are in `outputs/REPORT.md` (Racing); this section records what was built, what the gates
said, and where the build departs from the plan. The decisions for the final runs were
fixed in advance (next section).

| Step | Built | Gate result (held out) |
|---|---|---|
| Data from 2010 | `oldlaps.py` (Jolpica laps, 2010-2017), sprints, `extract/telemetry.py`, `powerunits.py`, `conditions.py` | Inferences checked against FastF1 (`outputs/analysis/`) |
| 2. Event timeline | `timeline.py`; audit in `outputs/timeline/AUDIT.md` | Retirement-cause model beats base rates held out a season at a time |
| 3. Race pace, degradation | `racepace.py` (stage A), `racemulti.py` (one joint lap-level model across seasons) | **Both pass** (held out 2012-2026, the deciding run). Split by era (descriptive, after the gate): race-specific pace improves from 2018 but not in 2012-2017 alone; degradation improves in 2012-2017 (unknown compounds) but not from 2018, and not in the 2018-only variant |
| 4. Reliability and errors | `reliability.py` | Driver error rates, team-season and supplier-season mechanical effects: none passes |
| Pit stops | `pitstops.py` | **Passes**: team operations rating |
| Consistency | `consistency.py` | Fails |
| Wet weather | `wetpace.py` (experimental) | Fails (15 wet races) |
| 5. Results benchmark | `benchmark.py` | A driver results effect is not established |
| 6. First-lap performance | `firstlap.py` | Standalone ranking fails with the lasting team term (the pre-specified final model); the lasting team term itself improves prediction |
| 7. Overtaking | `battles.py`, `overtaking.py` | Not feasible: attacker and defender effects both below the recovery criterion |
| 8. Equal-car championship | `championship.py` | Entry tests (held out 2012-2026, 306 races), interval above zero to enter: **first-lap performance and race-specific pace enter**; degradation and consistency do not; overtaking has no held-out draws |

**Where the build departs from the plan, and why:**
- **Timeline evidence that changes meaning over time is not used for causes.** "CAR n
  STOPPED" messages almost vanish after 2019 and "spun" messages after 2023, while
  track-limits messages surge from 2020. The cause model uses only evidence whose rate is
  stable (lap 1, contact incidents naming the car, other cars retiring nearby, a
  neutralisation, stopping in the pits, slow laps before stopping). Rates by season are in
  the audit.
- **Retirement evidence starts in 2018.** Retirements before 2018 use the coded Jolpica
  status only. Retirements coded only "Retired" without FastF1 evidence get the class base
  rates.
- **Race laps use the leader's lap count.** Race-control messages refer to the leader's lap,
  so every timeline row does too; rows from a car's own laps also keep that car's lap.
  FastF1 and Jolpica sometimes disagree on a car's completed laps; Jolpica's count is used.
- **Uncertain causes enter the hazard model as fractional events**, not a free mixture.
  With a free mixture, the "unknown" hazard (which gets some probability for every
  retirement) absorbed the lap-1 events and the cause-specific rates were not identified.
- **What Jolpica lacks before 2018 is inferred, and checked.** Neutralised laps come from
  the field's lap times, 2010 pit stops from lap-time losses (pit stops are recorded from
  2011), wet races from hourly precipitation at the circuit (a weak proxy: it finds about
  60% of FastF1's wet races, and 64% of the races it flags are wet). Models give
  proxy-wet races their own coefficient; wet pace uses FastF1 races only.
- **Race pace: a joint model, not the two-stage shortcut.** On 2024 the two-stage
  estimates did not agree with the joint lap-level model within a quarter of a posterior
  SD, so the shortcut's driver estimates are not published; `racemulti.py` pools every dry
  race in one likelihood instead. Before 2018 the compound terms are replaced by stint
  offsets with their own noise scale (k_old). k_old is only in fits that also have FastF1
  laps: with Jolpica laps alone (the held-out fits for 2012-2018) it is not identified
  apart from the common noise scale, and a timing check on 2010-2011 showed the sampler
  taking 631 leapfrog steps per iteration with it and 255 without. This was changed
  before any held-out result of that run was computed.
- **First-lap performance has a lasting team term.** The first build's driver effects
  failed the transfer test; the final model (fixed in advance) adds a lasting team term.
  With it, the driver effects no longer improve held-out prediction, while the team term
  does: what the driver terms picked up in the first build is explained by a lasting team
  effect. Without the team term the driver effects would pass both conditions; the model
  with it was fixed in advance as the one that decides.
- **Overtaking: not feasible, including attacker-only.** With the older seasons and
  sprints, the estimated attacker and defender effects are smaller than in the 2018-only
  data, and neither is recovered at the required correlation in simulation. The
  attacker-only test fixed in advance applies only if attackers pass alone; they do not.
- **Entry test from 2010.** The quality models now start in 2010, so the race stage in
  the entry test is fitted from 2010 up to each held-out season (from 2018 in the first
  build). Degradation, consistency and overtaking get the entry test too, when they have
  held-out draws.
- **Pit stops are held out from 2019.** The 2011-2017 durations (Jolpica) are used in
  training only; the test design was fixed when the data started in 2018.
- **Car terms in every driver test.** The first overtaking model had no car terms; it was
  stopped before its results were used and refitted with attacking-car and defending-car
  team-season effects, as the gate rule requires.
- **Race pace: an era split, added after the gate result.** The pooled intervals decide
  and both pass. The split by held-out era was added afterwards because the 2018-only
  variant disagreed on degradation. It shows that degradation's pass comes from the
  2012-2017 held-out seasons, where compounds are unknown and a teammate difference in
  degradation can include a difference in compound choice; from 2018 its interval includes
  zero. Race-specific pace is the reverse: it improves from 2018, not in 2012-2017 alone.
  Both are published as the gate rule says, with this caveat in the report.
- **What the entry tests do not show.** A quality that enters improves held-out finishing
  orders of the race stage; it does not show the term is a driver's skill rather than
  something correlated with it (team favouritism, strategy priority). Race-specific pace's
  entry effect is large next to first-lap's (its coefficient on the held-out draws is
  positive with a 90% interval above zero in every held-out season from 2013), and degradation
  passes its own gate without improving finishing orders.
- **Computation.** Long held-out runs are checkpointed per season and can be fitted by
  parallel workers; this does not change the fits (each has a fixed seed). The
  championship's entry tests clear JAX's compilation caches between seasons: without that, the
  final run reached its 4 GB memory cap.

## Decisions fixed before the final runs (2026-09-28)

Written down before the 2010-2017 Jolpica lap data were processed, so that the final results
cannot shape them:

- **First-lap performance.** The final model is the one with a lasting team term, sprint
  starts, and race starts from 2010 (Jolpica lap-1 positions before 2018); the held-out test
  starts in 2012. Its standalone ranking gate is unchanged: overall interval above zero, and
  the interval for drivers who changed team not entirely below zero. An interim run on
  2018-2026 with sprints passed that second condition by 0.001 (upper end of the interval);
  that run does not decide. Whatever the final run shows is reported.
- **Race pace.** The multi-season joint model (racemulti.py) decides the race-specific pace
  and degradation gates; with the 2010-2017 laps (weaker observation model: unknown compounds,
  stint offsets) the held-out test starts in 2012. The gate rule is unchanged (interval of the
  squared-error difference below zero).
- **Consistency.** The final run adds 2010-2017 races through a Jolpica-based stage A (no
  compound terms); held-out from the third season, same rule. An interim 2018-2026 run
  (interval -0.0070 to +0.0003) does not decide.
- **Overtaking.** The final episodes add sprint races and 2010-2017 races (Jolpica: no
  compounds or speed traps; neutralisations and, before 2012, pit stops inferred), with an
  indicator for each source and for the pre-DRS season (2010). The feasibility check is
  rerun with its criteria unchanged; if it passes, the held-out test starts in 2012.
- **Entry into the overall rating.** Every quality with held-out draws gets the nested entry
  test in championship.py, as the approach specifies; a quality enters if the interval is
  above zero. Attacker-only overtaking ratings, if tested, are labelled as a rule chosen
  after defender effects failed their feasibility check.

## Risks

- **Drivers not pushing** biases race pace. Mitigations and checks above; telemetry is
  the strongest candidate fix.
- **Team favouritism** (strategy priority, orders) looks like driver skill in results.
  It is partly visible in the unexplained contribution, but not separable.
- **Inferred causes can be wrong.** The timeline keeps uncertainty and an audit trail,
  and sensitivity sets show how much it matters.
- **Overtaking and errors may not be measurable precisely enough.** The gates decide.
- **The sequential model has many parts.** The early results benchmark keeps it honest:
  every added part must improve held-out results.

## Amendment (2026-10-02, before the regenerated racing runs)

The racing held-out tests take each test season's qualifying features from its fold,
`quali_fold<S>`, fitted under the fixed retry rule (`jobs.ATTEMPTS`). The 2021 fold is
the same fit as the qualifying validation's `lfo_end2020`, which failed all four
attempts on 2026-10-02 (R-hat 1.106, 1.072, 1.066, 1.187; the check is 1.05), so it is
not refitted with other settings and the fold has none. The 2014 fold, the same fit as
`lfo_end2013`, failed the same way later that day (R-hat 1.152, 1.197, 1.068, 1.077,
with 74, 145, 230 and 46 divergences). Nothing above said what a racing test does
without its fold.

Decided before any racing result under the corrected procedure: a test season whose
fold failed every attempt (`quali_fold<S>.failed.json`, no fit) is left out of every
racing held-out test that uses qualifying features (race-specific pace and degradation,
wet pace, the results benchmark, the championship's race-stage check and its entry
tests). Each output lists it under `excluded_unconverged_qualifying_folds`, and its
manifest records the failure. The season's races still train later seasons' folds. The
rule depends only on the qualifying fit, as the qualifying validation's exclusion of
unconverged cutoffs (`lfo_summary.json`) does, and no gate changes.

## Amendment (2026-10-04, before any score of the regenerated racing run was inspected)

The regenerated racing run (2026-10-04, 10:06 to 14:30) ended with one fit failing the
publication convergence check: the full fit in `firstlap` (R-hat 1.067 against the check
of 1.05, no divergences, 4,000 draws). The racing-quality fits make a single attempt
with a fixed seed. Unlike the qualifying fits, which use the retry rule `jobs.ATTEMPTS`
fixed before their runs, nothing above said what a racing-quality fit does when its one
attempt does not converge. The equal-car championship needs the draws of every quality
that enters, so `firstlap`'s failure stopped it (`outputs/firstlap/manifest.json` was
never written).

The only information used here is the failure message in the run's log. No racing score
from that run was read, and no output of that run is used: every racing output is
regenerated after the code change below, because the change alters the code hash that
each manifest records.

Decided before any racing score under the corrected procedure: every MCMC fit in the
racing-quality modules (`firstlap`, `consistency`, `reliability`, `pitstops`, `wetpace`,
`benchmark`, `overtaking` and the championship's race stage) that fails the publication
check is retried under one fixed ladder, `artifacts.RETRY`, relative to that fit's own
settings. The ratios are those of `jobs.ATTEMPTS` (700+400, 1,400+800, 2,100+1,600,
2,100+1,600 at target acceptance 0.98):

| attempt | warm-up | draws per chain | target acceptance | seed |
|---|---|---|---|---|
| 1 | as before | as before | as before | as before |
| 2 | 2 x | 2 x | as before | +1 |
| 3 | 3 x | 4 x | as before | +2 |
| 4 | 3 x | 4 x | 0.98 | +3 |

The first attempt is the fit made before this amendment, so a fit that already
converged is unchanged. The check, its thresholds, the models, the data, the gates and
the entry tests do not change. If all four attempts fail, the fit fails as before: that
quality is not published and nothing is retuned. The race-pace model (`racemulti`) is
not covered: every one of its fits converged in the run above. Each summary records the
number of fits and the attempts of any fit that needed more than one, with its
diagnostics (`fit_attempts`); attempts are also printed to the run's log.

## Regeneration completed (2026-10-05; results from the 2026-10-04 run)

This is an outcome record, not a change to the procedure. The canonical racing
outputs and championship manifests are current. All lanes finished with exit status
zero. Standalone gates pass for race-specific pace, first-lap performance and pit
operations; degradation, consistency, wet pace, reliability and results effects fail.
Overtaking and defending fail feasibility. The conditional entry test selects
race-specific pace, but the outer selection test fails (+0.125 log predictive density
per race; 95% interval −0.030 to +0.293, 207 races). No racing quality enters, and
the Overall dashboard withholds its experimental equal-car ranking. Sources:
`outputs/championship/summary.json` and the quality summaries listed in the README.

The amendments dated 2026-10-02 and 2026-10-04 remain unchanged: failed qualifying
folds exclude test seasons 2014 and 2021 where their features are required, and
racing-quality fits retry on the fixed ladder. None of the results caused retuning.
The three older FastF1-only race-pace sensitivity outputs remain stale and excluded
from the generated report; the report's stale-file note refers to those
secondary outputs.

### Runtime record

Source: `outputs/analysis/racing_regeneration/run.json`, transcribed from the lane
logs with their hashes. Elapsed seconds are wrapper wall times, including cache
loading and output work; reused fits' earlier sampling times are separate in the
model summaries. The f1 slice used CPUQuota=800%, CPUWeight=10 and Nice=19, with
a 13G memory admission budget. These are local measurements. CI runner time and
memory remain unverified, as does the cause of the earlier WSL OOM.

| Lane | Stage | Elapsed seconds |
|---|---|---:|
| rgpu1 | racemulti 2026 | 11 |
| rgpu1 | racemulti 2024 | 9 |
| rgpu1 | racemulti 2022 | 7 |
| rgpu1 | racemulti 2020 | 9 |
| rgpu1 | racemulti 2018 | 9 |
| rgpu1 | racemulti 2016 | 5 |
| rgpu1 | racemulti 2014 | 6 |
| rgpu1 | racemulti 2012 | 7 |
| rgpu2 | racemulti 2025 | 11 |
| rgpu2 | racemulti 2023 | 11 |
| rgpu2 | racemulti 2021 | 4 |
| rgpu2 | racemulti 2019 | 6 |
| rgpu2 | racemulti 2017 | 8 |
| rgpu2 | racemulti 2015 | 8 |
| rgpu2 | racemulti 2013 | 6 |
| rgpufinal | racemulti heldout | 48 |
| rgpufinal | racemulti full | 12 |
| rcpu | firstlap | 1007 |
| rcpu | consistency | 38 |
| rcpu | reliability | 1067 |
| rcpu | pitstops | 192 |
| rcpu | battles episodes | 66 |
| rcpu | battles feasibility | 1396 |
| rcpu | battles fit | 192 |
| rcpu | wetpace | 43 |
| rcpu | benchmark | 305 |
| rfinal | championship | 965 |
| rfinal | report | 25 |

### Every racing fit that needed a retry

Source: `fit_attempts.retried` in `outputs/firstlap/summary.json`, and `_attempts`
in `outputs/race/multi_heldout.json`; the complete records and fit labels are in
`outputs/analysis/racing_regeneration/run.json`. First-lap labels follow the call
order in `f1rank/firstlap.py` and the logged attempts. Repeated comparisons are
separate fit calls, retained here rather than deduplicated. All listed fits passed
on their second attempt. The first-lap full fit doubled per-chain warm-up and draws
from 1,000 to 2,000; its other retries doubled 500 to 1,000. Seeds changed from
0 to 1. Their divergences were zero on both attempts. Race-pace retries reused the
existing checked caches and doubled warm-up from 500 to 1,000.

| Module and fit | First R-hat | Retry R-hat | Divergences, first → retry |
|---|---:|---:|---:|
| firstlap: full model with lasting team | 1.066966 | 1.012407 | 0 → 0 |
| firstlap: held-out 2013: no driver effects, lasting_team=False | 1.060797 | 1.014541 | 0 → 0 |
| firstlap: held-out 2017: driver effects, lasting_team=True | 14012.129883 | 1.008298 | 0 → 0 |
| firstlap: held-out 2017: no driver effects, lasting_team=True | 1.069297 | 1.010812 | 0 → 0 |
| firstlap: held-out 2020: driver effects, lasting_team=True | 2621.858643 | 1.017643 | 0 → 0 |
| firstlap: held-out 2020: no driver effects, lasting_team=True | 1.052402 | 1.022516 | 0 → 0 |
| firstlap: held-out 2021: no driver effects, lasting_team=True | 1.051789 | 1.028609 | 0 → 0 |
| firstlap: held-out 2022: driver effects, lasting_team=True | 43047.050781 | 1.006199 | 0 → 0 |
| firstlap: held-out 2024: driver effects, lasting_team=True | 36699.667969 | 1.010296 | 0 → 0 |
| firstlap: held-out 2024: no driver effects, lasting_team=True | 116100.804688 | 1.025298 | 0 → 0 |
| firstlap: held-out 2025: driver effects, lasting_team=True | 1.057520 | 1.017843 | 0 → 0 |
| firstlap: held-out 2013: no driver effects, lasting_team=False | 1.060797 | 1.014541 | 0 → 0 |
| firstlap: held-out 2017: no driver effects, lasting_team=True | 1.069297 | 1.010812 | 0 → 0 |
| firstlap: held-out 2020: no driver effects, lasting_team=True | 1.052402 | 1.022516 | 0 → 0 |
| firstlap: held-out 2021: no driver effects, lasting_team=True | 1.051789 | 1.028609 | 0 → 0 |
| firstlap: held-out 2023: driver effects, lasting_team=False | 149410.671875 | 1.008373 | 0 → 0 |
| firstlap: held-out 2024: no driver effects, lasting_team=True | 116100.804688 | 1.025298 | 0 → 0 |
| racemulti: held-out 2013 (reused checked cache) | 1.050901 | 1.027500 | 1 → 2 |
| racemulti: held-out 2015 (reused checked cache) | 302043.687500 | 1.032805 | 0 → 2 |
| racemulti: held-out 2023 (reused checked cache) | 1.075509 | 1.017789 | 0 → 0 |

No retries were needed by consistency, reliability, pit operations, wet pace,
benchmark, battles fit or championship. The battles feasibility record remains
separate from the quality-fit summary. The canonical race full fit needed one
attempt. The results remain as of 2026-15; the scheduled refresh has not run on
GitHub and still needs the approved merge and cache-release setup.

The first Python check failed because the separate total-race-pace export still
fingerprinted the previous `artifacts.py`. Running its prescribed validation and
export command reused every checked posterior fit, regenerated the manifests and
preserved its gate decisions. The rerun passed all Python checks. This did not
change model settings, code or the canonical racing manifests.

The clean-clone check also exposed a publication problem: rerunning the exporter
without ignored posterior files relabelled current racing evidence as stale. The
dashboard workflow now verifies the committed immutable publication and its portable
receipt instead. The receipt is recorded on the fitting machine only after comparison
with the fully checked exporter, and binds the payload to the current source inventory
and hashes. The scheduled refresh records and commits its new receipt. This changes
publication tooling and workflows, with no change to model code, fits or gates.

The published `outputs/REPORT.md` received an editorial review: its stale warning
is scoped to the secondary files, and the simulated research standings and
contributions are withheld under the failed combined gate. The raw CSVs remain
committed for audit. Regenerating the Markdown with the unchanged report generator
requires applying this publication review again before publishing that document.

Final local verification is recorded in
`outputs/analysis/racing_regeneration/run.json`: the Python suite passed 148 tests,
and the local browser run passed 67. A fresh clone with Python 3.12.3 and Node
22.20.0, a new Python environment and `npm ci`, passed the portable-publication
check, 49 dashboard/publication Python checks, the production build with the
repository URL prefix, and 67 browser tests. It had no posterior caches and
preserved current racing rows and the not-established Overall status. Host browser
libraries were already available; this does not measure a GitHub runner or deploy
the site.
