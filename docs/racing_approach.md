# Stage 2: racing ratings and an overall driver rating

Status: first build of every step done on 2026-09-28 (see Build status below); results and
gates are in `outputs/REPORT.md` (Racing). The approach was revised on 2026-09-27 after an
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

## Build status (2026-09-28)

Every step has a first build. Numbers are in `outputs/REPORT.md`; this section records what
was built, what the gates said, and where the build departs from the plan.

| Step | Built | Gate result |
|---|---|---|
| 2. Event timeline | `f1rank/timeline.py`; audit in `outputs/timeline/AUDIT.md` | Retirement-cause model beats base rates held out a season at a time |
| 3. Race pace, degradation | `racepace.py` (stage A), `racemodel.py` (stage B), `racejoint.py` | Race pace tracks qualifying (gamma about 1). Race-specific pace and degradation effects do **not** improve held-out prediction. The two-stage shortcut is **not accepted** by the joint check (its driver estimates are not published); a held-out test with the joint model gives the same answer on race-specific pace |
| 4. Reliability and errors | `reliability.py` | Driver error rates and team-season mechanical effects do **not** improve held-out prediction |
| 5. Results benchmark | `benchmark.py` | Grid + ratings is the benchmark to beat (better than the grid alone); a driver results effect is not established |
| 6. First-lap performance | `firstlap.py` | Improves held-out prediction overall but **fails the transfer test** (worse for drivers in a new team): not published as a driver ranking |
| 7. Overtaking | `battles.py` (episodes), `overtaking.py` | **Not feasible** at real sample sizes: attacker effects are recovered in simulation, defender effects not well enough, so overtaking is reported as opportunity counts only |
| 8. Equal-car championship | `championship.py` | Race stage beats grid-only and ratings-only on held-out finishing orders; only gated qualities enter |

**Where the build departs from the plan, and why:**
- **Timeline evidence that changes meaning over time is not used for causes.** "CAR n
  STOPPED" messages almost vanish after 2019 and "spun" messages after 2023, while
  track-limits messages surge from 2020. The cause model uses only evidence whose rate is
  stable (lap 1, contact incidents naming the car, other cars retiring nearby, a
  neutralisation, stopping in the pits, slow laps before stopping). Rates by season are in
  the audit.
- **Outcomes go back to 2010; evidence starts in 2018.** Retirements before 2018 use the
  coded Jolpica status only. Retirements coded only "Retired" without FastF1 evidence (a few
  before 2018) get the class base rates.
- **Race laps use the leader's lap count.** Race-control messages refer to the leader's lap,
  so every timeline row does too; rows from a car's own laps also keep that car's lap.
  FastF1 and Jolpica sometimes disagree on a car's completed laps; Jolpica's count is used.
- **Uncertain causes enter the hazard model as fractional events**, not a free mixture.
  With a free mixture, the "unknown" hazard (which gets some probability for every
  retirement) absorbed the lap-1 events and the cause-specific rates were not identified.
  The first run also left own-error risk without a team term, against the gate rule that
  baselines keep car terms; with both fixed, the driver error-rate gate fails (it had
  passed).
- **Race pace needs a joint model for published numbers.** On 2024 the two-stage estimates
  agree with the joint lap-level model in order but not within a quarter of a posterior SD:
  the joint model's driver effects are more spread out (less shrinkage; ratios in the
  REPORT). A held-out test with the joint model, trained one season at a time, reaches the
  same conclusion as the two-stage test (no race-specific pace). A multi-season joint model
  is still needed before any race-pace numbers are published.
- **The equal-car championship folds first-lap, overtaking and strategy into a
  grid-to-finish ranking model** fitted on real races, instead of simulating them lap by lap,
  because no driver-specific racing quality has passed its gates. Retirements use the
  reliability model with equal mechanical risk and average driver error rates.
- **Car terms in every driver test.** The first overtaking model had no car terms; it was
  stopped before its results were used and refitted with attacking-car and defending-car
  team-season effects, as the gate rule requires.
- **Transfer tests decide standalone rankings.** First-lap driver effects improve held-out
  prediction overall but make it worse for drivers in a new team, so they look tied to the
  driver-team combination and are not published as a driver ranking.
- **Not built yet:** power-unit supplier effects (no supplier data in the sources), wet
  races and traffic in the error model, pit stops as team operations, the sprint races, a
  multi-season joint race-pace model, and a separate feasibility check for attacker-only
  overtaking ratings (attacker effects were recovered in simulation; defender effects were not).

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
