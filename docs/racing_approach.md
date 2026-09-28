# Stage 2: racing ratings and an overall driver rating

Status: proposed approach, not implemented (2026-09-27). Stage 1 (qualifying pace) is
in `f1rank/` and described in the README.

## Summary

1. **Split racing into parts that can each be measured and separated from the car**, and
   rank each part on its own:
   - Driver: race pace, tyre management, starts, racecraft (overtaking and defending),
     consistency and errors.
   - Car: race pace, tyre wear, reliability, pit stops.

   Every part uses the same logic as the qualifying model: teammates share a car, and
   drivers who change teams link the cars together.
2. **Publish a part's ranking only if the data supports it.** It must beat a baseline with
   no driver skill on races it has not seen, and agree with itself between two halves of
   the data. Parts that fail (likely candidates: racecraft, wet weather) still feed the
   overall rating, but are not shown as rankings of their own.
3. **Build the overall rating from race results, not from chosen weights.** A
   finishing-order model learns how many positions each part is worth, using about 340
   races since 2010.
   - A leftover driver term catches whatever the parts miss. Its size shows how
     complete they are.
   - Headline: an **equal-car championship**, the expected results if every driver had
     the same car.
4. **Classify compromised races once and apply the result everywhere.** Every
   driver-race gets a state (mechanical failure, damage, crash, penalty, safety car, ...).
   Each part then uses only the laps and events it can interpret. A mechanical failure
   counts against the car's reliability and nowhere else.
5. **Work at lap level**, with race events (starts, retirements, penalties, pit stops) on
   top. Do not use telemetry.

## What the data supports

| Source | Years | Provides |
|---|---|---|
| Jolpica (used now) | results 2010+; lap times and positions 1996+; pit stops 2011+; sprints 2021+ | time and running position of every car on every lap, pit laps and durations, finishing status |
| FastF1 | 2018+ | also: tyre compound, tyre age, stint, pit in/out laps, track status (safety car, VSC, yellow), a lap-accuracy flag, sector times, speed traps, weather, race-control messages |

Checked while writing this proposal:

- **Size:** 344 races and 7,241 driver-races since 2010.
- **Compromised races are common.** In **31%** of team-races, at least one of the two
  cars did not finish.
- **Qualifying explains much, but not all.** When both teammates finish, the one who
  started ahead finishes ahead **70%** of the time (2,469 cases). About 3 in 10
  teammate battles flip during the race.
- **Retirement causes stop in 2023.** From 2023 on, Jolpica records every retirement as
  "Retired" with no cause (before that: "Engine", "Collision", "Accident", ...). For
  2023 onward, causes must be inferred from FastF1 race-control messages. For example,
  in the 2024 Australian GP:
  - "CAR 44 (HAM) STOPPED AT TURN 10" with no incident message means a mechanical
    failure.
  - "CAR 63 (RUS) STOPPED" followed by "INCIDENT INVOLVING CARS 14 (ALO) AND 63 (RUS)"
    means a crash.

  A small manual override file (2–3 retirements per race) covers the rest.
- **Signal volume, 2024, per full-season driver** (median, with range):

  | Quantity | Median | Range |
  |---|---|---|
  | Clean race laps (green flag, no pit, not lap 1) | 1,148 | 878–1,252 |
  | Clean-air laps (more than 2 s behind the car ahead) | 664 | 415–941 |
  | Battle laps (within 1 s of the car ahead) | 259 | 126–300 |
  | On-track passes made | 28 | 18–50 |
  | On-track passes suffered | 31 | 8–58 |
  | Starts | 24 | |

  For comparison, qualifying gives about 60 laps per season.
  - Race pace therefore has far more data than qualifying.
  - Racecraft rests on about 30 events per season, so it needs several seasons pooled.
  - Starts give one noisy observation per race.
- **Teammates are usually comparable.** In 88% of 2024 team-races, both drivers have at
  least 20 clean laps. In 76%, both have at least 10 clean-air laps.
- **Race pace behaves like a separate, stable quality.** A rough check compared
  teammate race-pace gaps with teammate qualifying gaps. It used 36 dry races in
  2023–24: clean-air laps, corrected for lap number, compound and tyre age.
  - **Stable within a season.** Odd and even rounds agree, with split-half reliability
    0.62 (qualifying: 0.80).
  - **Only loosely tied to qualifying.** Across 17 teammate pairs, race-pace gaps
    correlate only 0.25 with qualifying gaps.
  - **The race-minus-qualifying difference is itself stable** (reliability 0.63).

  So qualifying alone would misrank race pace. That is why race pace gets its own
  model, and why its link to qualifying (`gamma` below) is estimated rather than
  assumed.

  Caveats:
  - The sample is small and the per-race estimate crude.
  - Part of that stability could be persistent team strategy choices. The full model and
    its validation must separate the two.
- **Cost of lap data.** Jolpica returns at most 100 lap rows per request, so about 13
  requests per race. Backfilling 2010–2017 is a one-time download of about 2,000
  requests, cached like the qualifying data. FastF1 covers 2018+ faster, and adds tyres
  and track status. 2010 has no pit-stop data, so pit laps there are detected from the
  lap-time spike.

## The parts

| Part | Measures | Unit | Main data | Expected signal |
|---|---|---|---|---|
| Qualifying pace (done) | one-lap speed | s/lap | qualifying times | strong |
| Race pace | speed on clean race laps, adjusted for fuel, tyres and traffic | s/lap | race laps | strong |
| Tyre management | how fast the driver's tyres fall off, vs the car's norm | s/lap per 10 laps | race laps, stints | moderate, mainly 2018+ |
| Starts | lap-1 positions gained, vs what the grid slot predicts | positions/race | lap-1 positions | moderate, noisy |
| Racecraft | chance of completing or resisting a pass, given the pace difference | positions/race | battle laps | uncertain, must prove itself |
| Consistency | lap-to-lap spread on clean laps | s | race laps | moderate |
| Errors | driver-caused crashes, spins, collision penalties | events/race | results, race control | weak (rare events) |
| Wet weather | pace change in the wet, vs teammate | s/lap | wet sessions | weak, experimental |

Car side: qualifying pace (done), race pace, tyre wear, reliability and pit stops. Each
car part comes out of the same models as the driver parts.

### Race pace, tyre management and consistency (one model)

Fitting every lap (about 350k laps) with NUTS would be slow. The model therefore works in
two steps, which keeps it near the size of the qualifying model.

**Step A: within each race** (fast robust regression, no sampling).

- Use clean laps only. Excluded:
  - green-flag conditions not met, lap 1, the last lap
  - in and out laps
  - laps flagged inaccurate
  - laps being lapped or lapping
  - laps after damage or a failure
- Remove what every car shares in that race:
  - a race lap trend covering fuel burn and track evolution. It is the same for every
    car, so fuel loads do not need to be modelled.
  - compound offsets (2018+)
  - a dirty-air term (gap to the car ahead)
- Summarise each driver-stint by:
  - pace at a reference tyre age
  - degradation slope
  - residual spread

  Each summary gets a standard error. About 7,000 driver-races × 2–3 stints gives about
  18,000 stint summaries.

**Step B: a Bayesian state-space model on the stint summaries**, with the same structure
as qualifying:

```
stint pace     = race intercept
               + car race pace         (random walk per team, regulation resets)
               + car x circuit
               + driver race pace      (= gamma x qualifying skill + race-specific skill)
               + team fit
               + stint noise (heavy-tailed)
stint slope    = race x compound + car tyre wear + driver tyre management
log(spread)    = race + driver consistency
```

- **Link to qualifying.** Race-specific skill has its own slow random walk. Through
  `gamma`, a driver with few races borrows from qualifying. The race-specific part is
  also worth showing: who is better on Sunday than on Saturday.
- **Tyre compounds before 2018 are unknown.** Compound differences then fall into the
  zero-mean stint noise. Driver averages over many stints stay unbiased, but tyre
  management is estimated mainly from 2018+.
- **Main bias: drivers not pushing.** Examples: leading comfortably, saving fuel or
  tyres, holding station on team orders. Mitigations:
  - an "unpressured" covariate (more than 5 s of gap both ahead and behind)
  - dropping laps where teammates run within 2 s of each other
  - one-sided noise, since a lap can be slower than a driver's potential but rarely
    faster (the split Student-t from stage 1)
  - after fitting, checking residuals against the gap behind
  - a sensitivity variant that drops the final third of each race

### Starts and lap 1

```
lap-1 positions gained = grid-slot expectation (slot, clean/dirty side, start compound)
                       + car launch (team-season) + driver start skill + heavy-tailed noise
```

- Pit-lane starts are excluded.
- A driver who is hit by someone else is excluded, where race control names or penalises
  the other car. Lap 1 is chaotic, which is why the noise is heavy-tailed.

### Racecraft (overtaking and defending)

**What counts as a battle lap:** a green-flag lap that starts with the attacker within
1.0 s of the car ahead. Excluded: laps where either car pits, teammate pairs, and laps
where one car is lapping the other.

```
logit P(pass on this lap) = circuit-era overtaking difficulty (incl. DRS era)
                          + b x expected pace difference (race-pace model, incl. tyre age/compound gap)
                          + car straight-line term (speed trap 2018+, team effect before)
                          + attacker skill(i) - defender skill(j)
```

- **Attack and defence are separate skills**, with a correlation learned from the data.
  They can be told apart because each driver fights many different cars.
- **Passes made through pit stops (undercut, overcut) don't count.** They are strategy,
  and pit laps are already excluded.
- **Conversion to positions:** the skills turn into positions per race using typical
  battle exposure.
- **The circuit term is reused.** It tells the overall model how much track position is
  worth at each circuit.
- **Feasibility is checked first.** With about 30 passes per driver per season, the
  estimates will be noisy. Run synthetic recovery at these sample sizes before fitting
  real data.

### Consistency and errors

- **Consistency:** the driver effect on the residual spread in the race-pace model.
- **Errors:** driver-caused events per race. These are:
  - crash or spin retirements
  - crashes the driver continued from (damage detected)
  - penalties for causing a collision (race control, 2018+)

  Before 2018 only retirements are available.
- **Model:** a negative-binomial rate with:
  - a driver effect (slow random walk)
  - a team effect (some cars are harder to drive)
  - exposure in laps
  - wet and midfield indicators
- **Blame for collisions:**
  - It goes to the driver the stewards penalised.
  - A collision with no penalty counts half for each driver.
  - A sensitivity variant excludes unpenalised collisions entirely.
- **Precision:** events are rare, so rates settle only over several seasons. Expect
  wide intervals.

### Car reliability and pit stops

- **Reliability:** a per-lap hazard of mechanical retirement, with:
  - a team-season effect
  - a power-unit-supplier effect shared by customer teams, which helps small samples
  - an era effect
- **Pit stops:** a team operations rating, from pit-lane time relative to the race
  median. Low priority.

## Compromised races: one classification, used everywhere

Every driver-race gets a state, built once in the data layer. Principles:
- Censor, don't impute.
- Assign blame only where the stewards did.
- Keep one table so every part agrees.

| Situation | Detection | Race pace | Starts | Racecraft | Errors | Reliability | Overall (results) |
|---|---|---|---|---|---|---|---|
| Clean finish | status | all clean laps | yes | yes | none | survived | ranked |
| Mechanical failure (lap k) | status to 2022; race control with no incident, 2023+ | laps before the problem | yes | laps before | none | failure | removed from that race's ranking |
| Damage, continued | incident message, or unscheduled stop plus lasting lap-time step | laps before damage | yes | laps before | if at fault | none | ranked, flagged |
| Driver's own crash or spin | status / race control | laps before | yes | laps before | event | none | ranked behind finishers |
| Collision | status / race control plus stewards | laps before | yes | laps before | at fault: event; unclear: half | none | at fault: ranked; victim: removed |
| Penalty | race control | unaffected | none | none | if for causing a collision | none | classified result |
| Safety car / VSC / red flag | track status 2018+; lap-time bunching before | laps excluded | none | laps excluded | none | none | ranked |
| Wet race | FastF1 weather and compounds; before 2018, lap-time inflation plus a checked list | separate wet term | yes | yes | wet exposure | none | ranked |
| Team orders | teammates swap, or run within 2 s | laps excluded | none | teammate passes excluded | none | none | ranked |
| Pit-lane start | grid = 0 | yes | excluded | yes | none | none | ranked |

**Retirement causes for 2023+:**
- Rules first: a STOPPED, INCIDENT or penalty message naming the car within 2 laps.
- A reviewed override file for the rest.
- A sensitivity variant drops unclear cases.

## The overall "best driver" rating

### Rejected options

- **Hand-picked weights:** arbitrary. The ranking just follows the weights.
- **Averaging ranks, or PCA:** these weigh parts by how spread out they are, not by how
  much they win races.
- **Points or finishing positions alone:** dominated by the car and by luck. They use
  little of the information in each race.

### Recommended: let race results set the weights

1. **Fit a finishing-order model** (Plackett–Luce, the standard model for rankings) to
   every race and sprint since 2010:

   ```
   strength(driver i, race r) = car race strength (team, race; from the car ratings)
                              + sum_k  w_k x part_k(i)
                              + w_quali x overtaking difficulty(circuit) x quali skill(i)
                              + residual results skill(i)      (slow random walk, shrunk)
                              + race-day noise
   ```

   - **No double-fitting.** Parts enter as the values the models would have predicted
     before each race (leave-future-out estimates). The weights `w_k` therefore reflect
     real predictive value, and aren't fitted twice to the same data.
   - **Qualifying enters as a skill, not as grid position.** The overall rating should
     include the benefit of qualifying well. The circuit term makes qualifying count for
     more where overtaking is hard (Monaco) and less where it is easy.
   - **Compromised races** are handled by the classification table. Mechanical failures
     are removed from that race's ranking; driver-caused retirements stay in, ranked
     behind finishers.
2. **Use the residual term as a completeness check.** It catches what the parts don't
   measure, such as strategy input, tyre warm-up, adaptability and pressure.
   - Report its share of teammate result variance.
   - If it's small, the parts capture racing.
   - If it's large, something is missing, and it still counts in the overall rating.
3. **Report the overall rating** (sum of weighted parts plus residual) in three forms:
   - **Positions per race**, gained over the average current driver in the same car,
     with a 90% interval.
   - **Equal-car championship (headline):** simulate the current calendar with every
     driver in the same average car, using posterior draws of skills, weights and error
     rates, with reliability equalised. Outputs: expected points per race, P(champion),
     rank ranges.
   - **Breakdown per driver:** positions from qualifying, race pace, starts, racecraft,
     errors, and unexplained.
4. **Rate cars the same way:** an equal-driver championship built from car race pace,
   tyre wear, reliability and pit stops.

Both a "portable" version (team fit excluded) and an "in current car" version are
reported, as for qualifying.

A "custom weights" view can be offered on the dashboard as a secondary, clearly labelled
option.

## Granularity

| Level | Decision | Reason |
|---|---|---|
| Lap | core | Smallest unit where teammates face comparable conditions. Public back to 2010, so careers link across team changes. |
| Race events (starts, retirements, penalties, pit stops) | yes | Needed for starts, errors, reliability and the compromised-race table |
| Sector times, speed traps (2018+) | optional inputs | Speed traps separate car straight-line speed in racecraft; sectors help detect traffic |
| Telemetry (throttle, brake, GPS) | no | 2018+ only, dominated by car behaviour, large. Adds explanation, not ranking accuracy. Consider later for "where the time comes from" charts. |

- **Data window:** 2010 onward for all parts. The refuelling ban began in 2010, so fuel
  behaviour is comparable across the whole window. Tyre-dependent parts rely mainly on
  2018+.
- **Updates:** after every race and sprint, like qualifying.

## Validation and publication gates

The framework is the same as stage 1: 25 leave-future-out cutoffs, synthetic recovery on
the real team-change network, and sensitivity variants.

| Part | Held-out target | Must beat |
|---|---|---|
| Race pace | teammate clean-lap pace gap per race | raw teammate gaps; qualifying-only prediction; zero |
| Tyre management | teammate degradation-slope gap | no driver effect |
| Starts | lap-1 positions gained | grid-slot-only model |
| Racecraft | pass / no pass on battle laps | pace-difference-only model (no driver terms) |
| Errors | driver-caused incidents | constant rate |
| Reliability | mechanical retirements | era rate, same for all teams |
| Overall | finishing orders; teammate head-to-heads when both finish; new pairings after transfers | grid order; qualifying-only rating; plain car + driver Plackett–Luce on results; teammate Elo |

**Rule for publishing a part as its own ranking.** It needs all three:
- It beats its baseline out of sample.
- Split-half reliability (odd vs even races, Spearman–Brown corrected) is at least 0.5
  over a rolling window.
- Synthetic interval coverage is 85–95%.

A part that fails still enters the overall rating, where its learned weight shrinks
towards zero if it does not predict.

**Racecraft and errors get an extra check.** Before fitting them on real data, run
synthetic recovery at the real sample sizes. If known skills can't be recovered, don't
build the ranking.

## Build order

Each step adds its own validation section to `REPORT.md`.

1. **Data layer**, with tests on known races (e.g. 2024 Australia: Hamilton mechanical,
   Russell crash). Contents:
   - race laps (Jolpica 2010–2017, FastF1 2018+)
   - pit stops, sprints, race-control messages
   - the compromised-race classification
   - retirement-cause rules and the override file
2. **Race pace, tyre management and consistency.** Largest value.
3. **Starts, reliability, errors, pit stops.** Simple count and hazard models.
4. **Racecraft**, after the synthetic feasibility check.
5. **Overall model**, the equal-car championship, exports and report sections.
6. **Dashboard fields:** one card per part, overall ranking, contribution breakdown.

## Risks

- **Drivers not pushing** can bias race pace. Mitigations and checks are listed above.
- **Team favouritism** (strategy priority, team orders) looks like driver skill in
  results. It is partly visible in the residual term, but cannot be separated.
- **2023+ retirement causes are inferred.** The override file and a sensitivity variant
  limit the damage.
- **Racecraft and errors may not be measurable precisely enough.** The gates decide.
- **Compute.** Stint summaries keep the race model near the size of the qualifying model.
  The results model is small.
