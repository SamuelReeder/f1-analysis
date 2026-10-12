# Overall successor: implementation contract

Status: development implementation, not an accepted model or a new published
ranking. This implements the common-state and simulation foundations in the
[Overall plan](overall_model_plan.md). Its source and this contract are hashed
together by `f1rank.overall.provenance`; nested modules are included. Historical
fits, previews, gate results and their registrations remain unchanged.

## Target and current boundary

The target is **current performance under equal machinery, team support and
mechanical opportunity**, summarized by expected Grand Prix points under a common
reference scenario. Qualifying and race measurements belong to the same model.
No current-team effect contributes to the counterfactual driver score. This is
conditional performance; observational driver/car separation is still uncertain.

The code supplies a joint likelihood, bounded data design, joint-draw interface,
weekend simulation, numerical checks, provenance and a registered fitting engine.
It is not wired
into the public producer. Real-data preflight only builds inputs; it never fits,
scores drivers or sets an acceptance flag. There is no alternative dashboard
ranking. The old failed combined gate cannot authorize this candidate.

Before real-data fitting or scoring, complete and commit the recovery protocol,
frozen reference-calendar artifact, outcome calibration tests, chronological
cutoffs/baselines and new prospective protocol described below. This document
does not pretend those pending scientific decisions are already registered.

## Shared probability model

The executable specification is [model.py](../f1rank/overall/model.py). All
observations enter one NumPyro model; they share the same latent states. Related
fits are never combined by pairing independently shuffled draws.

Let `q[i]` be current portable qualifying skill at a driver-event. The existing
qualifying likelihood in `f1rank.model.model` is included directly, with
`compat_unit="era"`: persistent driver and car walks, experience/age terms,
team-by-regulation-era effects, circuit loading, transient driver/car effects,
car-segment effects, segment intercepts and Student-t observation noise. Its
existing priors and centering constraints are inherited verbatim. In the successor,
the car-event prior is expressed as a proper Normal and sampled in noncentred
coordinates. This replaces the legacy improper site plus Normal factor with the
same distribution; the old model itself is unchanged. Density-equivalence tests
protect this transformation, and a proper generator permits known-truth studies.
This is a
starting likelihood, not a claim that every existing structural choice is proven.

Total current race ability is `r[i] = centre(a*q[i] + u[i])`. The coefficient `a`
has a Normal prior centered on unity. The residual `u` has a partially pooled
driver level and a random walk with separate within-season and between-season
innovation scales. Thus race performance can depart from qualifying and change
over time. No legacy career residual or additional direct qualifying coefficient
is added again to race strength. All prior scales are specified in the executable
model before fitting; they must not be selected using a desired driver's order.

The observed race state adds a team-by-era nuisance, car package and transient
weekend effects. Team nuisance combines a learned loading on qualifying's
team effect and its own partially pooled residual. Car nuisance combines a
learned loading on qualifying's persistent car/circuit state, a sum-to-zero
team-season residual and a race-day car effect. Driver weekend form combines a
learned loading on the qualifying weekend deviation and an independent race-day
deviation. Driver and team terms are centered by event; car terms by event's car
field. Car-season residuals use the existing orthonormal contrasts.

Dry race laps measure that total observed state after controlling for event
intercept, lap trend, compound, tyre age, transient wear, traffic and serial
correlation. These controls retain the existing total-race model's priors and
Student-t innovation construction. That construction is an AR approximation: its
first-lap scale is inherited from the existing model, not an exact stationary
Student-t marginal. Permanent driver tyre/wet/racecraft/error components are not
silently enabled by this implementation.

Finishing strength is `pace[context]*total_race_state - grid[context]*log(slot)`.
The conditional order of running finishers uses a Plackett–Luce likelihood.
Grid and pace coefficients are positive and partially pooled by circuit, with
wet and neutralisation effects. This is an outcome-level approximation to
strategy, traffic and conditions; it is not a physical strategy simulator.
Conditional independence between clean laps and finish order, given the shared
states, remains an assumption requiring calibration.

Completion uses a circuit/condition geometric per-lap hazard, with finishers
right-censored. It is currently a **common all-cause hazard**. The code does not
separate causal mechanical risk, driver mistakes or correlated incidents. That
limitation must be addressed or explicitly accepted and validated under the
final target before publication. It cannot be described as a validated model of
individual driver reliability.

## Data boundaries

[design.py](../f1rank/overall/design.py) filters raw tables at the cutoff before
constructing any statistics, factors or identifiers. Qualifying uses the existing
window and robust-lap rules. Dry laps and context-dependent outcomes use the
declared FastF1-era window, avoiding the incomplete tyre archive. Missing weather
excludes that event from the outcome likelihood and is recorded, rather than
being labeled dry. Race entrants absent from qualifying receive latent states
and uncertainty. Team identity conflicts and missing driver metadata fail.

The latest race's entrants define the field. Non-starts, disqualifications and
excluded results do not enter the completion/order likelihood; those exclusions
are recorded. Retirement rows enter completion exposure; only running finishers
enter conditional finishing order. This selection is explicit and requires
separate retirement calibration. The data design's injectable cleaned-lap argument
is for synthetic fixtures; the real preflight always uses raw cutoff-first
cleaning. No old failed qualifying fold is silently declared fixed here: the
new joint model needs a separate, complete fold inventory.

## Counterfactual weekend and scoring

[simulation.py](../f1rank/overall/simulation.py) flattens one checked joint
posterior's chain/draw axes consistently across driver states and coefficients.
Missing states or mismatched axes fail. The same current states apply throughout
the reference calendar; future career development is not simulated.

Qualifying draws shared session scales, heavy-tailed segment observations and
driver weekend form, then applies elimination. Equal car, car-segment and session
intercept terms cancel in within-session comparisons; no independent car noise
is assigned to drivers. Race weekend deviations share the fitted qualifying
form component. Qualifying affects race outcomes through the generated grid.

The simulator orders cars by completed laps, then their last line-crossing
order. A retired car can remain classified and receive points. It does not yet
model lapped running cars or correlated retirements. It requires a full-distance
finisher; shortened races fail explicitly rather than receiving full points.
The final scenario must declare its distance, elimination format and condition
mix. It must not be selected by inspecting driver standings.

Scoring uses the frozen [event-rule contract](overall_event_rules.md) and
[scoring module](../f1rank/overall/scoring.py). It requires explicit scheduled,
leader and consecutive racing distances rather than assuming every classified
event earns full points. Championship ties use race-position countback, then qualifying-position
countback. Residual exact ties retain a rank interval and share title probability.
Continuous race-order draws exclude exact race dead heats; that is a simulation
assumption, not a replacement sporting rule. Sources are FIA
[General Provisions, championship and scoring articles](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_03_-_2026-06-25.pdf)
and [Sporting Regulations, race classification](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf).

Driver identities determine the attachment of random draws, independently of
input row order. An alphabetic identity is never a championship sporting
tiebreak. Replicate seasons within each posterior draw distinguish expected
performance uncertainty from season variability. Conditional Monte Carlo error
is reported separately. Expected-performance intervals are withheld when their
simulation precision check fails. Reported simulation MCSE is conditional on
the posterior sample; MCMC integration error is a separate check.

## Numerical contract

The new checks in [diagnostics.py](../f1rank/overall/diagnostics.py) use
rank-normalized R-hat, bulk and tail effective sample sizes and mean Monte Carlo
error relative to posterior standard deviation. Thresholds and minimum chain
requirements are explicit constants. Constant stochastic coordinates fail;
structurally constant deterministic coordinates are reported separately. The
legacy diagnostics and previous accepted/failed runs are not reclassified.

The existing fixed `artifacts.RETRY` ladder remains the only permitted retry
policy for the candidate fitter. Do not weaken a threshold after seeing
results. A code correction changes the source fingerprint and requires a new
run, preserving the failure and amendment. The fitter requires committed source,
data and registration, records each attempt, preserves failed draws separately,
and checks hashes before loading a posterior. Interrupted and failed receipts
cannot be used as converged caches. A numerical-profile registration cannot
authorize a real-data fit or publication.

The [numerical profile](overall_profile_protocol.json) registers synthetic
likelihood-response generation and computational checks. It conditions on fixed
observation availability, control variables, finishing-order panels and exposure
windows. It therefore does not establish end-to-end weekend calibration, selection
behavior, or successful recovery over the complete required inventory. Current
progress and missing acceptance evidence are tracked in the
[execution ledger](overall_execution.md).

## Data-boundary correction after the numerical profile

Dated 2026-10-12, after the registered profile service finished. A regression
fixture showed that aggregating entirely null rainfall readings with `any()`
treated unknown weather as dry. The correction aggregates observed readings
only; events with no known reading are excluded from dry-lap cleaning and from
the outcome context, as this contract already required. This changes neither
priors, numerical settings, gates nor the intended missing-data policy.

The [input audit](../outputs/championship/development/weather-boundary-checks/audit.json)
found no all-null rainfall event in the committed inputs. The
[profile](../outputs/championship/development/overall-computation-profile-v1/review.json)
remains a record of its original source, including its exhausted confounded
case. The correction is not an explanation or rescue for that failure.
It changes the source fingerprint, so subsequent work requires fresh registered
evidence; old receipts must not be relabeled as fits of corrected source.

## Event-scoring implementation after the numerical profile

Dated 2026-10-12, with no fits running. The deterministic scorer now handles the
frozen points columns, explicit race-control eligibility and dead heats. The
simulator calls it and requires a consecutive-racing-lap assumption in each
scenario. It still refuses unsupported shortened-event generation and does not
infer missing race-control information from the frozen reference calendar.

This completes more of the scoring implementation; it does not change priors,
fit settings, seeds or acceptance thresholds, and cannot rescue the failed
profile. The hashed source change requires fresh registered evidence and
regeneration of affected publication artifacts before promotion. Existing
receipts remain records of their original source.

## Grid-feature boundary correction

Dated 2026-10-12, with no fits running. The forecast-catalogue work reproduced a
training-feature bug: the pit-lane fallback slot used the field size after result
exclusions. A later disqualification could therefore change another entrant's
nominal starting-slot feature. The correction computes that fallback from the
complete entered field before applying the documented outcome exclusions. It
matches the field available to the forecast and does not change a prior, gate,
cutoff, seed or scoring rule.

The failing regression is preserved under
`development/overall-forecast-catalog-checks/grid-before-fix.xml`. This correction
changes the source fingerprint, so affected model/publication outputs require
regeneration. The proposed coordinate study remains pending approval and must
use its updated input fingerprint. Its original synthetic-array hashes remain
mandatory; the correction must not redraw or replace the old failed experiment.

## Shared archive verification

Dated 2026-10-12, with no fits running. The fitter's source, receipt and archive
checks are shared with a bounded compressed-posterior reader. Archive checksums
are streamed, and successful context exit rechecks the source, receipt and open
archive. The new reader validates every array, even when only selected sites are
retained as temporary memory maps. Fit settings, retry rules, diagnostics, priors
and likelihoods are unchanged.

This source refactor changes the fingerprint. Existing receipts retain their
original source and cannot be relabeled as current. Affected fit/publication
artifacts require regeneration before promotion. The coordinate-study proposal
remains pending approval with its updated source identity; its fixed original
arrays, seeds, settings and acceptance rules are preserved. Cache-format test
fixtures do not constitute converged fits or scientific evidence.

## Race-order arithmetic correction

Dated 2026-10-12, with no fits running. Synthetic regressions reproduced numerical
errors in the inherited race-order helper: finite padding could enter a
denominator at sufficiently negative strengths, and subtracting large absolute
log normalizers could erase uncertainty within tied groups. A common strength
offset must not change a Plackett–Luce probability or its score gradient.

The successor now uses relative suffix normalizers with explicit empty states.
Its predictive helper uses the same mathematical calculation in NumPy. This
corrects evaluation of the specified probability distribution; it does not add
a ranking feature, change priors, retune an acceptance rule or alter the fixed
fit settings. The historical benchmark implementation and published studies are
not rewritten. Their old results are not evidence for this corrected source.

The fitting-source fingerprint changes and affected successor fits/publication
artifacts require regeneration. The proposed coordinate study remains pending
approval under the updated fingerprint and must preserve its original arrays,
truths, seeds and gates. A future study would evaluate the arithmetic correction
and proposed coordinates together; it could not isolate either change as the
cause of convergence. The original failed profile remains failed, and this
boundary regression is not proof of that failure's cause.

## Remaining acceptance work

- Complete synthetic recovery and simulation-based calibration, including the
  real teammate-transfer network, sparse rookies, regulation changes and
  intentionally confounded driver/car states. Finite gradients are necessary
  but do not demonstrate recovery or convergence.
- Set the immutable reference calendar and its forecast-time condition policy;
  complete the lapping/shortened-event and reliability assumptions. Resolve
  whether the common all-cause hazard meets the stated target.
- Exercise the implemented checked fitter, fixed retries, caches and immutable
  run receipts on the registered inventory; measure actual fitting runtime and
  memory before choosing a refresh schedule.
- Freeze chronological development replay, comparison arms and calibration
  requirements. Register new future predictions before claiming independent
  evidence of improvement. The old candidate's registration cannot be reused.
- Integrate the accepted producer, version-aware canonical export, refresh/cache
  workflow and rollback only after those dependencies are met. Complete software
  and intended-runner checks before requesting deployment approval.

This implementation addresses structural causes of the surprising preview,
without choosing a desired Hadjar position. It does not establish any driver's
new rank and must not change the live ranking on the strength of unit tests.
