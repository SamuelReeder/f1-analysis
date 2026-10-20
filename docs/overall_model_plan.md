# Plan for the canonical Overall driver ranking

Prepared 2026-10-08. This is an implementation roadmap, not a fitted model,
completed statistical registration, or permission to publish a failed result.

Implementation progress is recorded separately in the
[successor contract](overall_candidate.md). The shared likelihood, data interfaces
and simulation now have code and synthetic checks. The scientific acceptance and
production checklist below remains open; implementation is not evidence of a
successful real-data fit or predictive improvement.

The product goal is a single canonical Overall ranking. Improve and eventually
replace the existing championship pipeline in place. Qualifying, race pace and
other measurements explain that ranking; they are not alternative answers to
who is best. Existing experiments remain immutable research records and must
not become additional public ranking products.

## Define the result before building it

The recommended planning assumption is **current driver performance under equal
machinery, team support and mechanical risk**, summarized as expected points per
Grand Prix over the same fixed reference mix of circuits and conditions for every
driver. This is conditional performance under a stated scenario, not proof of
innate or universally transferable ability. The user was asked whether to retain
current-team effects instead; absent a different preference, use this definition.
That definition must be settled in the implementation's committed contract before
fitting. Changing it later requires a new model version, not a display toggle that
silently changes the meaning of Overall.

The contract will specify:

- **Field and date:** race entrants at the latest completed event, with an explicit
  data cutoff. Missing qualifying or dry-race observations must not silently drop
  entrants or assign them average ability with no uncertainty.
- **Common scenario:** a reference calendar and condition distribution fixed from
  information available at registration. Store its event identities, weights,
  distance, tyre and weather assumptions. Every driver faces the same mixture.
  Freeze it within a model version so a ranking change is not secretly a change
  of evaluation schedule. Do not infer circuit weights from favorable scores.
- **Current ability:** estimate persistent skill and current form separately from
  transient weekend effects. Evaluate ability as of the cutoff; integrate future
  weekend variation without presenting the result as a career or future-development
  ranking. A simulated season is a way of communicating outcome variability.
- **Equal opportunities:** fit car and team effects on real observations, then
  neutralize them only for the counterfactual score. Use common operational and
  mechanical conditions. Retain uncertainty about the separation instead of
  interpreting a nuisance estimate as known transferable skill.
- **Score:** sort by posterior expected points, using the scoring rule recorded in
  the protocol. Never average positions in separate qualifying and race tables,
  or hand-select component weights to obtain a preferred order.
- **Uncertainty:** separately report uncertainty in expected performance,
  simulated-season variability, and numerical simulation error. Near ties should
  look uncertain even if the table needs a deterministic display order.

## Weaknesses and required responses

The [completed audit](championship_model_audit.md) supplies the diagnostic evidence.
Its Hadjar probes explain a model behavior; moving that driver up or down is not an
acceptance criterion. The following problems apply to the whole field.

| Weakness in the current system | Evidence | Required response |
|---|---|---|
| Overall mixes current qualifying with a career race residual | `championship_preview.py` takes only `racemulti.u`; audit component and probe outputs | Represent current qualifying and current total race pace coherently, with time-varying driver states |
| Driver, car and team effects are difficult to separate | Audit posterior correlations and team-effect probes; incomplete race recovery; failed car prediction gate | Preserve nuisance effects, test recovery and team transfers, and carry separation uncertainty into the common scenario |
| Related estimates are combined as if their draws were independent | Qualifying medians enter race fitting; mean qualities enter finishing fitting; simulation samples stages separately | Infer shared effects jointly or use a fully specified conditional posterior procedure; forbid arbitrary pairing of independent marginal draws |
| Simulated qualifying differs from the fitted observation model | Gaussian simulator versus fitted heavy tails, segment scales and shared effects | Use the fitted generative observation process for qualifying and verify grid calibration |
| Race conditions and operations are compressed into a generic grid-to-finish coefficient | `championship.race_model` and `simulate` | Model current race strength, circuit/condition context and classification consistently; integrate common operational variation |
| Driver mistakes, tyre management and racecraft are not established as independent ranking components | Existing reliability, consistency, degradation, first-lap and battle summaries | Keep a fixed inclusion policy; evaluate additions conditionally inside Overall and preserve failed inclusion decisions |
| Championship ties are broken by sorting rather than a specified rule | Audit records points-tied seasons; preview rank calculation | Retain race results, apply a documented tiebreak, and keep unresolved exact ties exchangeable |
| R-hat/divergence checks alone do not measure all numerical uncertainty | `artifacts.diagnostics`; exhausted archive fits; incomplete recovery experiment | Add effective sample size, rank diagnostics and precision reporting to the new protocol without retrospectively relabeling old fits |
| A new race does not currently regenerate Overall | `dashboard.refresh_steps` calls qualifying and total race exports but no championship stage | Add Overall and all its dependencies to refresh, cache, commit and publication workflows |
| Export gating is tied to the old experiment's schema | `dashboard.overall_result` reads the old combined gate and hardcodes `in_team` | Introduce an explicit versioned Overall contract; a new model needs its own evidence, never a copied success flag |

### Evidence boundaries that must survive the redesign

The combined championship gate failed. The archive studies failed computationally
before scoring. The independent current-season race table passed a **teammate-gap**
prediction test; its car table did not pass, and its synthetic recovery experiment
is incomplete. None of those facts establishes cross-team equal-car identification.

The old prospective protocol evaluates the already frozen race-residual candidate.
It cannot validate this redesigned model or donate its registration date to it.
Historical data and previously inspected outcomes are development evidence.
No additional random split, renamed score file, seed search or rerun makes them
independent evidence again.

## Architecture of the final pipeline

Use a common event/driver/team catalog and a shared posterior state for the
canonical model. Internal modules are allowed; competing public rankings are not.

### Current driver states and observation models

Each driver has linked qualifying and race-performance states. Both contain
persistent ability and current form. Their population relationship is learned
with partial pooling, preserving differences between drivers who qualify well
and drivers who race well. New or lightly observed drivers borrow population
information and retain wider uncertainty.

Fit qualifying observations with the established session, car, circuit, age,
experience, transient form and robust-noise structure as the starting point.
Fit race-lap observations with the established compound, tyre-age, lap-number,
traffic and serial-noise controls as the starting point. Race pace is **total
current pace**, so do not also add the old qualifying-derived career residual as
another supposedly independent source of speed.

Car and driver-team terms remain nuisance states in the observational likelihood.
Their qualifying/race relationship, regulation changes and team-transfer handling
must be explicit. The starting specification uses team and regulation context
already represented in the data; it does not label every unexplained team effect
as car-handling compatibility. Estimate the relationship rather than force race
and qualifying pace or car performance to be identical.

The implementation's scientific specification must commit the state-transition
equations, centering constraints, priors, covariances and data windows before
real-data fitting. The existing qualifying and total-race implementations supply
tested observation-model pieces, not interchangeable finished ranking numbers.
The old lap archive has incomplete tyre information and failed fits; it must not
be added merely to increase sample size. Declare its likelihood and role, or its
exclusion, before using it in the new experiment.

### One coherent weekend-to-points calculation

Qualifying performance generates a grid through the registered qualifying
observation process. Race performance and that grid generate finishing outcomes,
with condition/circuit effects and common operational variation. Classification
and completion distance govern retirement outcomes and scoring. Driver incident
effects enter only under their committed inclusion rule; a failed component test
cannot be treated as proof that drivers have identical underlying error rates.

The finishing-stage coefficients are estimated from outcomes with uncertainty
in the latent pace states retained. Avoid fitting to smoothed full-data features
and then presenting evaluations made with a different information horizon as if
they tested the same procedure. Qualifying need not receive an arbitrary separate
weight in the score: its direct contribution is the starting position it earns.
Any additional direct qualifying term must have a stated role and a prespecified
conditional test rather than duplicate total race pace by default.

A joint model is the target because the inputs share latent effects and data.
Start by proving the data and likelihood interfaces on synthetic cases. If a
monolithic fit is computationally infeasible, a conditional modular implementation
must define its probability factorization, preserved dependence and approximation
error before real scoring. Independently sampling existing model exports is not
an acceptable substitute for that work.

### Included context versus optional driver qualities

Mandatory context includes circuits, race distance, tyres, traffic and race
conditions; common strategy and safety-car variability must be represented and
calibrated at the outcome level. This does not require a full physical lap-by-lap
simulator. The approximation has to match the claim and pass predictive checks.

Driver-specific tyre management, starts, overtaking/defending, consistency and
incident effects are conditional extensions inside this same model. Wet-driver
effects need their own evidence. When data cannot identify a driver-specific
effect, propagate a declared population uncertainty and disclose that limitation;
do not fabricate a confident zero or silently shrink the target to qualifying.
Sector times remain an untested candidate, not a prerequisite or promised fix.

First make the common qualifying/race/points pipeline coherent. Then use the
fixed conditional comparison policy for justified additions. Record unsuccessful
extensions; do not keep adding features until an inspected historical gate passes.

## Ordered implementation work and acceptance evidence

Each row is a dependency for the next. Work is complete only when its evidence
exists, not when the code or a plan for it exists.

| Work package | Concrete deliverables | Evidence required to proceed |
|---|---|---|
| Contract and protocol | A versioned Overall definition; frozen field/scenario policy; complete likelihood, priors and transitions; computational and evaluation specifications | No unresolved scientific settings; sources and registration committed before fitting or scoring |
| Data and probability interfaces | Shared identifier/cutoff catalog; draw-level state interface; qualifying, lap and classification likelihoods | Hand-computable synthetic checks; missing-session, substitute-driver, team-switch and era-boundary cases; no future observations in training |
| Identification and computation | Identifiable parameterization; shared posterior inference; complete synthetic recovery and resource profile | Every declared synthetic case accounted for; recovered uncertainty and ranking functionals; no unresolved failed fit or silent fallback |
| Weekend and points simulation | Fitted qualifying generator; current race-strength mapping; common scenario; completion and scoring; explicit ties | Driver-permutation invariance, equal-driver symmetry, monotonicity with common randomness, scoring/classification fixtures, and separate ability/outcome/Monte Carlo uncertainty |
| Predictive evaluation | Frozen chronological replay, baselines, calibration and component comparisons; subsequent independent evidence under a new registration | All required folds accounted for; primary comparison and calibration rules met; limitations of cross-team identification reported |
| Canonical publication | Replace the Overall producer; version-aware exporter; one table with component explanations and evidence | Schema/manifest checks, rejection of stale or failed outputs, matching field and draws across table and explanations; no preview fallback |
| Refresh and rollout | Overall dependencies in refresh and cache; portable artifact; clean-environment run; failure/rollback handling | New-event, unchanged-event, corrected-data and failure cases demonstrated; measured runtime/memory; local tests, build and browser checks; approved deployment and live verification |

Suggested code boundaries are `championship_design`, `championship_model`,
`championship_validation` and `championship_simulation`, with the existing
`championship.py` orchestrating the canonical output. These are planned modules,
not new ranking products. Keep output ownership under `outputs/championship/`.
Immutable run directories may hold development and historical evidence, but only
the accepted canonical run may advance the public pointer.

## Validation and stopping policy

### Freeze the question and the information available at prediction time

For the race-stage comparison, evaluate finishing-order predictive density on
the same eligible races as the baselines, conditional on the actual starting grid
for every arm. Driver and car states must be estimated strictly before the test
weekend; no test-race laps, finish position or later qualifying may enter training.
Define how classification conditions the score and evaluate retirements separately
so selecting finishers cannot hide poor incident predictions.

Also require an end-to-end pre-qualifying weekend check that integrates the
simulated grid. It uses only information available before qualifying for all arms.
Passing a race-stage test with the real grid does not validate the generated grid
or the equal-car weekend simulation. Freeze the forecast origin and unknown-weather
treatment separately for these evaluations.

Use the existing grid-plus-qualifying race model as the primary legacy comparison.
Report grid-only and current-race-only ablations as prespecified secondary
comparisons. Fit every arm on the same eligible history and preserve uncertainty;
secondary successes cannot rescue a failed primary test. Do not select a baseline
after learning which is easiest to beat.

### Statistical and computational requirements

- Retain the existing principle that the paired improvement interval must exclude
  no improvement. The current season-block implementation requires a positive
  lower **95%** bound and at least **3** season blocks. Those are inherited
  references from `championship.paired_seasons`, not a new successful result.
- For pace prediction, the existing project calibration criterion is **85–95%**
  coverage of **90%** intervals, with narrower intervals than its baseline.
  Retain it where the target is comparable. Define scoring and calibration for
  grids, retirements and final orders in the new protocol before inspecting their
  results; do not assume the pace gate automatically validates these outputs.
- Use complete simulation-based calibration and recovery checks on known truths,
  including the real teammate-switch network, sparse rookies, dominant teammates,
  regulation changes and intentionally confounded cases. A correct posterior
  can remain broad; a plausible-looking order is not evidence of identification.
  All declared truths must be reported, including failures.
- Preserve the current fixed retry ladder and failure recording. Existing fits
  require **R-hat < 1.05** and at most **1 divergence per 1,000 draws**. For the new
  model, register additional rank-normalized diagnostics, bulk/tail effective
  sample size and Monte Carlo precision criteria before production fitting. These
  are additional checks, not retrospective changes to old gates. The numerical
  appendix is a required deliverable of the contract work package.
- Prespecify all cutoffs, exclusions, failure handling, block resampling,
  component comparisons, required folds, stopping dates and numerical precision.
  Do not drop a difficult season or extend a trial because a result disappoints.

Simulation-based calibration checks inference under a specified generative model;
it does not prove that model describes real racing. Rank-normalized diagnostics,
effective sample size and Monte Carlo error address computational weaknesses that
ordinary R-hat can miss. These additions follow the primary methodological work
on [simulation-based calibration](https://arxiv.org/abs/1804.06788) and
[rank-normalized convergence diagnostics](https://arxiv.org/abs/1903.08008).

### Separate model development from evidence of reliable improvement

Chronological historical replay is required for development and failure diagnosis.
It does not erase our prior exposure to these outcomes. Freeze the final candidate
before collecting its independent future predictions, with an untouched evidence
inventory, fixed horizon and stopping rule. A newly proposed archive requires a
prior-use audit; the previously failed archive cannot simply be relabeled unseen.

The old prospective protocol remains unchanged and cannot be reused to claim this
new model was preregistered. The new protocol must define its own eligible outcomes
after the new candidate is frozen. Do not promise an immediate reliable-improvement
claim or a calendar completion date before that evidence and computational profile
exist. Development can proceed now; obtaining independent evidence is a separate
condition for acceptance, not a reason to create another public provisional table.

If the final comparison fails or uncertainty remains uncalibrated, preserve the
failure and withhold the canonical ranking with a clear reason. A failed component
cannot be rescued by a preferred driver's position, numerical convergence alone,
or a positive secondary comparison. Revisions are new registered model versions
of the same product; they do not overwrite previous results.

## Publication, refresh and operational completion

The user's [refresh-runtime priority](overall_refresh_runtime.md) adds a practical
constraint to the work order: after the current registered computational profile,
measure a resumable, warm-initialized update and evaluate sequential inference if
needed before expanding expensive calibration workloads. Full model-development
studies are versioned evidence, not a required recurring race-refresh workload.
The user selected **by the next morning** in America/Toronto for the complete
race-to-publication turnaround. Establish a precise local deadline before its
timed acceptance benchmark, cover the race calendar in the intake schedule, and
report delayed source data separately. The current code has no measured successor
update budget. This priority does not change the active fit, its gates or the
scientific acceptance checklist.

The canonical payload needs a model-version identifier, scenario identity, data
cutoff, convergence/recovery/calibration evidence, acceptance state, field, point
estimates, pairwise comparisons and separately labeled uncertainty. Component
explanations must use the same posterior and scenario as the table. Changes from
neutralizing correlated components are explanatory probes, not an additive
decomposition of uniquely identified talent.

Update `dashboard.overall_result`, frontend types and the Overall view together.
The old `in_team` selection and combined-gate schema cannot silently stand in for
the revised target. Remove the choice between competing Overall methodologies
from the user flow; keep evidence and diagnostic details available underneath the
single result. Preserve archived previews and published snapshots for audit, with
their original labels and dates.

Extend `dashboard.refresh_steps`, `cachestore` and the refresh workflow's committed
paths to include the entire accepted Overall pipeline and its cached dependencies.
Source or data corrections must invalidate affected fits. Publish an atomic
snapshot only after all required stages pass; a failed refresh preserves the last
accepted snapshot and marks its age and the failure. Until an accepted Overall
snapshot exists, display its unavailable status. Test interrupted runs, incomplete
files and stale manifests as well as the happy path.

Before modifying any hashed model or extraction source, announce the required
regeneration. Preserve frozen experiments at their recorded revisions and rebuild
every affected racing artifact before promoting the new publication. Do not edit
hashed sources while their fits run.

Use the current resource policy: fitting lanes share the **400%** CPU quota,
weight **10**, nice **19**, with aggregate memory caps no greater than **13G**.
The user requested the lower shared cap during the active numerical profile;
the [resource-change record](../outputs/championship/development/overall-complete-design-profile-v1-resource-changes/20261013T170054Z-cpu-reduction.json)
preserves the earlier cap and the live change. Keep that lower limit for subsequent
work unless the user changes it again. Historical registrations and runtime
records retain their original limits; do not silently reset the shared cap to
match an older registration.
Other heavy checks run separately at **3G** and **200%** CPU. The remaining policy sources are
[`racing_regeneration/run.json`](../outputs/analysis/racing_regeneration/run.json)
and [`audit/verification.json`](../outputs/analysis/championship_model_audit/verification.json).
Measure runtime, resident memory, cache size and restart behavior on the intended
runner before calling the refresh production-ready; current full-fit CI resource
use and the earlier OOM cause remain unverified.

Run the full Python suite, dashboard build and browser suite for the repository
URL, then the workflow-equivalent clean-environment check. These establish software
behavior, not scientific acceptance. Pushing, merging the PR, and creating or
uploading the cache release still require explicit user approval. Previous push
and deployment approvals do not authorize these future actions automatically.

## Completion checklist

- [ ] The scientific contract defines a single stable Overall estimate.
- [ ] Current qualifying and race performance share a coherent uncertainty model.
- [ ] Driver/car/team separation and every declared synthetic case are evaluated.
- [ ] The weekend simulation, classification and tiebreak rules are verified.
- [ ] The required prediction and calibration tests pass under their registered rules.
- [ ] Independent evidence supports any claim of reliable predictive improvement.
- [ ] Only the accepted canonical ranking feeds the dashboard; explanations match it.
- [ ] New data, corrections, cache restore, failed runs and rollback work end to end.
- [ ] Runtime and memory are measured on the intended runner, and software checks pass.
- [ ] Approved deployment serves the verified artifact and its current evidence.

This roadmap is complete when its decisions, sequence and acceptance evidence are
reviewable. The final ranking is complete only when the implementation checklist
above is actually satisfied. No item above is claimed complete by writing this plan.
