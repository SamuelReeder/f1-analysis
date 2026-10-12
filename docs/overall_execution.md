# Overall plan: execution and acceptance ledger

This ledger tracks the full [plan](overall_model_plan.md). A completed code
module is not a completed scientific or production requirement. The goal remains
open until the full acceptance checklist has direct evidence.

| Requirement | Current evidence | Remaining work |
|---|---|---|
| Stable equal-machinery/team target | `overall_candidate.md`; driver-only state export interface; `overall_reference_calendar.json` fixes the common calendar, distances and context | Forecast-time condition policy; completion/reliability, lapping and shortened-event assumptions |
| Shared current qualifying/race model | Joint model and synthetic interface/gradient tests | Fit, recover and calibrate its shared uncertainty |
| Scientific specification | Executable likelihood/prior definitions and candidate contract | Final numerical/evaluation/prospective protocol before real-data fitting |
| Complete entrant/cutoff catalog | Cutoff-first data design; outcome-free forecast catalogue; development state propagation reconstructs uncentred history and retains unseen entrants with population uncertainty | Verified as-of source/calendar inventory, accepted-posterior integration and recovery on the real transfer network |
| Reproducible fit engine | Immutable attempts/caches; shared source/receipt/archive checks for full and bounded loaders; corrected race-order arithmetic has known-probability/gradient checks; the exhausted confounded case and geometry diagnosis are preserved | Separate approval and registration before the proposed post-failure numerical study on corrected source; fresh recovery and runner resource measurements |
| Known-truth recovery and simulation-based calibration | Generative responses and complete numerical-profile inventory: connected and sparse-rookie fits converged after retries; confounded fit exhausted its ladder | Complete registered SBC and stress studies, including the unresolved confounded case; no failed case may be discarded |
| Coherent weekend simulation | Student-t elimination and shared weekend uncertainty; conditional kernels preserve teammate effects, context, order and completion within each joint draw; compensated race-order generation has controlled, exact-sum and compiled checks | Forecast-time conditions, official-grid eligibility/penalties, common operations calibration, lapping, shortened-race and incident assumptions |
| Scoring and ties | Explicit distance/race-control scorer, dead-heat allocation, race/qualifying countback and frozen FIA source record; targeted and full-suite checks in `development/overall-event-rule-checks/verification.json` | Generate and calibrate the required race-control and completion inputs end to end; adjudication and qualifying edge cases remain outside the simulator |
| Ability/outcome/numerical uncertainty | Nested simulation, MCSE and rank/ESS diagnostics; saved-draw diagnosis finds qualifying-contrast failures in the final confounded attempt | Resolve inference and check ranking functionals under converged posteriors with fixed simulation precision |
| Chronological development comparison | Plan defines information horizons and baseline roles; blind scoring utilities preserve joint draws, conditional weights and required events; outcome-free catalogue, joint states and conditional observation kernels have synthetic checks | Complete source verification, unknown-condition integration, unseen-circuit and classification-conditioning policies and baseline implementations; freeze event inventory, calibration and stopping rules; run every fold |
| Independent predictive evidence | Prior evidence boundaries are preserved | New prospective registration and predictions for this successor, followed by eligible outcomes |
| Canonical producer and explanations | Old failed gate still withholds a ranking; the versioned frontend requires successor evidence, common field and matching explanation draws/scenario | Producer/export, pairwise comparisons, and matching component explanations from accepted draws; full contract validation |
| Refresh, caches and rollback | Existing last-good publication behavior remains | Wire accepted Overall dependencies; demonstrate new, unchanged, corrected and failed refreshes |
| Software and runner readiness | Batched states and conditional observation kernels; bounded checked archive loading; capped synthetic resource and software records under `development/overall-archive-checks/` and `development/overall-observation-checks/` | Accepted-fit integration and end-to-end fitting/refresh resources; workflow-equivalent checks after producer integration; clean environment and intended runner; final deployment source |
| Deployment | No successor push or deployment | Reviewable accepted artifact, explicit push/merge/release approval, CI and live verification |

The numerical profile is registered separately in
[`overall_profile_protocol.json`](overall_profile_protocol.json). It answers
whether the inference machinery can run on the specified synthetic designs, and
records the cost and failures. Its small inventory cannot establish SBC, full
recovery, real-data calibration or permission to publish. Full recovery must
also include the dominant-teammate, regulation-change and real-transfer-network
cases provided by `recovery.py`, under a complete registration.

No model fit or real-data predictive score has been used to choose the profile
settings. Previous finite-gradient tests, software checks and data-shape checks
are development evidence only. A failed profile remains failed; it does not
authorize weakening a gate or revising a desired driver's position.

The reference calendar is generated by `analysis/overall_reference_calendar.py`
from every completed event in the committed reference-season data. Every event
has equal weight; all drivers face the same contexts. Its virtual GP lengths are
defined by the reference winners' completed distances. This is a fixed comparison
scenario, not a forecast of a future calendar or a claim about shortened-event
scoring. Missing reference inputs stop construction instead of dropping an event.
It is not yet connected to the posterior simulator; its remaining assumptions
are explicit in the artifact.

The successor frontend uses `contracts/overall.json` and the same canonical
Overall route as the legacy release. It cannot accept the legacy combined gate
as evidence for the new model. Its table and briefing distinguish uncertainty
about expected points from simulated season outcomes, and explanations identify
their posterior and common scenario. Synthetic browser fixtures are interface
checks only. The producer has not yet emitted an accepted successor artifact;
the public payload remains unchanged.

Interface verification is recorded under
[`overall-interface-checks`](../outputs/championship/development/overall-interface-checks/verification.json).
The final run installed dependencies from the lockfile, built for the repository
URL and passed the browser suite on committed source. Earlier test-harness
failures are retained alongside the corrected run. The dependency advisories and
post-update audit are also preserved. These checks do not validate the model,
replace an end-to-end publishing run, or establish intended-runner readiness.

The [scoring contract](overall_scoring.md) defines arithmetic and complete-event
comparison safeguards, without authorizing fitting or setting a calibration gate.
Its synthetic checks are separate from the registered computational profile.

A data-boundary regression has reproduced an unknown-weather bug: an event whose
rainfall rows are all null is treated as dry by the current aggregation. The
intended exclusion policy is already documented in the candidate contract.
`development/weather-boundary-checks/` preserves the failing checks and an audit
of the committed weather inputs. After the profile service terminated, the
aggregation was corrected to use only observed rainfall readings. The dated
candidate-contract note preserves the evidence boundary. The correction requires
regeneration; the old profile cannot be relabeled as current evidence.

The completed [profile review](../outputs/championship/development/overall-computation-profile-v1/review.json)
records every attempt and service resources. Its confounded case failed after
the complete fixed ladder. This was a numerical profile on synthetic designs,
not a real-driver ranking, complete recovery study or calibration success.
The failure and tree-depth saturation remain part of the record. No sampler
setting or acceptance threshold has been changed in response to them.
The [profile report](overall_profile_results.md) links its recorded diagnostics
and post hoc saved-draw analysis. The latter does not provide a substitute
acceptance gate or permit selecting a different retry.

The [event-scoring verification](../outputs/championship/development/overall-event-rule-checks/verification.json)
records the targeted known-outcome fixtures and full local Python suite against
their committed source. The nested simulator now uses explicit race-control
inputs, and supplied shortened outcomes receive distance-appropriate points.
The reference calendar and failed numerical profile remain unchanged. A scorer
does not provide a model for when an event stops, which cars are lapped, or how
incidents correlate. Those requirements and the final publication-source
regeneration remain open.

The [inference proposal](overall_inference_revision.md) responds to the saved-draw
geometry evidence with a distribution-preserving coordinate transformation for
the lap controls. Its mathematical/software checks pass. The original model,
registered sampler settings, gates and failed profile remain unchanged. The
matched experiment is prepared under a separate identity and explicitly pending
user approval; the runner refuses that pending registration. No transformed
MCMC fit has been run and no convergence improvement is claimed.

The [forecast catalogue](overall_forecast_catalog.md) maps the full declared field
to training history while rejecting target-weekend observations. It preserves
new-driver, new-team and new-circuit uncertainty requirements without filling
missing states with average ability. Its source-vintage claims remain unverified;
it does not propagate states or produce predictions. The separate
[state implementation](overall_forecast_states.md) now reconstructs uncentred
historical states before centring the new field. It retains the joint draw axes,
propagates the existing driver/car walks, and preserves population uncertainty
for unseen states. It has no outcome predictor or acceptance role.

This work reproduced a grid-feature bug: the pit-lane fallback depended on field
size after excluding result statuses. The correction uses the complete entered
field before those exclusions. The failing regression is preserved alongside
the input audit in `development/overall-forecast-catalog-checks/`. The audit
reproduces the original synthetic observations and truths without fitting.
The pending numerical proposal's source fingerprint is updated; its approval
status, synthetic arrays, priors, settings and acceptance rules remain unchanged.
Affected outputs require regeneration before promotion.

The [forecast and grid verification](../outputs/championship/development/overall-forecast-catalog-checks/verification.json)
records the failing regression, passing focused checks, input audit and full local
Python suite. It distinguishes the audit's base revision from its exact corrected
source hashes, verified against the committed fix. These checks establish the
input correction and software behavior; they measure no predictive improvement
and do not complete the acceptance plan.

The [forecast-state verification](../outputs/championship/development/overall-forecast-state-checks/verification.json)
records agreement with the joint model under matched innovations, checks of new
states and retained dependence, and the full local suite on committed source.
The initial distribution-property assertion failure is preserved. It was a test
fixture issue, not a changed prior or acceptance criterion. Fitting sources and
the pending numerical registration are unchanged. This implementation still
requires accepted-posterior integration, full-archive resource work and the
complete predictive experiment; it has produced no ranking.

The [batching and resource verification](../outputs/championship/development/overall-forecast-batch-checks/verification.json)
adds bounded parameter checks, terminal-state reconstruction and complete draw
batches without changing the innovations or model. The resource workload uses
synthetic values in temporary memory maps at the current design dimensions and
the largest draw shape from the unchanged retry ladder. It verifies actual local
scope limits, records complete finite output coverage and retains no driver table.
Compressed accepted-posterior loading and intended-runner resources remain open;
neither this workload nor its passing software checks authorize a ranking.

The [archive verification](../outputs/championship/development/overall-archive-checks/verification.json)
adds shared immutable fit checks and bounded decompression. Every archived array
is checked even when a forecast retains only selected sites. The compressed
synthetic format fixture matches the earlier uncompressed parameter identity and
forecast-state checksum, under the enforced local resource caps. Its convergence
Boolean is explicitly a fixture field, not a successful numerical fit.

The shared fitter refactor changes the model source fingerprint and the pending
proposal is updated accordingly. Protected sampling functions/settings and the
original probability-model/generator sources are verified unchanged. Prior
failures remain preserved; affected outputs still require regeneration and the
proposed numerical study still requires the outstanding approval. Accepted-model
integration, scientific evidence and intended-runner readiness remain open.

The [conditional observation implementation](overall_observation_kernels.md)
connects the state forecaster to the existing qualifying and race likelihood
pieces while preserving joint draws. It keeps official-grid, classification,
unknown-condition and unseen-circuit policies explicitly unresolved. Known-truth
fixture checks compare observation densities and coefficients with model traces,
normalize small-panel order and censoring probabilities, and retain the draw set
across field permutations and batches. This is development implementation, not
an accepted prediction experiment or a substitute for the failed numerical case.
The [verification record](../outputs/championship/development/overall-observation-checks/verification.json)
preserves the initial fixture failures, corrected focused run, full local suite
and compressed synthetic resource measurement. The fitting-input identity and
pending numerical proposal are unchanged. No model was fitted or ranking
published by this work.

The [race-order verification](../outputs/championship/development/overall-order-numerics-checks/verification.json)
preserves the failing offset/tied-group regressions, corrected probability and
gradient checks, full suite and current compressed synthetic resource check.
The original profile observations and truths are reproduced exactly by the
separate input audit. This correction changes the fitting-source fingerprint,
and the pending proposal identifies it without granting approval. The
historical failure remains recorded; no causal explanation for it is established.

Conditional predictive scoring now includes the posterior weights implied by a
discrete conditioning event. Synthetic checks preserve uncertainty even when that
event has a very small common probability, and reject impossible conditioning
or mismatched draws. The final choice of conditioning information and exposure
still belongs in the complete predictive registration. The numerical regression
evidence concerns evaluated probabilities and gradients; sampled-order behavior
under extreme offsets was subsequently checked separately as recorded below.

The [generation correction](overall_order_generation.md) reproduces and fixes
rounded-away race-order perturbations in the joint generator, conditional
forecast and counterfactual race simulation. It uses the same random draws and
retains their roundoff component during sorting. The
[verification record](../outputs/championship/development/overall-order-generation-checks/verification.json)
preserves the failing regressions, corrected compiled/known-sum checks, full
suite, unchanged original synthetic inputs and capped resource measurement.
Invalid generated indices are rejected before they can select an observation.
The fitting-source identity changes and affected evidence requires regeneration;
the numerical study remains pending approval and the original failure remains.
These checks close the identified order-generation arithmetic gap, not the
remaining recovery, condition, exposure, classification or calibration work.
