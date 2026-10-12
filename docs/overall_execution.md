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
| Reproducible fit engine | `fitting.py`; immutable attempts/caches; the exhausted confounded case is preserved; saved-draw geometry diagnosis and equivalent-coordinate prototype | Separate approval and registration before the proposed post-failure numerical study; fresh recovery and runner resource measurements |
| Known-truth recovery and simulation-based calibration | Generative responses and complete numerical-profile inventory: connected and sparse-rookie fits converged after retries; confounded fit exhausted its ladder | Complete registered SBC and stress studies, including the unresolved confounded case; no failed case may be discarded |
| Coherent weekend simulation | Student-t elimination, current total race pace, shared weekend uncertainty | Common operations/conditions calibration, lapping, shortened-race and incident assumptions |
| Scoring and ties | Explicit distance/race-control scorer, dead-heat allocation, race/qualifying countback and frozen FIA source record; targeted and full-suite checks in `development/overall-event-rule-checks/verification.json` | Generate and calibrate the required race-control and completion inputs end to end; adjudication and qualifying edge cases remain outside the simulator |
| Ability/outcome/numerical uncertainty | Nested simulation, MCSE and rank/ESS diagnostics; saved-draw diagnosis finds qualifying-contrast failures in the final confounded attempt | Resolve inference and check ranking functionals under converged posteriors with fixed simulation precision |
| Chronological development comparison | Plan defines information horizons and baseline roles; blind scoring utilities preserve joint draws and required events; outcome-free catalogue and joint state propagation have synthetic checks | Complete source verification, outcome/condition generation and baseline implementations; freeze event inventory, calibration and stopping rules; run every fold |
| Independent predictive evidence | Prior evidence boundaries are preserved | New prospective registration and predictions for this successor, followed by eligible outcomes |
| Canonical producer and explanations | Old failed gate still withholds a ranking; the versioned frontend requires successor evidence, common field and matching explanation draws/scenario | Producer/export, pairwise comparisons, and matching component explanations from accepted draws; full contract validation |
| Refresh, caches and rollback | Existing last-good publication behavior remains | Wire accepted Overall dependencies; demonstrate new, unchanged, corrected and failed refreshes |
| Software and runner readiness | Full local Python suite after joint forecast-state propagation (`development/overall-forecast-state-checks/verification.json`); earlier grid, coordinate-wrapper, event-scoring, weather, interface and dependency checks are preserved | Full-archive forecasting resources/batching; workflow-equivalent checks after producer integration; clean environment and intended runner; final deployment source |
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
