# Overall plan: execution and acceptance ledger

This ledger tracks the full [plan](overall_model_plan.md). A completed code
module is not a completed scientific or production requirement. The goal remains
open until the full acceptance checklist has direct evidence.

| Requirement | Current evidence | Remaining work |
|---|---|---|
| Stable equal-machinery/team target | `overall_candidate.md`; driver-only state export interface; `overall_reference_calendar.json` fixes the common calendar, distances and context | Forecast-time condition policy; completion/reliability, lapping and shortened-event assumptions |
| Shared current qualifying/race model | Joint model and synthetic interface/gradient tests | Fit, recover and calibrate its shared uncertainty |
| Scientific specification | Executable likelihood/prior definitions and candidate contract | Final numerical/evaluation/prospective protocol before real-data fitting |
| Complete entrant/cutoff catalog | Missing-qualifier, team-transfer, era and future-row tests; data preflight | Recovery on the real transfer network and forecast catalog for unseen entrants/events |
| Reproducible fit engine | `fitting.py`; fixed retry, source-change, cache-corruption and terminal-failure tests; completed numerical profile preserves an exhausted confounded case | Diagnose failed inference without retuning; fresh registered evidence after source correction; full recovery and runner resource measurements |
| Known-truth recovery and simulation-based calibration | Generative responses and complete numerical-profile inventory: connected and sparse-rookie fits converged after retries; confounded fit exhausted its ladder | Complete registered SBC and stress studies, including the unresolved confounded case; no failed case may be discarded |
| Coherent weekend simulation | Student-t elimination, current total race pace, shared weekend uncertainty | Common operations/conditions calibration, lapping, shortened-race and incident assumptions |
| Scoring and ties | Explicit distance/race-control scorer, dead-heat allocation, race/qualifying countback and frozen FIA source record; targeted and full-suite checks in `development/overall-event-rule-checks/verification.json` | Generate and calibrate the required race-control and completion inputs end to end; adjudication and qualifying edge cases remain outside the simulator |
| Ability/outcome/numerical uncertainty | Nested simulation, MCSE and rank/ESS diagnostics; saved-draw diagnosis finds qualifying-contrast failures in the final confounded attempt | Resolve inference and check ranking functionals under converged posteriors with fixed simulation precision |
| Chronological development comparison | Plan defines information horizons and baseline roles; `overall_scoring_contract.json` and blind scoring utilities preserve joint draws, required events and the inherited primary comparison | Complete forecast design and baseline implementations; freeze event inventory, calibration and stopping rules; run every fold |
| Independent predictive evidence | Prior evidence boundaries are preserved | New prospective registration and predictions for this successor, followed by eligible outcomes |
| Canonical producer and explanations | Old failed gate still withholds a ranking; the versioned frontend requires successor evidence, common field and matching explanation draws/scenario | Producer/export, pairwise comparisons, and matching component explanations from accepted draws; full contract validation |
| Refresh, caches and rollback | Existing last-good publication behavior remains | Wire accepted Overall dependencies; demonstrate new, unchanged, corrected and failed refreshes |
| Software and runner readiness | Full Python suite after the event-scoring implementation (`development/overall-event-rule-checks/verification.json`); earlier weather correction, interface build/browser/contract checks and dependency audit are preserved | Complete workflow-equivalent checks after producer integration; clean environment and intended runner; final deployment source |
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
