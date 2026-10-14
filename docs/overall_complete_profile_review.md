# Review of the complete computational profile

The current profile records separate child results and completion records. The
[review collector](../analysis/overall_complete_profile_review.py) is prepared for
that layout. Its focused software checks pass in the
[prepared-check record](../outputs/championship/development/overall-prepared-profile-checks-v1/verification.json).
The actual run was [interrupted](overall_complete_profile_results.md), leaving no
whole-run summary. This collector cannot treat that missing summary as completion.
Its explicit interruption mode is described below; that mode never changes the
original case or whole-run status.

Run the focused tests in the normal capped scope first. After the original
service is authoritatively terminal, commit its complete raw output inventory
before collecting a review. Run collection in a capped scope too: fingerprinting
the retained archives reads the complete files, even though it does not load
their arrays into memory.

The collector binds the frozen registration, model inputs, original evidence,
child completion records and available child results. It preserves every required
case, including a child killed before writing its result or receipt. It checks
the saved retry settings and complete current-driver comparison inventory and
records the original diagnostics and archive identities. An inconsistent case
remains in the review with its issues while the other cases are processed.
An incomplete whole-run summary cannot be presented as a completed review.

The original registered resources and committed resource-change records are
included together. Measurements spanning the user's CPU reduction must retain
that qualification. The collector grants no new attempt or acceptance decision.

Consistent records do not establish successful inference: a correctly recorded
failed fit remains failed. Archive-array validation, diagnostic reproduction,
truth regeneration, full recovery, calibration, ranking precision and publication
remain separate requirements. The collector explicitly records that its archive
arrays and numerical diagnostics have not been independently verified.

A separate [saved-draw verifier](overall_complete_profile_diagnostics.md) is now
prepared for the complete profile. Its focused tests pass; actual archive
verification remains pending. It checks every available retained archive and reproduces the original
coordinate and driver-contrast reports, preserving failed and unavailable attempts.
Sampler extras were not archived, so their recorded counts cannot be independently
verified from the saved draws.

## Explicit interruption evidence

The optional `--interruption` argument names a committed interruption observation.
This path requires the original service to remain absent in the same observed
boot. A live service, changed boot or newly appearing terminal record requires
new inspection. The normal complete-run path still rejects a missing service or
summary; there is no general bypass of its completion checks.

Interruption review checks the exact original output and cache inventories,
every raw-output hash, committed metadata snapshots, cached archive identities,
every declared case and the absent terminal-record inventory. It verifies the
last sampling-start event against the original log and fixed retry settings.
All ordinary model/source, data, receipt, truth and attempt checks still apply.
Every original file remains unchanged.

An absent child completion is represented as null, with its presence flag false;
it receives no invented exit code or convergence decision. The original receipt
and every available attempt remain in the review. `original_run_complete` stays
false even if the available records are mutually consistent. Such a review can
support saved-report reproduction, but it does not authorize a new fit, complete
the missing attempt or change the original scientific result.
