# Track-context correction and regeneration

The [preserved regressions](../outputs/championship/development/overall-track-context-checks/before-fix.xml)
reproduce missing track-status observations being treated as green and VSC
periods ending at the ending notification. The latter contradicts FastF1's
[documented status semantics](https://raw.githubusercontent.com/theOehrly/Fast-F1/v3.8.3/fastf1/_api.py):
the ending notification precedes the actual clear signal. The correction retains
the VSC interval until clear or a replacing status, preserving the source code
when a partial feed begins with the ending notification.

The successor design now requires observed raw track-status evidence after its
training cutoff is applied. SC, red-flag or VSC evidence establishes that an
event had neutralisation. Without positive evidence, absent rows, null values or
unrecognised statuses cannot establish a nonneutral event. Such unresolved
events are excluded from context-dependent outcomes and clean-lap evidence and
remain visible in the context exclusions. Recognised yellow flags alone are
not whole-race neutralisation. This rule checks observed evidence; it cannot
prove the external archive's completeness.

The [normal timeline rebuild](../outputs/championship/development/overall-track-context-checks/regeneration.json)
changed VSC intervals at 72 events. Event-level neutralisation labels remained
unchanged in the current eligible archive, whose track-status observations
were present. The unchanged labels do not imply unchanged pace-model inputs:
the [lap comparison](../outputs/championship/development/overall-track-context-checks/lap-impact.json)
records every added/removed clean-lap identity and changes to transformed pace,
using the original and corrected timelines with the same raw tables and cutoff.
The generated timeline and its normal retirement-cause audit are committed.
No settings were chosen from those audit results.

The [synthetic-input audit](../outputs/championship/development/overall-track-context-checks/input-audit.json)
reproduces the original registered observations and truths exactly. Synthetic
fixtures now declare their intended green observations instead of relying on
absence of timeline rows. The original probability model, sampling procedure,
seeds, retry ladder and diagnostic gates remain unchanged. The failed confounded
case is preserved; the pending coordinate proposal identifies the corrected
source and data but remains unapproved.

The [verification record](../outputs/championship/development/overall-track-context-checks/verification.json)
separates reproduced failures, the mistyped initial test-path invocation,
corrected focused checks, the full suite and capped regeneration/impact audits.
These establish software and data-boundary behavior, not improved predictive
accuracy or a new driver ranking. All affected racing artifacts still require
fresh evidence before promotion. Forecast-time condition integration and the
remaining scientific and production requirements remain open.

The initial full-suite record had 611 passing checks and one publication failure:
the race-pace export named the previous timeline hash. The normal GPU producer
has now completed regeneration with unchanged model, folds, ablations, sampling
settings, seeds, retry procedure and gates. Its fresh manifests pass verification,
and the validation metrics reproduce from the generated prediction files.
The [run review](../outputs/championship/development/overall-track-context-checks/race-pace-review.json)
and [fit log](../outputs/championship/development/overall-track-context-checks/race-pace-fit.txt)
preserve every attempt. The driver teammate-gap gate passes and the car gate
fails; the car table remains withheld. The existing total-race-pace procedure is
separate from the successor's unresolved numerical profile and cannot establish
that model's acceptance.
The [post-regeneration full Python suite](../outputs/championship/development/overall-track-context-checks/race-pace-verification.json)
passes, including the previously failing checked current dashboard data contract.
Refreshing the portable publication and its final production checks remains
separate work.

The review retains the per-attempt durations and the interval inferred from the
launch timestamp and completion-file modification time. Their totals disagree,
so an exact end-to-end runtime is unverified. The transient service had already
been collected and no matching journal records were available; peak memory and
CPU accounting could not be recovered. Its enforced launch limits and CPU slice
settings remain in the launch receipt. Intended-runner resources remain untested.
