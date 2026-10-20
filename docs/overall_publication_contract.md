# Canonical Overall export statistics

The successor export now has a checked boundary between the paired simulation
summary and the dashboard. This implements part of the
[canonical production plan](overall_model_plan.md); it does not accept a real
fit, authenticate scientific evidence, or publish a ranking.

## Preserved statistics and identities

[`prepare_components`](../f1rank/overall/publication.py) accepts a complete paired
summary whose existing conditional simulation checks pass. It carries the full
current field, grouped posterior count, replicate count and common race inventory.
Names and scenario labels are attached without changing the estimates.

Every explanation and driver comparison retains the same fit, posterior archive,
scenario and simulation identities. All current entrants need explanations, and
every unordered pair needs a comparison. Missing, duplicated, self or unknown
driver comparisons fail. Pairwise means must agree with the point means in the
standings, within arithmetic roundoff.

Intervals and numerical errors are copied from the joint summary. Marginal driver
intervals are never subtracted to construct a comparison interval. Reversing a
comparison negates its mean and swaps and negates its interval endpoints. The
adapter omits finite-simulation ordering frequencies: those are not verified
posterior rank probabilities.

Explanations retain their expected-point loss, interval, posterior spread and
conditional simulation errors. The required component inventory is unchanged.
The frontend displays the paired intervals and keeps the warning that correlated
component changes are not an additive allocation of credit.

The existing conditional precision limits are unchanged. Both the Python and
browser checks validate each supplied precision result and reconstruct the
existing worst-case bound from the replicate inventory. An empty interval,
insufficient precision, inconsistent bound, mismatched identity or missing pair
withholds the claimed established table.

## Acceptance remains separate

The adapter returns export components with scientific verification and publication
explicitly false. It supplies no public status/schema and writes no file or
pointer. `check_payload` verifies internal consistency and the presence of the
required recorded decisions; it does not authenticate those decisions. Passing
it cannot substitute for the acceptance controller's registration, source,
posterior, recovery, predictive, calibration, functional precision and prospective
evidence checks.

The unreleased successor payload now requires complete paired comparisons and
structured explanation uncertainty. Its browser safeguard requires matching
posterior and simulation identities as well as fit and scenario identity.
The current published legacy snapshot is not rewritten by these changes.

## Software verification

The [verification record](../outputs/championship/development/overall-export-contract-checks-v1/verification.json)
records **155 passing focused Python checks**, a successful dashboard build and
**131 passing browser checks** for the repository URL. The shared synthetic
payload exercises both validators. An adapter check consumes actual bounded
accumulator output and demonstrates that a paired interval can be narrower than
a difference of marginal intervals. The browser suite checks the displayed
comparison in both directions, rejected inventories/identities/precision,
existing dashboard views and small-screen accessibility.

The record preserves the initial fixture block-size failure and Node JSON-import
failure, their corrections and the subsequent successful checks. No scientific
setting or acceptance threshold changed. The Python payload validator also
passes a standalone check with site packages disabled, and its lightweight tests
are included in the dashboard workflow. The simulator-dependent adapter check
stays in the fitting test suite.

These are local software checks using synthetic successor data and the existing
committed legacy payload. They do not establish clean-clone CI, intended-runner
resources, accepted real-data export or production readiness. The authenticated
acceptance controller, canonical producer, full refresh/cache integration and
regeneration of affected publication artifacts remain required before approved
deployment.
