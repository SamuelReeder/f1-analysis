# Exact information identities in model comparisons

The comparison contract already required a common information origin. The
development implementation checked only the origin category, field, cutoff and
target label. Synthetic regressions showed that changing a model's actual
forecast time, input snapshot, common training history or conditioning-rule
identity could still produce a complete, passing comparison. No real model
result is reclassified by this finding.

The corrected [scorer](../analysis/overall_predictive_scores.py) requires each
inventory entry and every arm's result to identify:

- the exact timezone-aware forecast origin;
- the same normalized outcome-free input snapshot;
- the common bounded upstream training-history inventory;
- the same registered scoring target and conditioning definition.

The origin is normalized to UTC. Content identities use canonical SHA-256
digests. Missing, malformed or mismatched result identities leave their required
event unresolved and prevent a comparison; an invalid common inventory is
rejected. The comparison records a digest of the normalized complete inventory,
including the field. Equivalent timezone spellings and input row permutations
do not change that identity.

The common training-history digest is not an arm's model-array digest. The
race-only arm deliberately discards qualifying observations under its separately
specified definition; other arms also have different array layouts. They must
still originate from the same preserved, cutoff-bounded upstream inventory.
Likewise, the target-definition digest must identify the actual registered
conditioning rule. Sharing a broad label while one arm conditions on additional
completion information is insufficient.

These are consistency checks on claims. A matching digest string does not prove
that its underlying artifact exists, contains the claimed inputs or was available
at the declared origin. The evaluation controller must verify the preserved
snapshot, common history and target specification against those identities.
The comparison explicitly leaves source availability and publication false.
This correction supplies neither historical source evidence nor the remaining
classification-conditioning decision or final predictive registration.

## Evidence

The [verification record](../outputs/championship/development/overall-comparison-origin-checks/verification.json)
preserves the reproduced failures and passing corrected focused checks. It also
compares the numerical scoring and season-block comparison functions against
the preceding implementation. Their formulas, priors, bootstrap settings and
acceptance thresholds remain unchanged; no model was fitted or real-event score
evaluated to choose this correction.

The fitting sources, input data and pending numerical-study proposal remain
unchanged. Full integration checks still await the separate race-pace
regeneration; this change does not authorize publication or resolve the failed
Overall numerical case.
