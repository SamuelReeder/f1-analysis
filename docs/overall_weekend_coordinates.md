# Proposed conditional weekend coordinates

This is a separate computational proposal after the failed control-coordinate
experiment and its [complete diagnosis](overall_weekend_diagnosis_results.md).
It is not a change to the probability model or a rescue of either failed run.
No new fit is authorized by preparing this document or its prototype.

## Fixed transformation

Use the original qualifying form scale `sf`, independent race form scale `sr`
and shared weekend loading `b`, with their exact original independent priors.
For every entry define `s = hypot(b * sf, sr)`, `a = b * sf / s`, `c = sr / s`.
The scales are positive on their original support; do not clip or regularize
them. Sample independent standard Normal vectors `u` and `v` and reconstruct
the original innovations as

```
zq = a * u + c * v
zr = c * u - a * v
```

The matrix is orthogonal and self-inverse because `a² + c² = 1`. Its absolute
determinant is one. Conditional on the unchanged hyperparameters, reconstructed
`zq` and `zr` have their original independent standard Normal distribution.
Hyperparameters are unchanged coordinates, so the full transformation Jacobian
is block triangular and also has absolute determinant one, despite the
innovation map's dependence on those hyperparameters. No extra density factor
or new prior is needed.

With the original common event-centering projection `P`, qualifying form is
`sf * P(a*u + c*v)` and race form is `s * P(u)`. The implementation reconstructs
the original sites and evaluates the original model, preserving its likelihood
and output definitions. Keep all entry directions, including event means that
the likelihood centers away. There is no rank truncation or selection of
well-behaved entries, chains or attempts.

Compose this fixed transformation with the already evaluated lap-control
rotation. The new transformation uses hyperparameters as coordinates, not
estimated posterior means, observed responses, chosen diagnostic cutoffs or
driver rankings. It has no choice of tuning constants. Preparing it after seeing
a failure makes it a post hoc computational proposal, not independent scientific
evidence. It does not strengthen information about driver/car separation.

## Before any sampling

Check invertibility, orthogonality, conditional covariance and full Jacobian,
including loading zero and negative, and unequal scales. Compare the complete
joint density and each observation factor against the original model, both
alone and composed with the existing control rotation. Retain every original
state and check finite gradients. The model's existing prior families and
parameters must be checked rather than silently assuming an updated model is
equivalent. These are mathematical/software checks, not a convergence result.

The separate proposed study must retain the original complete study inventory,
synthetic observations and generating truths, sampling seeds, sampler settings,
retry ladder and numerical thresholds. Report all original states, all added
coordinates, every current qualifying/race driver contrast, resources and every
failed attempt. A favorable earlier attempt cannot replace an exhausted ladder.
Do not extend the ladder or select another coordinate variant after the run.

The proposal uses a new output/cache identity, refuses an existing experiment,
binds every implementation and original truth file, and requires separately
approved, committed frozen registration before fitting. Its scope remains
synthetic computational feasibility. Full recovery, SBC, ranking precision,
predictive validation, real-data integration and production acceptance remain
required regardless of this experiment's result. There is no push, merge,
release or ranking publication in this proposal.
