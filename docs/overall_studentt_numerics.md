# Independent Student-t arithmetic investigation

Preserve the failed fits, original observations, all fixed network points and
model assumptions. Investigate the Student-t calculation through an algebraic
identity, without MCMC or a new model-acceptance rule. The proposed normalizer is
`log(scale) + log(df)/2 + betaln(1/2, df/2)`, which equals the existing expression
using the difference of log-gamma terms. Keep the residual-dependent term,
distribution support, prior, random generator and observation model unchanged.

First freeze the source, installed numerical-library identities, prior
fixed-coordinate cache and separate reference-wheel identity. Import the pinned
mpmath wheel directly from its isolated cache; do not install it into or modify
the frozen fitting environment. Its [published release](https://pypi.org/project/mpmath/1.3.0/)
provides arbitrary-precision arithmetic and the wheel checksum.

## Independent primitive reference

Use the complete saved-coordinate inventory for both `nu` and `overall_lap_nu`.
Add the fixed logarithmic grid and boundary points defined by
[`overall_studentt_numerics.py`](../analysis/overall_studentt_numerics.py): quarter-decade
degrees of freedom from `10^-3` through `10^8`, the named special values and
adjacent float32 log-coordinates around the log-beta implementation's branch.
Round each log-coordinate once to float32 and lift that same exact value for
float64 evaluation. Cross every point with the complete declared standardized
residual grid. Evaluate at zero location and log-scale, retaining derivatives
with respect to log degrees of freedom, location and log-scale.

Compute the defining log-gamma density and independent numerical derivatives with
mpmath at both declared arbitrary precisions. Require their rounded float64
references to agree. Evaluate both the original and log-beta calculations in
separate float32 and float64 processes. Preserve every value and derivative,
including failures. The fixed numerical error budget is the precision-specific
multiplier in the code times machine epsilon times `1 + abs(reference)`. These
criteria assess arithmetic on this declared domain, not model convergence,
posterior calibration or scientific acceptance. They must not be relaxed after
the results are seen.

## Matched whole-model evaluation

Evaluate every original network point through the unchanged frozen model with
only the Student-t log-probability arithmetic temporarily replaced in the
dedicated process. Restore the original method on exit and leave installed files
unchanged. This explicit research substitution affects every Student-t density
evaluation, including fixed-degree priors; it does not call a sampler.

Reuse the checked fixed-coordinate evaluator and complete derivative comparison
in both precisions. Preserve every field and all original baseline results.
Its `original_float32_report_identical` flag is expected to be false where the
new arithmetic differs; it is retained as a comparison with the old calculation,
not used as an acceptance claim for the changed arithmetic. Report primitive
reference failures even if the whole-model discrepancies improve. Improvement
cannot prove that this caused the original sampling failure or that another fit
will converge.

Run each stage sequentially in the standing capped analysis scope, retaining
progress, failures, resource measurements and cache hashes. Numerical correction
of the production code, further computational profiles, full scientific checks
and publication remain later requirements of the complete plan.
