# Equivalent Student-t arithmetic correction: frozen research definition

The [previous audit](overall_studentt_numerics_results.md) failed its primitive
accuracy criteria. Preserve those results and the exhausted sampling ladder.
This is a separate numerical implementation experiment; likelihood, priors,
support, random sampling, scientific gates and error budgets remain unchanged.
No model fitting or publication is authorized by passing this arithmetic check.

## Diagnosed formula and proposed normalizer

The installed JAX gamma-ratio routine uses `x = h/(1+h)` while assuming `a <= b`.
The [SciPy source cited by that routine](https://raw.githubusercontent.com/scipy/scipy/c89dfc2b90d993f2a8174e57e0cbc8fbe6f3ee19/scipy/special/cdflib/algdiv.f)
uses `x = 1/(1+h)` in that case. The latter is the ratio `b/(b+a)` needed in the
geometric sum of correction terms. The installed library source hash remains
bound by the previous protocol. Do not patch installed files.

Implement the half-argument gamma ratio directly from the
[Stirling expansion and positive-real remainder bound](https://dlmf.nist.gov/5.11#ii).
For `b=df/2`, write the unit-scale normalizer as
`log(2*pi)/2 + 1/2 - b*log1p(1/(2*b)) + delta(b)-delta(b+1/2)`.
Evaluate the correction difference with `x=b/(b+1/2)` and geometric sums;
do not subtract nearly equal inverse powers. Use the exact rational Bernoulli
coefficients committed in the implementation, including its fixed term count.
The first omitted positive-real Stirling terms bound the two separate remainders.
Retain the library's existing branch location and the direct log-gamma expression
below it. Bound inactive branch arguments only to avoid invalid intermediate
calculations; the selected density is unchanged.

## Stable residual derivative

With `u=residual^2/df` and `q=u/(1+u)`, the residual term's derivative with respect
to `log(df)` is `(q - df*(log1p(u)-q))/2`. Evaluate the difference as
`sum(q^k/k, k=2..N)` when `q <= Q`, with `N` and `Q` fixed in the source.
Its positive remainder is bounded by `q^(N+1)/((N+1)*(1-q))`; relative to the
leading term, the bound is `2*q^(N-1)/((N+1)*(1-q))`. The committed choices
place this below the existing double-precision error budget by arithmetic
reasoning, before the new grid is evaluated. Outside this region use the direct
difference. Retain the exact log-density value expression and the analytic
residual derivative through a custom JVP, including higher differentiation.
This changes arithmetic, not the distribution's tail behavior.

## Fixed experiment and interpretation

Freeze sources, tests, this definition, installed library identities, the repaired
reference protocol and both prior primitive result files before executing the
new audit. Retain every original point and its exact inputs. Append the positive
and negative float32 residual nearest the new series boundary, and each adjacent
float32 residual, for every original parameter point. Lift these same rounded
coordinates for the double-precision process.

Run the original density, installed log-beta density, corrected-normalizer-only
density and complete stable density on every row. Reproduce the old methods'
values exactly on the original inventory; failures must remain recorded. Use the
unchanged independent reference precisions and error budgets, retaining every
value, derivative and failure. The candidate passes this primitive investigation
only if all its rows meet the budget in both precisions. The separate normalizer
arm distinguishes its effect from the residual-derivative correction.

Then evaluate the complete stable calculation at every original fixed network
point, using the unchanged observations, coordinate map and rotation, in both
precisions. Compare complete gradients; preserve finite checks and all differences.
Always record this descriptive comparison, including after a primitive failure;
it cannot override a failed primitive criterion. Process completion is not a
successful criterion. Do not infer posterior convergence from fixed-point
arithmetic agreement.

Execute stages sequentially with the standing resource caps. Preserve logs,
source hashes, active scope evidence and separate timing measurements. Production
integration, new computational and scientific evidence, full acceptance and
deployment remain later requirements of the [Overall plan](overall_model_plan.md).
