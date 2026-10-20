# Equivalent Student-t arithmetic passes its declared reference checks

The [frozen audit](overall_studentt_stable_protocol.json) completed with the stable
candidate meeting the fixed value and derivative budgets on every declared row
in both precisions. The [verification record](../outputs/championship/development/overall-studentt-stable-v1-execution/verification.json)
contains **1,755** rows per precision: the original **1,053** rows plus **702**
signed series-boundary cases. The independent references stabilize throughout.
Original methods reproduce every earlier value exactly on the original inventory.

| Calculation | Float32 failing rows | Float64 failing rows |
|---|---:|---:|
| Original Student-t | 633 | 436 |
| Installed log-beta normalizer | 18 | 495 |
| Corrected normalizer alone | 18 | 14 |
| Corrected normalizer and stable residual derivative | 0 | 0 |

The original-study failures remain preserved. Counts here include the additional
boundary inventory and use precision-specific budgets. They must not be compared
across precisions as if the absolute tolerance were the same. No budget, model
assumption, prior, seed, retry setting or scientific acceptance rule changed.

The complete [matched-model comparison](../outputs/championship/development/overall-studentt-stable-v1/compare/result.json)
retains **33** original points and **59,466** gradient coordinates per point.
Every value and gradient is finite. Maximum discrepancies within each complete
group are:

| Point group | Absolute potential difference | Relative full-gradient difference | Absolute gradient-coordinate difference |
|---|---:|---:|---:|
| Generating point | 0.021822 | 0.000018622 | 0.026572 |
| Attempt 1 | 0.087784 | 0.000020837 | 0.027050 |
| Attempt 2 | 0.066515 | 0.000012778 | 0.017274 |
| Attempt 3 | 0.183097 | 0.000096511 | 0.121457 |
| Attempt 4 | 0.079614 | 0.000331358 | 2.637552 |

This is descriptive fixed-point evidence, not a whole-model accuracy criterion
or a convergence result. Arithmetic outside the declared domain has not been
exhaustively established. The correction does not prove that it caused the
original sampler to stall or that a fresh fit will succeed.

The execution record preserves **25** passing focused checks in each precision;
these are the same tests and should not be added as unique checks. Every stage
ran sequentially in its verified **3G**, **200%** CPU scope with nice **19**.
Timing and memory records remain stage-specific, with internal and external
clocks reported separately. These are local deterministic calculations, not
measurements of full fitting or CI resources.

Production integration must preserve these executable calculations and the
original distribution's sampler. A new source-bound inference registration is
required before another fit; the exhausted historical ladder remains closed.
The [Overall plan](overall_model_plan.md) still requires successful full inference,
complete scientific validation, accepted publication, refresh and approved rollout.
