# Student-t arithmetic audit: correction remains unaccepted

The [repaired reference audit](overall_studentt_numerics_v2_protocol.json)
completed every declared calculation. The independent reference now stabilizes,
but the proposed log-beta calculation fails the fixed arithmetic criteria in both
execution precisions. It is not ready for production integration or another
full-model fit. No sampling, publication or scientific acceptance followed.

## Reference repair and preserved failures

The [reference diagnosis](../outputs/championship/development/overall-studentt-reference-diagnosis-v1/result.json)
records **64** numerical differentiation discrepancies among **1,053** input rows.
They occur at mathematically exact zero derivatives. The analytic reference has
no precision-instability rows and no disagreement with nonzero high-precision
numerical derivatives. The repaired study retains the same density, inputs,
candidate implementations, precisions and error budgets. The original failed
audit remains preserved under its original source and protocol.

The [verification record](../outputs/championship/development/overall-studentt-numerics-v2-execution/verification.json)
confirms that the repaired study reproduces all original float32 candidate values
and failure rows exactly. Each precision evaluates all **1,053** rows, including
the saved-coordinate inventory and the declared grid and boundary points.

| Execution precision | Original calculation: failing rows | Log-beta calculation: failing rows |
|---|---:|---:|
| Float32 | 303 | 18 |
| Float64 | 260 | 291 |

These are failures against precision-specific fixed budgets, so counts across
precisions are not a comparison under the same absolute tolerance. Passing a
subset or improving the whole-model comparison cannot rescue a failed primitive
criterion. The float32 log-beta failures involve the log-degrees-of-freedom
derivative on the declared grid; the remaining error needs diagnosis without
relaxing the budget or changing the likelihood.

## Whole-model comparison

All **33** original fixed network points and **59,466** gradient coordinates per
point were retained. The largest absolute potential discrepancy in the first
failed-attempt group falls from **803.866138** in the
[original precision study](overall_network_precision_results.md) to **0.056534**
with the log-beta calculation. Its largest absolute gradient-coordinate discrepancy
falls from **907.734059** to **0.027080**. Complete groups, including the generating
point and all other exhausted attempts, remain in the verification record.

This isolates an arithmetic sensitivity and supports further numerical work. It
does not establish that the candidate is accurate throughout the declared domain,
that this caused the stalled sampler, or that another fit would converge. Both
precisions can share an error. The exhausted fit ladder remains closed.

## Local checks and resources

The [focused regression record](../outputs/championship/development/overall-studentt-reference-repair-checks/pytest.txt)
reports **19** passing tests. Each arithmetic stage verified its active **3G**
memory and **200%** CPU scope with nice **19**. The stages ran sequentially.

| Stage | Internal seconds | Separate external elapsed | Peak RSS (KiB) |
|---|---:|---:|---:|
| Primitive float32 | 2.193 | 0:02.04 | 393,960 |
| Primitive float64 | 2.033 | 0:02.35 | 385,084 |
| Whole model float32 | 8.840 | 0:08.20 | 771,356 |
| Whole model float64 | 8.637 | 0:08.03 | 803,464 |
| Complete comparison | 2.718 | 0:02.97 | 90,644 |

These separately recorded clocks measure local deterministic checks, not the
runtime of a successful fit or production refresh. Production model code remains
unchanged by this audit. The next numerical task is to isolate the remaining
normalizer and residual-term derivative errors against this repaired reference.
Successful fitting, scientific validation and deployment remain outstanding in
the [complete Overall plan](overall_model_plan.md); there is no dependable
completion date yet.
