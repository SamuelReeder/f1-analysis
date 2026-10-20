# Overall now uses the independently checked Student-t calculation

The corrected density and derivative implementation is integrated into
[`f1rank/overall/studentt.py`](../f1rank/overall/studentt.py). Overall explicitly
selects it for qualifying observations, car-step innovations and lap innovations.
The legacy model keeps its original distribution by default. The new distribution
inherits the original sampler, support and parameterization; no likelihood,
prior, tail-weight parameter, retry setting or scientific criterion changed.

The inference method is now
`conditional_weekend_batched_controls_stable_studentt_v1`. A fresh registration
must bind that method and the complete current source/data key. The historical
failed fits and their exhausted ladder remain preserved. This is a numerical
implementation correction after diagnosed failures, not independent validation
or another attempt in the exhausted run.

## Complete migration evidence

The [source-bound production audit](overall_studentt_production_protocol.json)
evaluated the actual production coordinate model at all **33** saved network
points, retaining **59,466** gradient coordinates per point. Its potential and
every gradient coordinate are exactly identical to the independently checked
research implementation in both float32 and float64. The
[verification record](../outputs/championship/development/overall-studentt-production-audit-v1-execution/verification.json)
binds complete records and cache hashes. This is stronger than checking only a
summary norm or a selected gradient coordinate.

The focused production tests also reproduce the complete primitive reference
inventory, preserve inherited random samples and distribution tree reconstruction,
check explicit model bindings and retain the original legacy default. They record
**7** passing checks in each precision; these are the same tests, not additive
unique counts. The earlier integration run preserved **41** passing checks and
**3** failures in a new fixture that tried to sample the legacy ImproperUniform
car prior. Conditioning that prior in the fixture resolved those test failures.
The [original failed output](../outputs/championship/development/overall-studentt-production-checks-v1/float32.stdout.txt)
remains committed. No model setting or numerical criterion was changed to resolve
the fixture error.

| Full production comparison | Internal seconds | Separate external elapsed | Peak RSS (KiB) |
|---|---:|---:|---:|
| Float32 | 6.814 | 0:07.77 | 769,080 |
| Float64 | 6.797 | 0:06.73 | 821,276 |

Each comparison verified its active **3G**, **200%** CPU scope and nice **19**;
execution was sequential. These clocks and local resource observations remain
separate and do not predict full-fit or intended-runner performance.

## What this permits next

The arithmetic correction is ready for a newly registered computational profile.
Preserve the model, the original complete synthetic case inventory, generating
seeds, fixed sampling ladder and convergence criteria. Bind the new source and
its numerical evidence before fitting, retain complete sampler evidence and
respect the user's current CPU and memory limits. Do not reuse an old scientific
registration date or relabel the failed earlier profile.

No new MCMC ran during these checks. A converged full fit, complete recovery and
calibration, chronological and independent predictive validation, accepted public
producer, refresh integration and approved rollout remain outstanding in the
[Overall plan](overall_model_plan.md). The changed model-source identities require
affected outputs to be regenerated before publication. This integration does not
authorize an Overall ranking.
