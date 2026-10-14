# Complete-design profile: interrupted

The [interruption record](../outputs/championship/development/overall-complete-design-profile-v1-interruption/interruption.json)
preserves the original outputs and cache metadata after a new WSL boot. The
original service is missing, and the final case, whole-profile summary and
service exit record are absent. The cause of the restart is unknown. The previous
boot contains a Codex fatal-signal record; that does not establish why WSL
restarted or prove that the fit caused an OOM.

The completed cases record convergence under the frozen parameter gate. The
real-network case records failed attempts followed by an interrupted attempt.
The following values come from the committed original receipts, copied unchanged
into the interruption inventory. Their coordinate and driver-contrast reports
have subsequently been reproduced exactly from the saved archives, as recorded
below. Divergence counts remain original recorded inputs to that check.
Durations are the fitter's recorded monotonic seconds
for each attempt, including its diagnostics; they are not total-run wall time.

| Case and original receipt | Attempt | Recorded decision | Maximum R-hat | Divergences | Recorded seconds |
|---|---:|---|---:|---:|---:|
| [Connected](../outputs/championship/development/overall-complete-design-profile-v1-interruption/cache-metadata/connected/receipt.json) | 1 | Failed | 1.010748 | 0 | 92.414 |
| Connected | 2 | Converged | 1.005343 | 0 | 140.357 |
| [Confounded](../outputs/championship/development/overall-complete-design-profile-v1-interruption/cache-metadata/confounded/receipt.json) | 1 | Failed | 1.010395 | 0 | 130.688 |
| Confounded | 2 | Converged | 1.005718 | 0 | 231.715 |
| [Sparse rookie](../outputs/championship/development/overall-complete-design-profile-v1-interruption/cache-metadata/sparse_rookie/receipt.json) | 1 | Failed | 1.011913 | 0 | 77.628 |
| Sparse rookie | 2 | Converged | 1.005274 | 1 | 99.659 |
| [Dominant teammate](../outputs/championship/development/overall-complete-design-profile-v1-interruption/cache-metadata/dominant_teammate/receipt.json) | 1 | Converged | 1.009351 | 1 | 60.502 |
| [Era boundary](../outputs/championship/development/overall-complete-design-profile-v1-interruption/cache-metadata/era_boundary/receipt.json) | 1 | Converged | 1.009934 | 0 | 57.754 |
| [Real network](../outputs/championship/development/overall-complete-design-profile-v1-interruption/cache-metadata/real_network/receipt.json) | 1 | Failed | 15.588357 | 0 | 31576.368 |
| Real network | 2 | Failed | 16.227598 | 0 | 120355.604 |

All recorded current-driver qualifying and race contrasts pass in each completed
case's final attempt. They fail in both saved real-network attempts. Both
real-network attempts record a tree-depth-limit fraction of **1.0**. Their zero
divergence counts do not establish convergence. These are the original reports;
the sampler extras needed to independently reproduce those counts were not saved.

The [last recorded sampling start](../outputs/championship/development/overall-complete-design-profile-v1/real_network.output.txt)
is real-network attempt **3**, with warmup **2400**, samples per chain **3200**,
**4** chains, target acceptance **0.9** and seed **2**, exactly as specified by the
fixed ladder. It has no completed diagnostic report or saved draw archive. It
must not be assigned the preceding attempt's failure or an invented final status.
The full retry ladder has not been exhausted.

The [resource-change record](../outputs/championship/development/overall-complete-design-profile-v1-resource-changes/20261013T170054Z-cpu-reduction.json)
records the user's shared CPU reduction from **800%** to **400%** during the run.
The lower cap remains in force after the restart. The original **13G** memory cap
is a limit, not an observed final peak. Measurements spanning the cap change
cannot be attributed entirely to either CPU limit. No complete run duration or
final resource measurement is available.

The separate [software check](../outputs/championship/development/overall-prepared-profile-checks-v1/verification.json)
passes **103** tests in **4.94 seconds**, as recorded in its
[test output](../outputs/championship/development/overall-prepared-profile-checks-v1/pytest.txt).
It covers the prepared metadata collector, saved-draw verifier and batching
software under the ordinary capped scope. It does not reproduce the real saved
archives or execute the full-model batching benchmark.

The [interruption-aware metadata review](../outputs/championship/development/overall-interrupted-profile-review-v1/review.json)
finds the original records consistent while retaining the missing child
completion and incomplete whole-run status. The
[saved-archive audit](../outputs/championship/development/overall-interrupted-profile-diagnostics-v1/audit.json)
verifies all retained fields, dtypes and grouped axes, and reproduces every
coordinate and driver-contrast report exactly. All **10** saved attempts remain
in the audit, including the failed attempts. No discrepancy or case was discarded.
The audit records **1864.483** monotonic seconds and a process peak of
**2,464,652 KiB** under its separate capped scope. These are local archive-check
measurements, not fitting throughput or CI-runner measurements. Sampler extras
were not saved; divergence counts, tree-depth counts and acceptance probabilities
remain recorded values rather than independently recovered evidence.

The [execution-recovery definition](overall_profile_continuation.md) preserves
the original files and numerical settings. Its new controller restarts only the
unfinished registered attempt from its original seed, retaining the earlier
attempts unchanged and stopping according to the same fixed ladder. Its
[software checks](../outputs/championship/development/overall-profile-continuation-checks-v1/verification.json)
pass **44** tests in **0.18 seconds**, as recorded in the
[test output](../outputs/championship/development/overall-profile-continuation-checks-v1/pytest.txt).
The separate execution registration and service launch remain pending. The
original runner has not been blindly relaunched, and the prepared batching
implementation is not substituted into the continuation.

The complete computational profile remains incomplete. Full recovery,
simulation-based calibration, functional precision, predictive validation,
production integration and publication remain outstanding. The dashboard still
has no accepted successor Overall ranking.
