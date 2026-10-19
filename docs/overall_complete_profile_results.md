# Complete-design profile: interrupted run and exhausted continuation

The [interruption record](../outputs/championship/development/overall-complete-design-profile-v1-interruption/interruption.json)
preserves the original outputs and cache metadata after a new WSL boot. The
original service is missing, and the final case, whole-profile summary and
service exit record are absent. The restart's origin was unknown when that
record was written. The user has since
[confirmed initiating the restart](../outputs/championship/development/overall-complete-design-profile-v1-restart-report/user-report.json),
without specifying Windows versus WSL, its exact time or their reason for
restarting. The original interruption record remains unchanged. The previous
boot's Codex fatal-signal record does not establish that the fit caused an OOM;
the cause of the separately reported earlier OOM remains unverified.

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
At the original interruption, the full retry ladder had not been exhausted.
The separate continuation below subsequently exhausted it.

The [resource-change record](../outputs/championship/development/overall-complete-design-profile-v1-resource-changes/20261013T170054Z-cpu-reduction.json)
records the user's shared CPU reduction from **800%** to **400%** during the run.
The lower cap remains in force after the restart. The original **13G** memory cap
is a limit, not an observed final peak. Measurements spanning the cap change
cannot be attributed entirely to either CPU limit. No complete run duration or
final resource measurement is available for the original interrupted run.

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
The separate [execution registration](overall_profile_continuation_protocol.json)
is frozen, and the [launch observation](../outputs/championship/development/overall-complete-design-continuation-v1-launch/sampling-start.json)
records the unfinished original attempt starting in its new capped service.
Earlier attempts and original files remain unchanged. The original runner has
not been blindly relaunched, and the prepared batching implementation is not
substituted into the continuation. Its terminal result is recorded below.

A [partial observation dated 2026-10-19](../outputs/championship/development/overall-complete-design-continuation-v1-observations/20261019-status/observation.json)
preserves the continuation's completed attempt **3**, which failed convergence.
Its [saved diagnostic record](../outputs/championship/development/overall-complete-design-continuation-v1-observations/20261019-status/attempt_3.json)
reports a maximum R-hat of approximately **6.122893e15**, minimum bulk ESS
**4.010025**, and **0** divergences. Its recorded duration is **225539.501
seconds**, including diagnostics, or approximately **62.650 internal-timer
hours**. Earlier engineering evidence contains UTC/monotonic clock disagreement;
this is not an independently verified wall-clock duration or a forecast of the
next attempt's cost. Independent reproduction from the retained arrays was
still pending at this partial observation; the completed audit is recorded below.

The [log snapshot](../outputs/championship/development/overall-complete-design-continuation-v1-observations/20261019-status/profile-log.txt)
then records attempt **4**, the final attempt in the unchanged registered ladder,
as sampling. It provides no iteration-level progress or reliable remaining-time
estimate. This partial observation preserves the failure without replacing the
original interrupted run's missing completion records or declaring the
continuation complete.

The [terminal continuation record](../outputs/championship/development/overall-complete-design-continuation-v1-terminal/terminal.json)
preserves its completion on 2026-10-19 with `failed_convergence` and wrapper exit
status **1**. The [raw result](../outputs/championship/development/overall-complete-design-continuation-v1/result.json)
records `combined_recorded_convergence: false`. The final registered attempt
failed; the fixed ladder is now exhausted. There was no execution exception.

| Continuation attempt | Maximum R-hat | Minimum bulk ESS | Divergences | Recorded seconds, including diagnostics |
|---|---:|---:|---:|---:|
| 3 | 6.122893e15 | 4.010025 | 0 | 225539.501 |
| 4 | 5.909770 | 4.137888 | 0 | 235920.495 |

Both new attempts record a tree-depth-limit fraction of **1.0** and failed
current-driver contrast checks. The terminal snapshot preserves every earlier
attempt and its original settings. The new coordinate and driver-contrast reports
have now been reproduced independently from the retained arrays, as recorded
below. The sampler extras cannot be independently reconstructed from the saved
posterior files.

The continuation records **461927.568** internal elapsed seconds and a process
peak of **13,369,612 KiB**. Its separate
[external resource report](../outputs/championship/development/overall-complete-design-continuation-v1-launch/resources.txt)
records **118:22:26** elapsed wall-clock time. These clocks disagree; neither
duration is presented as an interchangeable or independently verified measure
of the other. They cover this continuation only, not the original interrupted
execution or its earlier completed cases.

The [focused software checks](../outputs/championship/development/overall-continuation-software-checks-v1/verification.json)
pass **105** tests in **3.24 seconds**, as recorded in the
[test output](../outputs/championship/development/overall-continuation-software-checks-v1/pytest.txt).
They cover continuation aggregation, failed-evidence handling, batching and its
resource guards in the separate capped scope. They do not reproduce the newly
saved real-network arrays or establish complete-model batching equivalence.

The separate [continuation array audit](../outputs/championship/development/overall-continuation-review-v1/verification.json)
subsequently reproduced every coordinate and driver-contrast diagnostic in
attempts **3** and **4** exactly, including their failed convergence decisions.
It verifies their complete array inventories and reuses the earlier **10**
audited attempts only after checking their bound identities. All original cases
remain present; no cache entry is missing or unreferenced. The
[full review](../outputs/championship/development/overall-continuation-review-v1/review.json)
records **4947.702** internal elapsed seconds and a process peak of
**2,435,152 KiB** in its separate capped scope. The external resource log is also
preserved with the same clock limitation. Audit execution is complete and the
records are consistent; combined convergence and computational acceptance remain
false. The full-model batching measurement is still pending.

The combined profile does not satisfy convergence, and the original interrupted
run remains incomplete. This candidate cannot support an accepted Overall
ranking. Full recovery,
simulation-based calibration, functional precision, predictive validation,
production integration and publication remain outstanding. The dashboard still
has no accepted successor Overall ranking.
