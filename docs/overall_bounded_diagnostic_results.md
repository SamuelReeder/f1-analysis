# Bounded diagnostics: equivalence and resource results

The [registered computation change](overall_bounded_diagnostics.md) completed its
full verification. The [saved evidence](../outputs/championship/development/overall-bounded-diagnostic-checks/verification.json)
binds the implementation, tests, original attempts and complete resource workload.

All 7 saved attempts from the approved weekend-coordinate profile reproduce their
entire original parameter diagnostic reports exactly, including the failed
attempts. Connected and sparse-rookie each retain their failed initial attempt
and converged retry; confounded retains its failed initial and second attempts
and converged third attempt. The implementation changes neither the statistics
nor the acceptance decisions. Focused semantic checks pass 25 tests, covering
different coordinate group sizes, array layouts and invalid inputs.

The full resource fixture contains all 93 retained fields at the real-network
largest retry shape: 4 chains and 3,200 draws per chain, with
7,494,144,000 bytes of logical array values. Every field receives the frozen
seeded synthetic values; no coordinate or iteration was removed. These values
are independent normal fixtures, not draws from the joint model. Their passing
synthetic diagnostic result provides no scientific convergence evidence.

The complete mapped-input check finished under the verified 3G memory and 200%
CPU scope, in the unchanged f1 slice with nice 19. Recorded process peak RSS is
3,098,972 KiB. Clean file-backed pages contribute to this measurement; it is not
an isolated anonymous-working-memory measurement or a full-fit peak. Temporary
arrays were removed after completion.

The [audit](../outputs/championship/development/overall-bounded-diagnostic-checks/audit/audit.json)
records 36.158 seconds to construct the full fixture and 709.806 seconds for its
diagnostic calculation. The entire audit's monotonic duration is 783.594 seconds,
while [external wall time](../outputs/championship/development/overall-bounded-diagnostic-checks/audit-resources.txt)
is 11:54.76. Both measurements are retained; their disagreement is unexplained.

The [complete-design fitting runner](overall_complete_profile.md) now uses this
bounded checker while retaining the original collector, sampler and fixed retry
ladder. Its [connected software checks](../outputs/championship/development/overall-complete-profile-checks/verification.json)
pass 75 tests, overlapping the earlier focused checks. This is not a new full
repository, browser or hosted-CI run. Actual sampler memory, inference on the
remaining designs, full calibration and functional ranking precision still need
their separate evidence. No real-driver ranking or publication was produced.
