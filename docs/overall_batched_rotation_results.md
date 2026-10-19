# Complete-model batching comparison

The [frozen registration](overall_batched_rotation_audit_protocol.json) binds the
[definition](overall_batched_rotation.md), source, original preflight designs,
generation seeds, computational environment and unchanged equivalence tolerances.
It was committed before measurement. The
[complete run summary](../outputs/championship/development/overall-batched-rotation-audit-v1/summary.json)
accounts for every required design, both implementations and their separate
comparison processes.

The [verification record](../outputs/championship/development/overall-batched-rotation-audit-v1/verification.json)
finds identical supplied coordinates, reconstructed states, probability factors,
joint densities, unconstrained potentials and full gradients at every tested
point. The recorded maximum absolute discrepancy is zero for every comparison
kind in every case. Inventories, shapes, dtypes, generating-data identities,
original maps and full gradient-coordinate counts agree. All fixed checks pass.

This supports integrating the batched evaluation of the existing map in a later
frozen runner. The original profile and its exhausted convergence failure remain
unchanged. This comparison samples no posterior and grants no ranking acceptance.

## Recorded computation cost

The table shows compilation plus the first synchronized complete potential/gradient
call, the range of all registered repeated calls, and process peak resident memory.
Every cell lists **original / batched**. The repeated-call ranges include all
8 evaluations at the same original generating point for each implementation;
the raw lists are preserved in the verification record. They are descriptive
measurements, with no fitted speedup model or statistical confidence claim.

| Case | Compile + first call (s) | Repeated-call range (ms) | Peak RSS (KiB) |
|---|---:|---:|---:|
| connected | 3.700 / 2.992 | 0.554–0.755 / 0.577–0.744 | 910,432 / 900,012 |
| confounded | 3.655 / 3.067 | 0.646–0.781 / 0.574–0.792 | 911,492 / 895,164 |
| sparse rookie | 3.664 / 3.112 | 0.574–0.740 / 0.597–0.709 | 924,760 / 914,952 |
| dominant teammate | 3.704 / 3.095 | 0.644–0.888 / 0.575–0.710 | 911,832 / 906,424 |
| era boundary | 3.678 / 3.016 | 0.590–0.774 / 0.586–0.681 | 912,204 / 902,120 |
| real network | 18.664 / 5.382 | 9.617–14.037 / 3.454–6.009 | 1,675,952 / 1,278,808 |

The smaller-fixture timing ranges overlap. At the real-network generating point,
every recorded batched repeat took less time than every original repeat in this
measurement. Compilation and peak resident memory are also lower for that case.
No measured NUTS trajectory or inference throughput follows from this single-point
calculation benchmark.

Each numerical child verified membership in its own scope with **3G** memory,
**200%** CPU and nice **19**, with no populated sibling work in the shared slice.
Both implementations used the same explicit CPU environment and disabled persistent
compilation caching. The original and batched compilation, execution and external
resource logs are retained for every case. The overall
[execution record](../outputs/championship/development/overall-batched-rotation-audit-v1-execution/execution.json)
identifies the controller and committed source. Internal and external clocks remain
separate measurements, consistent with the clock limitation documented in the
[profile report](overall_complete_profile_results.md). These are local measurements;
intended-runner performance remains unverified.

Fitting convergence, full recovery and simulation-based calibration, functional
precision, predictive validation, production integration and approved publication
remain required. This computation result does not reopen the exhausted retry
sequence or establish a publishable Overall ranking.
