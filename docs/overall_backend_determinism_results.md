# Deterministic GPU fixed-input candidate passed

The [frozen candidate](overall_backend_determinism_protocol.json) completed the
entire original fixed-input inventory with double precision and the registered
deterministic-operation setting. Every retained value, gradient and model-state
output matched bitwise across fresh GPU repetitions. Each repetition also met
the unchanged original comparison budgets against the committed double-precision
CPU reference. The original failed single-precision screen remains failed.

The complete [evaluation execution record](../outputs/championship/development/overall-backend-determinism-evaluate-execution-v1/run.json)
and [comparison execution record](../outputs/championship/development/overall-backend-determinism-compare-execution-v1/run.json)
account for every case. Their per-case comparison records report:

| Case | Bitwise GPU repetition | Both repetitions within CPU comparison budgets |
|---|---|---|
| [connected](../outputs/championship/development/overall-backend-determinism-v1/compare/cpu32/connected/result.json) | pass | pass |
| [confounded](../outputs/championship/development/overall-backend-determinism-v1/compare/cpu32/confounded/result.json) | pass | pass |
| [sparse_rookie](../outputs/championship/development/overall-backend-determinism-v1/compare/cpu32/sparse_rookie/result.json) | pass | pass |
| [dominant_teammate](../outputs/championship/development/overall-backend-determinism-v1/compare/cpu32/dominant_teammate/result.json) | pass | pass |
| [era_boundary](../outputs/championship/development/overall-backend-determinism-v1/compare/cpu32/era_boundary/result.json) | pass | pass |
| [real_network](../outputs/championship/development/overall-backend-determinism-v1/compare/cpu32/real_network/result.json) | pass | pass |

This is evidence for the tested numerical implementation at the complete fixed
coordinates, including all original failed-attempt endpoints. It is not a proof
of determinism for every compiled computation or all possible parameter values.
The compiler setting changes multiple implementation choices together; this
experiment does not isolate the cause of earlier variation.

The [sampling and restart work](overall_refresh_runtime.md) can now proceed to a
separate, prospectively defined engineering check of complete model chains.
Repeated fixed-coordinate evaluation is insufficient to establish checkpoint
replay, posterior accuracy, fitting speed or memory. The candidate has not run
a full fit, passed recovery/calibration, produced a driver ranking or demonstrated
the next-morning update target. Production settings and deployment are unchanged.
