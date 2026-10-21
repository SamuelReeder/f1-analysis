# CPU/GPU comparison: full-network gradient screen failed

The [frozen comparison](overall_backend_comparison_protocol.json) completed every
prepared point and both backend evaluations. The
[complete comparison record](../outputs/championship/development/overall-backend-compare-execution-v1/run.json)
reports a failed full-network numerical screen. The smaller cases passed.
The registered budgets, points, data, model and original failed attempts are
unchanged. The proposed matched GPU sampling benchmark has not started.

## Complete inventory and outcome

Each case includes its generating point and the first and last saved draw of
every chain of every recorded attempt. The input preparation verified the
original observations, generating parameters, truth and coordinate map; both
backends used the same serialized unconstrained coordinates and observations.
The [software checks](../outputs/championship/development/overall-backend-comparison-checks-v1/stdout.txt)
record **14** passing tests of complete inventories and corruption rejection.

| Case | Fixed points | Potential-energy screen | Complete state screen | Gradient screen | Failing gradient coordinates |
|---|---:|---|---|---|---:|
| [connected](../outputs/championship/development/overall-backend-comparison-v1/compare/cpu/connected/result.json) | 17 | pass | pass | pass | 0 |
| [confounded](../outputs/championship/development/overall-backend-comparison-v1/compare/cpu/confounded/result.json) | 9 | pass | pass | pass | 0 |
| [sparse_rookie](../outputs/championship/development/overall-backend-comparison-v1/compare/cpu/sparse_rookie/result.json) | 17 | pass | pass | pass | 0 |
| [dominant_teammate](../outputs/championship/development/overall-backend-comparison-v1/compare/cpu/dominant_teammate/result.json) | 9 | pass | pass | pass | 0 |
| [era_boundary](../outputs/championship/development/overall-backend-comparison-v1/compare/cpu/era_boundary/result.json) | 9 | pass | pass | pass | 0 |
| [real_network](../outputs/championship/development/overall-backend-comparison-v1/compare/cpu/real_network/result.json) | 25 | pass | pass | fail | 115 |

The full-network failure is in derivatives, despite passing potential-energy and
retained-state comparisons. This does not show which backend is more accurate.
The completed CPU profile remains the original recorded computation; this
comparison does not retroactively change its convergence decision or establish
scientific acceptance of either implementation.

## Full-network discrepancies

The [original failure readout](../outputs/championship/development/overall-backend-failure-readout-v1/readout.json)
lists the CPU and GPU values, original point identity and unchanged budget at
every failed coordinate. It performs no new model evaluation.

| Gradient field | Failing coordinates | Largest absolute discrepancy | Largest discrepancy / registered budget |
|---|---:|---:|---:|
| `age_slope` | 15 | 0.073059082 | 10.998788 |
| `car_jump_z` | 12 | 0.010187149 | 2.014041 |
| `nu` | 18 | 0.093021616 | 18.603304 |
| `overall_car_link` | 4 | 0.014068604 | 2.383993 |
| `overall_lap_controls_rotated` | 3 | 0.006278992 | 1.238735 |
| `overall_lap_nu` | 5 | 0.030242920 | 5.921473 |
| `overall_qualifying_link` | 2 | 0.006317139 | 1.252715 |
| `overall_sd_lap` | 14 | 0.026653290 | 5.200480 |
| `rho` | 7 | 0.019989014 | 3.538492 |
| `rho_reset` | 6 | 0.016387939 | 2.934010 |
| `sd_car_reset` | 12 | 0.051696777 | 7.403253 |
| `sd_car_season` | 12 | 0.055664062 | 6.845362 |
| `sd_skill_season` | 5 | 0.012657166 | 2.464669 |

For example, at the saved `nu` coordinate with the largest discrepancy, CPU gives
**0.136975542** and GPU gives **0.229997158**, a difference of **0.093021616**
against the registered budget of **0.005000274**. That point is retained in the
readout, including its original attempt, chain and draw identity. The discrepancy
cannot be dismissed solely because few coordinates fail among many checked.

The evidence supports investigating accumulated floating-point error in the
large model, but does not establish that as the cause. A higher-precision
comparison must reuse the full fixed-point inventory, report complete gradients
and reproduce the original single-precision baseline. It must preserve these
failures and cannot relax the screen or select favorable points. No precision
change, new fit or new scientific acceptance decision follows from this report.

## Recorded fixed-point costs

The linked per-case evaluation records under
[CPU](../outputs/championship/development/overall-backend-comparison-v1/evaluate/cpu/real_network/result.json)
and [GPU](../outputs/championship/development/overall-backend-comparison-v1/evaluate/gpu/real_network/result.json)
supply the following stage measurements. Execution and transfer columns sum
all declared points for that case; lowering and compilation are separate.

| Case | Backend | Lowering seconds | Compilation seconds | Synchronized execution seconds | Transfer seconds | Peak host RSS (KiB) |
|---|---|---:|---:|---:|---:|---:|
| connected | cpu | 1.146857 | 2.790948 | 0.013171 | 0.003968 | 585,432 |
| connected | gpu | 1.250788 | 12.248977 | 0.241219 | 3.407990 | 1,367,104 |
| confounded | cpu | 1.146102 | 2.757989 | 0.006541 | 0.002299 | 581,872 |
| confounded | gpu | 1.140862 | 11.264496 | 0.069014 | 1.965084 | 1,354,032 |
| sparse_rookie | cpu | 0.993790 | 3.060127 | 0.015572 | 0.008069 | 593,964 |
| sparse_rookie | gpu | 1.088069 | 12.681749 | 0.086070 | 3.583218 | 1,301,344 |
| dominant_teammate | cpu | 0.942657 | 2.923696 | 0.006879 | 0.002283 | 580,048 |
| dominant_teammate | gpu | 1.131649 | 10.751208 | 0.052510 | 1.715564 | 1,355,964 |
| era_boundary | cpu | 1.055686 | 2.765082 | 0.007036 | 0.002171 | 583,868 |
| era_boundary | gpu | 1.003645 | 10.215217 | 0.049151 | 1.746503 | 1,286,440 |
| real_network | cpu | 1.329816 | 5.346466 | 0.146477 | 0.006838 | 747,436 |
| real_network | gpu | 1.327566 | 19.092379 | 0.170655 | 4.453770 | 1,515,772 |

The [external execution records](../outputs/championship/development/overall-backend-evaluate-execution-v1/run.json)
retain separate process timing and resource files. Internal and external clocks
must not be combined into a claimed exact runtime. GPU admission was checked
before each GPU process; the study did not measure peak device memory.

This workload includes full state outputs and individual host transfers. MCMC
uses repeated gradient calculations inside compiled transitions and a different
collection schedule. These measurements do not establish a full-fit speedup,
production runner feasibility or next-morning publication. Full-model numerical
agreement must be resolved before the matched sampling benchmark. Restart
support, warm initialization, final archives/diagnostics integration and the
complete refresh benchmark remain part of the [runtime work](overall_refresh_runtime.md).
