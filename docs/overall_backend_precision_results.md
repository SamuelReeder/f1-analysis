# Precision diagnosis: agreement improves, exact GPU replay remains unproven

The [frozen diagnostic](overall_backend_precision_protocol.json) evaluated every
original fixed point on both backends in single and double precision, including
fresh-process double-precision repetitions. All declared operations completed.
The original CPU single-precision results reproduced bitwise. The GPU results
did not, in any case. These mismatches remain in the complete comparisons;
the original failed numerical screen is unchanged.

Double-precision CPU/GPU comparisons satisfy the original numerical budgets in
every case. Both backends' single-precision gradients differ from their
double-precision evaluations. This supports precision sensitivity as a material
part of the original disagreement. It does not establish exact truth, isolate
each arithmetic operation responsible, or prove posterior accuracy.

## Complete inventory and repeatability

The [readout](../outputs/championship/development/overall-backend-precision-readout-v1/readout.json)
derives the table from the committed complete comparisons. Every case uses its
original observations, coordinate map, generating point and archived attempt
endpoints, including failed attempts. No seed, coordinate or tolerance changed.

| Case | Points | Original CPU reproduced bitwise | Original GPU reproduced bitwise | CPU double-precision repetitions bitwise equal | GPU double-precision repetitions bitwise equal | Double-precision cross-backend screen |
|---|---:|---|---|---|---|---|
| connected | 17 | yes | no | yes | no | pass |
| confounded | 9 | yes | no | yes | no | pass |
| sparse_rookie | 17 | yes | no | yes | no | pass |
| dominant_teammate | 9 | yes | no | yes | no | pass |
| era_boundary | 9 | yes | no | yes | no | pass |
| real_network | 25 | yes | no | yes | no | pass |

All **36** backend/precision evaluations and **6** case comparisons completed,
as recorded by their [evaluation](../outputs/championship/development/overall-backend-precision-evaluate-execution-v1/run.json)
and [comparison](../outputs/championship/development/overall-backend-precision-compare-execution-v1/run.json)
execution records. The [software checks](../outputs/championship/development/overall-backend-precision-checks-v1/stdout.txt)
record **22** passing tests. These are numerical and engineering evidence;
they do not provide scientific acceptance or a driver ranking.

## Full-network differences

The [complete full-network comparison](../outputs/championship/development/overall-backend-precision-v1/compare/cpu32/real_network/result.json)
retains every field and declared comparison, including shapes, norms and the
values at each field's largest discrepancy. Selected summaries follow; the
complete inventory is not restricted to these examples.

| Comparison | Largest absolute gradient difference | Field |
|---|---:|---|
| Original GPU versus repeated single-precision GPU | 0.0048828125 | `age_slope` |
| CPU double-precision repetitions | 0 | all fields identical |
| GPU double-precision repetitions | 1.1368683772161603e-11 | `age_slope` |
| CPU versus GPU in double precision | 8.822098607197404e-11 | `sd_car_season` |
| CPU single versus double precision | 0.21758163685446164 | `overall_lap_nu` |
| Repeated GPU single versus double precision | 0.2477024865507147 | `overall_lap_nu` |

At the `nu` point highlighted in the original failure report, the original CPU
and GPU derivatives were **0.136975542** and **0.229997158**. Double precision
gives **0.16488583937395374** on CPU and **0.1648858393810857** on GPU.
Neither original backend should be treated as a known exact reference.

Fresh-process GPU differences occur in both precisions. This experiment does
not separate compilation-time variation from execution-time variation. OpenXLA
documents both sources of GPU nondeterminism and optional deterministic
compilation settings. Those settings were not enabled or tested here and must
not be described as a demonstrated remedy.
[OpenXLA determinism documentation](https://openxla.org/xla/determinism).

## Consequence for the faster refresh

This result supports testing a separately registered double-precision sampling
implementation with the original probability model and complete diagnostics.
It does not make the failed single-precision backend screen pass. Exact GPU
checkpoint replay on the full model remains unproven: the earlier successful
restart checks used a small Normal fixture. A repeated density calculation
alone does not establish resumed-chain agreement.

The study measured fixed-coordinate evaluation, including compilation and host
transfers, rather than a full sampler. Its timings cannot establish an MCMC
speedup or the user's next-morning refresh target. The double-precision GPU may
also have a different performance and memory profile. Full-model sampling,
bounded archive/diagnostic integration, and the complete intended-runner refresh
benchmark remain outstanding. No production fit, source-data snapshot, public
payload or deployment changed in this experiment.
