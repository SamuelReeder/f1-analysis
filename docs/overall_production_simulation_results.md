# Candidate simulation integration results

The [registered production candidate controller](overall_production_simulation.md)
now connects the fit receipt and bounded posterior reader to paired simulation,
the complete summary, grouped diagnostic moments and export components. A
candidate reader rechecks those artifacts and their registered inputs. Both
paths keep scientific acceptance and publication false. The public dashboard
producer has not yet been replaced.

The [verification record](../outputs/championship/development/overall-production-simulation-checks-v1/verification.json)
records **277 passing focused checks** against the committed corrected source.
It includes the original paired-simulation, points, summary, archive,
development-controller and publication-validator checks. Migrated numerical
definitions match their original implementations; changed artifact roles and
source locations are explicit metadata changes.

The first run recorded **2 failures and 268 passes**. The new candidate reader
correctly rejected a protocol copy whose formatting changed its registered byte
identity. The writer now preserves the original bytes. The raw failure and its
source commit remain in the verification record. Additional read checks
recompute the complete block inventory and detect dependencies changed after
their initial check. No model, sampler setting or precision threshold changed.

End-to-end tests use invented posterior archives, registered-format fit receipts
and a fake sampler. They run the actual paired simulator through the full declared
small workload, preserve failed conditional precision, write complete diagnostic
artifacts and read them back. Tests also exercise the real-data registration
role using those same invented fixtures; this is not a real-data fit or driver
score. Failed precision produces no export components. The successful component
writer is checked separately with invented precise statistics. The complete
passing-precision workload still needs its own resource and numerical validation.

The final run reports **26.13 seconds** through pytest; its external resource
log separately reports **0:23.92** and **340,404 KiB** peak resident memory. Those
clocks are not interchangeable. The active scope record confirms the configured
local **3G** memory and **200%** CPU limits. The initial scope inspection happened
after exit, so its inactive properties are not presented as active-limit evidence.
These are local software-check measurements, not intended-runner measurements.

The failed full-network inference remains failed. Scientific protocol completion,
recovery/calibration and independent prediction evidence, functional numerical
precision, unresolved reference assumptions, authenticated acceptance, canonical
dashboard publication, complete refresh integration and approved rollout remain
required by the [full plan](overall_model_plan.md). No new scientific run or
public payload was produced by this integration.
