# Overall production inference integration

The coordinate-migration evidence below describes the original integration.
The current implementation also includes the
[verified equivalent Student-t arithmetic correction](overall_studentt_production_results.md),
with a distinct inference-method identity and its own complete migration evidence.
The historical profile failures remain failures.

The production fitting engine can now select the checked conditional weekend
coordinates and batched lap-control map. Previously those implementations lived
only in research runners; the default production engine still called the original
coordinate model and allocated whole-site diagnostic temporaries.

The implementation is copied into the hashed `f1rank/overall/` package:

- [control_rotation.py](../f1rank/overall/control_rotation.py) includes the existing
  design-only orthogonal map and its complete prior directions.
- [batched_rotation.py](../f1rank/overall/batched_rotation.py) groups the same map
  evaluations without changing its fingerprint.
- [weekend_coordinates.py](../f1rank/overall/weekend_coordinates.py) reconstructs
  the original weekend effects under the same joint prior and likelihood.
- [bounded_diagnostics.py](../f1rank/overall/bounded_diagnostics.py) applies the
  existing numerical criteria to every coordinate with bounded temporary arrays.
- [inference.py](../f1rank/overall/inference.py) composes these implementations,
  retains every posterior field and requested sampler extra, and reports every
  current-driver contrast.

These modules import no research implementation at runtime. Their dependencies
are covered by the existing recursive source snapshot. The original research
files, registrations and failed experiments remain unchanged in their frozen
checkout and historical commits. Moving code does not confer acceptance on them.

## Registration and evidence

A new registration must explicitly select the production coordinate method and
bind the complete current model/data source key. Historical registrations retain
their original implementation. The production registration cannot supply a
different sampler callback. The fixed warmup/draw ladder, seed rules, target
acceptance settings and diagnostic thresholds remain those already defined in
[fitting.py](../f1rank/overall/fitting.py) and
[diagnostics.py](../f1rank/overall/diagnostics.py). Driver-contrast reports retain
their separate role; this integration does not change the retry gate.

For each completed new attempt, the runner saves the grouped divergence flags, tree-step
counts and acceptance probabilities beside the posterior or failed draws. Their
hashes and the complete posterior site inventory enter the attempt record.
Opening a successful production posterior verifies every attempt's sampler
archive and recounts its recorded diagnostics, both before and after the consumer
uses the posterior. Missing, changed or inconsistent sampler evidence is rejected.
Historical archives lack these fields and are not retroactively described as
independently recounted. Exhausted runs remain terminal and cannot be resumed as
another retry.

## Verification and limits

The [committed verification record](../outputs/championship/development/overall-production-inference-checks-v1/verification.json)
preserves the initial **127 passing checks**, followed by **177 passing checks**
against the committed implementation and its archive/recovery/simulation
consumers. These runs overlap and must not be added together as unique tests.
The intermediate run refused uncommitted fitting source at the simulation
controller's existing source check. Its failed output is preserved; committing
the tested implementation resolved the refusal without changing that check.

The focused checks compare every migrated definition with its research source,
and compare complete traces, probability terms and gradients for the declared
small synthetic designs. They exercise sampler dispatch with a test double,
preserve each failed attempt, check new registration/source requirements, and
reject changed sampler evidence. These software checks do not perform MCMC.
The real-network calculation evidence remains the separately recorded
[batching audit](overall_batched_rotation_results.md), under its original source
identity; no new full-size fit or runtime claim follows from this migration.

The [failed complete profile](overall_complete_profile_results.md) remains failed.
The canonical acceptance controller, complete recovery/calibration evidence,
independent prediction evidence, actual canonical producer, full refresh
integration and rollout remain required by the [plan](overall_model_plan.md).
This change grants no publication authority. Finish the fitting-source integration
before freezing new scientific runs, and regenerate affected output identities
before publication.
