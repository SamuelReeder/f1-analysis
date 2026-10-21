# Deterministic GPU numerical candidate

The preceding [precision investigation](overall_backend_precision_results.md)
found close double-precision CPU/GPU agreement but failed exact GPU repetition.
This new prospective computational candidate enables
`--xla_gpu_exclude_nondeterministic_ops=true` on the double-precision GPU arms.
Keep the probability model, observations, coordinate map, complete fixed-point
inventory, numerical budgets and resource limits unchanged. Preserve every
earlier failure. This is an implementation check, not a fit or scientific gate.

The setting is documented to select deterministic GPU operations and disable
live autotuning; it may slow execution or reject unsupported operations. Its
effect on this model is untested before this registration.
[OpenXLA documentation](https://openxla.org/xla/determinism).

The [runner](../analysis/backend_determinism/__init__.py) binds the complete
preceding evidence and reuses its checked lifting, model evaluation, full-field
comparison and resource admission functions. Freeze its definition before
execution. Every original case and point is required, including the endpoints
of failed sampling attempts. Evaluate each declared GPU arm in a fresh process;
preserve each terminal status, including compilation or resource failures.
Commit raw evaluations before comparing them.

Require bitwise equality of every retained value, gradient and model-state
output across the fresh GPU repetitions. Compare each repetition to the already
committed double-precision CPU reference using the unchanged original budgets.
Missing or failed cases prevent a complete candidate pass. Neither tolerance
changes nor dropping a difficult point can rescue a failure. A pass authorizes
preparation of the next engineering experiment, not production sampling or
publication. In particular, complete-model resumed versus uninterrupted chain
behavior and measured fit cost remain to be checked separately.

The original screen and precision experiment remain immutable. No new result
relabels the failed single-precision backend comparison. CPU reference values
are numerical comparison values, not a claim of exact mathematical truth.
Compilation, synchronized execution, transfer and external timings stay separate;
fixed-coordinate timings cannot establish next-morning publication.
