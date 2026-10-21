# Complete-design checkpoint replay

This prospective engineering experiment tests the complete Overall model under
the previously verified double-precision deterministic GPU implementation.
It preserves every original synthetic design and the stored observations and
coordinate maps. It does not generate new data or select cases based on results.
The existing failed comparisons and fits remain immutable.

The [runner](../analysis/checkpoint_model_replay/__init__.py) declares the exact
short engineering workload, chunk size, chain method, seed, resource limits,
phase time limit and exit statuses. Freeze and commit these settings, their
sources and preceding verification evidence before execution. The short workload
exercises initialization, adaptation, sampling and finalization; it is not a
shortened production fit and cannot pass a convergence, recovery or predictive
gate. Production fitting settings and the original retry ladder are unchanged.

For every design, run the checkpointed implementation uninterrupted and in a
separate execution interrupted after its first complete warmup chunk. Resume
that execution in a fresh process, stop after its first retained chunk, and
resume again in another fresh process. Keep complete adaptation state and random
keys; restarting must not reinitialize an attempt or change its target. Each
continuation requires its exact preceding result and checkpoint bytes. New
initial executions may not reuse existing caches.

Retain all model-state fields from the original complete collector at the new
execution precision, every declared sampler extra and the complete final HMC
state. Stream completed draws into numeric archives with the verified writer.
Commit raw execution results before comparison. Require bitwise equality of
every posterior value, sampler extra and final sampler-state leaf between the
uninterrupted and resumed executions, together with identical state containers,
runtime identity and collector inventory. Preserve every mismatch; there is no
tolerance adjustment or replacement seed. This checks the checkpointed procedure
against itself across interruptions. It does not claim identical full-model
trajectories to a separately compiled monolithic NumPyro invocation.

All declared designs and phases must have a terminal outcome, including crashes,
timeouts and dependent phases blocked by an earlier failure. A planned stop has
its own exit status and is not a convergence failure. A timed-out process group
is terminated explicitly before another phase runs. There are no automatic
numerical retries in this engineering experiment. A failed case prevents a
complete replay pass; other independent cases remain in the inventory.

Run sequentially within the registered host cap and existing reduced CPU slice.
Use the previously tested single-process GPU pool and admission rule; the
orchestrator does not initialize JAX or reserve a device pool. Record compilation,
synchronized execution, transfer/checkpoint and archive costs separately where
the implementation supports them. Initialization includes its compiler work.
Keep external process measurements separate. These short-workload timings do
not establish full-fit runtime, mixing, calibration or next-morning publication.

A successful complete replay is a prerequisite for the complete sampling
benchmark and bounded diagnostics integration. It cannot accept a real-data fit,
publish a driver ranking, modify the production refresh or authorize deployment.
