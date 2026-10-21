# Checkpoint timing and bounded final archives

The isolated runtime branch now has an instrumented checkpoint sampler in
[`analysis/checkpoint_sampling`](../analysis/checkpoint_sampling/__init__.py)
and a bounded archive writer in
[`analysis/checkpoint_archive`](../analysis/checkpoint_archive/__init__.py).
Both remain unselected development implementations. Existing production fitting
sources, registered experiments and public outputs are unchanged.

The sampler uses the installed NumPyro collection function and preserves its
chain state, adaptation schedule, random keys and complete retained inventory.
It lowers and compiles each required chunk shape explicitly, then records
synchronized execution, transfer and checkpoint-writing stages. The transfer
stage includes conversion to host arrays and sampler-extra validation.
Initialization has a separate combined measurement that includes its compiler
work; the instrumentation does not claim to separate that initial cost.
The private collection API and checkpoint implementation are bound into the
saved runtime identity. Reusing an executable within a process is recorded;
persistent compilation caching remains disabled.

The archive writer reads one site's checkpoint chunk at a time and streams
ordinary numeric archives in chain/draw order. Its memory requirement depends
on the largest site chunk, not on retaining the entire posterior simultaneously.
It checks the complete externally supplied site inventory and requires all
declared warmup and retained draws. It preserves empty sites, array types and
signed zeros, including input arrays stored in Fortran order. A directory rename
seals the output only after source checks and durable archive writes. Failed or
interrupted partial directories cannot be consumed as completed artifacts.
Neither completion nor finalization assigns convergence or publication status.

## Completed engineering checks

The [CPU execution record](../outputs/championship/development/overall-checkpoint-engineering-checks-v1/run.json)
and [test output](../outputs/championship/development/overall-checkpoint-engineering-checks-v1/stdout.txt)
record **44 passing checks** with double precision enabled. They include the
existing checkpoint corruption and identity checks, complete archive comparison,
chain ordering, interrupted sealing, staged collection versus uninterrupted
NumPyro, and warmup/sampling recovery in fresh processes. Recorded sources were
unchanged during execution.

The [GPU execution record](../outputs/championship/development/overall-checkpoint-engineering-gpu-checks-v1/run.json)
and [test output](../outputs/championship/development/overall-checkpoint-engineering-gpu-checks-v1/stdout.txt)
record **4 passing selected sampling/restart checks** under double precision and
the deterministic-operation setting. These compare all fixture samples and
sampler extras exactly. They cover single-chain collection, vectorized collection
and fresh-process interruptions in warmup and retained sampling. They do not
rerun the entire CPU suite on the GPU.

Both executions ran under the recorded **3G** host cap and **200%** CPU quota
inside the existing shared slice. The GPU test reserved a BFC pool fraction of
**0.0625** per process, admitting the parent and restart child together against
half the observed free device memory. The recorded before/after observations
do not measure peak device memory or impose a global GPU memory cap.

These are small Normal-model and synthetic archive fixtures. They establish
engineering behavior for those fixtures, not complete Overall-model chain
replay, production-scale archive memory or fitting speed. The next engineering
check must preserve the complete model's source/data identity and full site
inventory through uninterrupted and interrupted execution. Complete fitting,
bounded diagnostic integration and the user's next-morning refresh benchmark
remain outstanding. No faster recurring update is deployed or demonstrated yet.
