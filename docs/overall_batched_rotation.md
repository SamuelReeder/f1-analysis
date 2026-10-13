# Batched evaluation of the existing control rotation

The registered control-coordinate map evaluates each race's small matrix product
and writes each resulting coefficient into its original site separately. Prepare
a batched evaluation of that exact map for subsequent source integration. The
running complete-design profile keeps its original implementation and registration.
This preparation does not authorize replacing or restarting that run.

Keep every original block, eigenbasis, sign convention, scale, latent index and
output coordinate. Group blocks by matching matrix dimensions and numeric dtypes. Evaluate their
matrix-vector products together, multiply by their original scales, and gather
the resulting coefficients into the original site shapes. Preserve the complete
original map fingerprint and log absolute determinant. No observation, generating
truth, fitted draw or convergence result determines a group or coefficient.

This changes array operations, not coordinates, priors, likelihoods, sampling
settings, diagnostics or retry rules. Do not recompute an eigenbasis, discard a
direction, approximate a matrix, change precision or choose a new geometry.
Reject incomplete or overlapping assignments and invalid shapes. Own immutable
copies of the prepared numeric layout so later mutation of the original Python
arrays cannot silently change an already prepared map.

Before any use in a fit, compare original and batched reconstruction and their
derivatives on complete heterogeneous block layouts, including multidimensional
sites and nontrivial block ordering. Compare the full original and batched
probability calculations and gradients on every fixed preflight design, using
the same original observations and generating-parameter fingerprints. Preserve
any discrepancy; the existing equivalence tolerances must not be relaxed.

Measure compilation and repeated complete potential/gradient evaluations for
both implementations on that full inventory. Freeze that measurement's workload
before execution; do not use a reduced network or timed matrix kernel alone as
evidence of a full-model speed improvement. Retain both results and all resource
limits. These checks must wait until the active fit releases its resource budget.

Only successful complete equivalence and resource checks can support integrating
this implementation in a subsequent frozen runner. A faster map alone would not
prove a proportional fitting speedup, convergence, recovery, ranking precision
or production readiness. Earlier runs and their source identities remain intact.

## Prepared complete-model audit

The [audit runner](../analysis/overall_batched_rotation_audit.py) prepares a fixed
workload for after the active profile finishes. It remains unexecuted. Run its
focused software checks before freezing and committing a measurement registration.
The registration binds both implementations and every original preflight design,
generation seed, data/truth fingerprint and model source.

Each implementation runs in a fresh capped process for each complete design.
Both use the original map's inverse and the original generating parameters, so
the supplied coordinates are identical. Save every traced state, every sample
site's probability factor, the joint density, the unconstrained potential and its
complete gradient. A separate capped comparison checks the entire inventories,
numeric dtypes and shapes, exact unconstrained inputs and every numeric coordinate.
Keep the existing density and state tolerances; the gradient comparison uses
the state tolerance. A discrepancy remains a failure, with no tolerance change.

The runner records preparation and compilation with the first synchronized full
potential/gradient call separately. It then measures **8** synchronized complete
evaluations at that same point, as fixed by `REPEATS` in the committed runner.
Both arms use the same explicit CPU environment with persistent compilation
caching disabled. Each process records its peak resident memory and an external
resource log under the existing scope policy. Do not replace the full network
with a smaller timing fixture. Preserve every required case and both methods
after failures; saved comparison arrays live in the ignored cache, while their
fingerprints and full comparison reports belong to the reviewable outputs.

This is a local, single-point calculation benchmark. It cannot establish typical
posterior gradient cost, NUTS trajectory length or an equivalent fitting speedup.
It supplies no additional sampling, recovery, ranking or publication evidence.
