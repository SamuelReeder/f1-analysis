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
