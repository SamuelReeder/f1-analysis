# Full-model CPU/GPU numerical comparison

This engineering study precedes a matched sampling benchmark. It performs no
MCMC, changes no model or convergence rule, and cannot authorize publication.
The completed corrected CPU profile and its exact saved-array diagnostic audit
supply immutable inputs. The CPU/GPU checkpoint tests used a small Normal model;
they do not establish full-model numerical agreement or a speedup.

## Fixed input inventory

The [runner](../analysis/overall_backend_compare.py) registers every case in the
corrected profile. For each case it prepares the original generating point and
the first and last saved draw of every chain of every recorded attempt. Failed
attempts remain included. There is no outcome-based point selection or new seed.

Preparation uses the original CPU precision, source/data identities and generator.
It verifies the original observation, parameter, truth and coordinate-map hashes,
then converts the complete stochastic parameter inventory to unconstrained
coordinates once. Both backends consume exactly those serialized observations
and coordinates. The GPU does not regenerate observations, redraw latents or
repeat the coordinate transformation. The original complete retained-site
inventory also remains required.

The frozen registration binds the current source/data inventory, original
committed evidence and selected numerical library sources in both environments.
Checkpoint modules added on the isolated branch do not replace any original
model source. Each stage verifies its source and input identities before and
after work; existing outputs and numeric archives cannot be overwritten.

## Comparison rules fixed before execution

Evaluate potential energy, its complete gradient and every retained deterministic
or stochastic model site at every declared point. Check all coordinates, shapes,
dtypes and finite values. Integer and Boolean outputs must agree exactly.

For potential energy and gradient coordinates, use the absolute and relative
budgets already recorded as `density_tolerance` in the corrected preflight.
For retained floating-point state, use its existing `state_tolerance`. The CPU
value is the reference magnitude in the absolute-plus-relative expression.
Applying the density budget to derivatives is a prospective engineering screen,
not a rigorous accumulated-roundoff bound or evidence of posterior equivalence.
The complete per-field report records every failing coordinate and the maximum
absolute and budget-normalized discrepancy. No averaging can conceal a failure.

A failed screen is preserved and blocks the proposed matched sampling benchmark
until its cause is understood. Do not relax its budget, remove points or select
another seed after inspecting a result. Any justified numerical implementation
change needs a separate registration and preserved original evidence. Passing
this screen still requires a matched complete sampling comparison, existing
convergence checks, posterior/functional agreement and scientific acceptance.

## Resources and timing

All stages use the existing capped analysis scope, run sequentially and refuse
other active heavy work. The exact CPU/GPU environments and resource settings are
fixed in the runner before registration. GPU admission compares its requested
BFC pool against currently free device memory; host cgroup limits do not cap GPU
memory, and the pool setting does not cap all driver or context allocations.
An admission failure is a resource failure, not a numerical result. Preserve it
and wait for capacity rather than silently changing the budget.

Each evaluation records lowering, compilation, synchronized execution, transfer
and archive writing separately. External process timing and peak host RSS remain
separate measurements. These fixed-point timings omit MCMC warmup, sampling,
posterior diagnostics and publication. They cannot establish an overnight
refresh, predict its duration or justify a claimed full-model speedup.

Preparation artifacts must be committed before either backend evaluation; both
evaluations must be committed before comparison. Numeric arrays stay in the
ignored championship cache, with their hashes and complete inventories in the
committed reports. Original fit archives remain untouched. All original cases
must have a final stage status, including failures or unavailable dependent stages.
