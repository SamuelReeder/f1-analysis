# Making race updates practical

This implementation priority follows the user's concern that the elapsed
development work is infeasible after each race. It changes the work order,
not the registered experiment, model, retry ladder or acceptance rules. No faster
refresh has been benchmarked by this review.

## User-selected turnaround

The user selected **"By the next morning"** for updates after a race. The target
is the following morning in **America/Toronto**, covering source availability,
fetching, fitting, validation and the published dashboard. It is an operational
requirement, not a measured capability or a reason to relax scientific gates.
An exact local deadline must be fixed before registering a timed acceptance
benchmark; the user has not specified a clock time.

The current [refresh workflow](../.github/workflows/refresh.yml) checks on Monday
and Tuesday. It does not cover the following local morning for every possible
race completion day. The accepted refresh needs an intake schedule that follows
the event calendar, uses the local deadline across daylight-saving changes, and
leaves room for all stages through publication. A scheduled check is not proof
that a runner starts promptly or that source data is available.

Measure both race-to-publication delay and ready-input-to-publication runtime,
recording source availability separately. The latter cannot establish the former.
Late or incomplete source data must result in an explicit missed-deadline or
waiting-for-data status, with the last accepted snapshot's cutoff visible;
it cannot silently move the deadline or produce a falsely current ranking.
The schedule and deployment remain unchanged while this requirement is prepared.

## What the current code actually does

- [`inference.sample`](../f1rank/overall/inference.py) constructs a fresh sampler,
  initializes from prior medians and runs warmup against the complete joint
  likelihood. It returns posterior draws and sampler diagnostics, but does not
  retain a resumable chain/adaptation/random-generator state.
- [`fitting.run`](../f1rank/overall/fitting.py) accepts a cached fit only for its
  exact registered source, arrays, catalog and settings. Changed data requires a
  new fit. Its attempt timer includes diagnostics; it does not separately measure
  compilation, warmup, retained sampling and diagnostic costs.
- The [CPU profile registration](overall_stable_profile_protocol.json)
  disables persistent compilation caching. The sampler also clears process-local
  JAX caches after each call. Reusing compilation could avoid repeated compiler
  work, but cannot by itself fix expensive sampling or poor mixing.
- [`build_design`](../f1rank/design.py) recomputes circuit factors from all
  eligible training observations. Adding a race can therefore change features
  used by earlier observations. Catalogs are sorted again, and entry, car and
  control dimensions can grow. Array position is not a stable entity identity.
- [`dashboard.refresh_steps`](../f1rank/dashboard.py) and
  [`cachestore.PATHS`](../f1rank/cachestore.py) still implement the legacy refresh,
  not an accepted successor Overall update. There is no measured production
  turnaround for the successor.

The [exhausted earlier profile](overall_complete_profile_results.md) includes
poorly mixing attempts and an interrupted execution. The
[corrected profile](overall_stable_profile_results.md) is a synthetic computational
study using corrected arithmetic. Neither workload is an end-to-end measurement
of updating an accepted ranking after a race. Full model development and
calibration should be versioned work, reused when its registered scope permits;
new-race inference and publication still need their own checks.

## Implementation order

The isolated `overall-refresh-runtime` branch now contains an unselected
checkpoint store and sampling implementation in
[`checkpoints.py`](../f1rank/overall/checkpoints.py) and
[`resumable_sampling.py`](../f1rank/overall/resumable_sampling.py). They save
initialization, partial warmup and complete retained-sampling chunks with random
keys, adaptation, grouped posterior draws and sampler extras. Warmup uses the
original total adaptation schedule and saved iteration; its temporary samples
are not retained as posterior draws. An atomic directory
rename commits each chunk together with its following state; interrupted partial
writes remain separate. Identity checks reject a different target or runtime.
The [initial checkpoint verification](../outputs/championship/development/overall-refresh-checkpoints-checks-v1/stdout.txt)
passes **21** checks, including corruption, interruption, exact comparison with
uninterrupted sampling and recovery in a fresh interpreter. It tests the initial
CPU parallel-chain implementation; it is not a production-size runtime benchmark.
The subsequent [expanded CPU verification](../outputs/championship/development/overall-refresh-vectorized-checks-v1/stdout.txt)
passes **29** checks. These include exact uninterrupted comparisons for parallel
and vectorized chains and fresh-process recovery during warmup and sampling,
including vectorized chains on a single logical CPU device. This path is not selected by
any registration or production runner and supplies no convergence decision.
Restoring JAX's weak array typing uses a private conversion function whose source
is included in the runtime identity; the replay checks must pass under the exact
intended environment before this compatibility choice can be used.
The original CPU replay checks cover both warmup and retained-sampling
interruptions. Integration with bounded diagnostics
and final archives, new-race initialization, and measured runtime remain
outstanding.

**Make the existing target resumable and measure where time goes.** After the
profile audit and checkpoint checks, instrument compilation, warmup, sampling, diagnostics,
archive writing and simulation separately. Preserve external wall-clock and
internal timings separately where they disagree. Save chain state, adaptation,
random keys, completed draws and source/data identities atomically at bounded
checkpoints. Verify resumed versus uninterrupted execution with the complete
declared draw inventory. Resuming the same target must neither repeat accepted
draws nor start an extra attempt. Benchmark persistent compilation caching with
memory limits retained. Changed argument shapes can still require compilation.
[JAX documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html),
[shape-dependent compilation](https://docs.jax.dev/en/latest/201/jit.html).

**Benchmark the existing GPU before committing to a larger algorithm redesign.**
The [device and environment observation](../outputs/championship/development/overall-refresh-runtime-hardware/observation.json)
confirms a local NVIDIA device and a separate environment with CUDA-enabled JAX
packages. The [selected source comparison](../outputs/championship/development/overall-refresh-runtime-hardware/source-comparison.json)
finds matching Python source bytes for the profile's registered runtime files and
the additional checkpoint compatibility dependencies in both environments.
It does not compare compiled backends or establish numerical equivalence.
The registered profile explicitly selects CPU. Package presence and device
memory observations do not demonstrate that this model can execute correctly or
fit in available GPU memory. First compare the registered fixed density/gradient
inventory; then measure a matched complete sampling workload with the same model,
chain/draw inventory and diagnostics. A different chain execution strategy needs
its own implementation evidence. Keep host-memory and CPU limits, bound device
allocation, and account for other device use. This must not revise the completed
profile's recorded backend. Local GPU performance also would not establish the resource
requirements of the current scheduled workflow or authorize a runner migration.

The installed NumPyro implementation falls back from parallel to sequential
chains when the requested chain count exceeds the available devices. A GPU
benchmark must explicitly select and record its chain execution method rather
than inherit that fallback. Vectorized chains are an available experimental
method. The new checkpoint path accepts it explicitly and binds the selected
method and allocator environment into checkpoint identity. Its expanded tests
include multiple chains on a single logical CPU device and reject implicit
fallback or changing methods during a resumed attempt. Those CPU checks pass. The
[local GPU verification](../outputs/championship/development/overall-refresh-gpu-checkpoints-v1/stdout.txt)
then passes the **4** selected vectorized checks, using the same exact-array
comparisons. The remaining **25** CPU checks were deselected, not rerun on the
GPU. The recorded backend is the local NVIDIA device; the tests cover a small
Normal-model fixture, not the production joint model or a matched speed benchmark.
[NumPyro implementation](https://num.pyro.ai/en/stable/_modules/numpyro/infer/mcmc.html).

GPU allocation also needs explicit controls. Disabling preallocation alone does
not establish a device-memory ceiling, and the host systemd memory cap does not
cap device memory. The benchmark must record its allocator, pool fraction,
actual device use and remaining capacity for other applications. The completed
[small GPU check](overall_refresh_gpu_checks.json) explicitly uses a BFC pool
fraction of **0.0625** per process with preallocation disabled, admitting the
parent and restart child together against observed free memory. Its
[execution record](../outputs/championship/development/overall-refresh-gpu-checkpoints-v1/run.json)
preserves before/after device-memory observations; these are not peak device
memory measurements or a cap on total driver/context allocations. Avoid the
slow platform allocator as an unexplained performance benchmark choice; first
verify the installed backend's pool behavior and record the chosen settings
before running either comparison arm.
[JAX GPU allocation documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

The initial GPU-test launch stopped before any workload because `pytest` was
missing. The [environment preparation record](../outputs/championship/development/overall-refresh-gpu-test-environment-v1/run.json)
preserves that failure and the subsequent pinned test-only installation with
dependency resolution disabled. It verifies the numerical packages' existing
files and versions unchanged before and after installation. The original CPU
runtime was not modified. Full-model fixed-coordinate density/gradient checks,
matched complete sampling, final archives and diagnostics integration, and an
intended-runner overnight refresh benchmark remain outstanding.

**Benchmark initialization from the previous accepted joint fit.** Map original
latent variables by stable driver, event, team and circuit identities; reconstruct
the new coordinate transforms and initialize added states coherently. Use distinct
chain states, recompute the new target and perform the newly registered
adaptation procedure. Reuse of previous draws as starting positions does not
replace the new likelihood or convergence checks. Do not use the old posterior
as a prior while also counting the old observations again. A raw saved sampler
state is only a same-target continuation mechanism unless compatibility and
target refresh have been established. NumPyro supports initialization and
continuation, but this project has not implemented the required mapping or
validated its cost. [NumPyro MCMC documentation](https://num.pyro.ai/en/stable/mcmc.html).

**Evaluate a sequential update if full-history sampling remains too expensive.**
Carry a joint distribution over shared parameters and driver/car states forward,
introduce the new event's latent states, update weights and use rejuvenation
moves. Keep correlations and uncertainty in nuisance parameters; do not
independently shuffle marginal draws or freeze historical estimates at point
values. Sequential Monte Carlo provides a possible algorithmic basis, not a
demonstrated speedup for this model.
[Sequential state and parameter inference](https://arxiv.org/abs/1101.1528).

The sequential implementation must address the circuit-feature issue above:
either account for changed historical likelihood factors in the target-density
ratio, or propose and validate a new version with an explicitly stable feature
policy. Treating all old likelihood contributions as unchanged is incorrect.
Historical data corrections, new entrants, new circuits, team transfers and
season/regulation boundaries also require explicit transition and invalidation
rules. Weight concentration or poor mixing must trigger a registered fallback
or an unavailable refresh, not a confident ranking.

Full rebuilds remain necessary for incompatible source/feature changes and as
validation references. Their frequency should follow measured drift and cost,
not an unsupported promise of cheap sequential inference. If neither warm
initialization nor sequential inference is practical, simplify the model in a
new registered version and evaluate the lost predictive and uncertainty
behavior explicitly.

## Evidence required before scheduling it

Register the benchmark inventory, computational rules, comparison tolerances and
runtime budget before executing new performance experiments. Include unchanged
data, an appended event, corrections, catalog changes and interruption/restart.
Measure the full refresh on the intended runner under the user's existing
resource limits. Compare the candidate update to independently converged batch
fits on matched cutoffs, including expected points, pairwise uncertainty,
predictive calibration and difficult sparse/confounded cases. Particle-weight
diagnostics and MCMC diagnostics serve different purposes; one cannot replace
the other.

Demonstrate the user's next-morning target, including intake and publication,
before calling the recurring update production-ready. Prioritize runtime
feasibility before expanding expensive calibration workloads. No measured
turnaround or speedup is established yet. If an update cannot meet its registered
numerical and runtime requirements, retain the last accepted snapshot with its
cutoff and status. When no accepted snapshot exists, keep Overall unavailable.

The original registered sources and settings remain fixed. These
subsequent inference changes require their own registration and regeneration of
affected artifacts. Publication, pushing, merging and cache-release changes
remain subject to the existing authorization requirements.
