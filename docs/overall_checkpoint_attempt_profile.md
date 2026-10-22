# Complete synthetic fitting with staged checkpoint attempts

This prospective computational registration selects the verified deterministic,
double-precision GPU checkpoint implementation for every original synthetic
design, including the full real-network observation design. It retains the
stored observations, their original control basis and the complete retained
collector. Float observations are lifted from their stored original precision;
they are not regenerated under a new numerical environment. The model, seed
schedule, full fitting settings, numerical gate and retry ladder are inherited
from the original stable profile. The new registration binds the current complete source snapshot, including the
checkpoint modules added after the original profile, while requiring every input
to match the already verified backend/replay source inventory. It separately
records the original source key. The new registration preserves that profile's
failures and selects a different verified computational implementation; it does
not retrospectively change an earlier registration or its decision.

Freeze and commit the registration and complete worker/controller sources before
starting. The complete-model engineering replay and the isolated-attempt fixture
checks must already be committed and unchanged. A service controller runs every
registered case sequentially. Each numerical attempt runs the checkpointed GPU
sampler and archive finalizer, then exits that interpreter before starting CPU
diagnostics against the exact sealed artifact. A short or incomplete attempt
cannot pass. No public ranking, calibration claim or deployment is authorized by
this computational profile, even if every numerical gate passes.

The profile uses the already tested checkpoint chunk size and chain method.
Every new attempt starts with its own registered seed, settings and directory;
a same-target operating-system interruption resumes the same attempt instead.
Recovery requires authoritative terminal state of the previous service and the
original process identities, immutable request, checkpoints and result checks.
The controller cannot overwrite a terminal failed execution. Only complete failed
numerical diagnostics advance the existing ladder. After exhaustion, retain the
failed case, process the other independent cases and report the full inventory.
There is no retuning or additional seed search.

Keep the existing shared CPU quota, low priority and sequential heavy-work rule.
Each worker inherits the same bounded service. Device admission uses the earlier
single-process pool policy. At checkpoint boundaries, preserve allocator memory
statistics where the backend provides them; these include the allocator's own
peak counters when available, not a measured global device-memory ceiling.
Record cgroup memory peak/current/max/events and CPU counters while the service
still exists, separately from process maximum RSS and stage timings. A cumulative
service peak includes the controller, worker, and cgroup-charged cache; it is not
an isolated per-stage resident-memory measurement. Missing observations remain
missing. Separate internal compiler/execution/archive/diagnostic timing from the
external service elapsed time.

This is a full synthetic computational profile. Its result cannot establish a
new-race end-to-end deadline: it does not include source availability, intake,
calibration, predictive evaluation, simulation or publication. Those requirements
remain in `overall_model_plan.md` and `overall_refresh_runtime.md`. The user's
next-morning target remains unverified until the complete intended-runner refresh
is registered, measured and accepted.

The initial frozen selection's [preflight](../outputs/championship/development/overall-checkpoint-fit-preflight-v1/run.json)
failed before device initialization or sampling: the minimal worker environment
omitted its user-service-manager connection. A new frozen selection explicitly
binds the local runtime directory and bus address needed to verify resource
limits. The initial registration and failed preflight remain preserved. This
environment repair changes no model, observations, fitting settings or gate.
