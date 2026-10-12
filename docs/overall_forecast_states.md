# Joint states at a forecast origin

[`overall_forecast_states.py`](../analysis/overall_forecast_states.py) is a
development implementation of the existing model's state transitions. It
reconstructs uncentred training states and propagates them through an announced
future field path. It reads no future observations and performs no fitting.
It does not change the likelihood, priors or pending numerical experiment.

## Preserving the joint uncertainty

Every parameter keeps its common chain and draw indices. The function requires
the underlying driver, car and team latent coordinates; reported centred
ratings cannot substitute for them. The supplied training-array identity must
match the catalogue's actual arrays. Required shapes, finite values and parameter
supports are checked. These checks do not establish convergence or verify a fit's
source: a production caller must use the fitter's checked posterior loader.

Driver qualifying walks are reconstructed from their levels and innovations,
then combined with the existing experience and age terms. Race ability retains
the same draw's qualifying loading and race residual walk. Future innovations
use the model's within-season event gaps and between-season season gaps. Drivers
absent at an intervening event retain the appropriate gap at their next entry.
New drivers draw levels from the fitted population scales. They remain in the
forecast field with uncertainty.

Team effects retain their driver/team/era identity. New stints and eras draw
from the existing population distributions. The car walk uses the existing
heavy-tailed within-season steps, seasonal carryover and regulation-reset rules.
It follows the model's sequence of team entries; it does not invent unobserved
car developments for events a team did not enter. New team-season track loadings
are drawn from their population distribution.

The training car-season residuals use zero-sum Normal contrasts. Before adding
a previously unseen team, the implementation restores the unobserved common
season mean. For prior scale `s` and `n` existing teams, that mean has standard
deviation `s / sqrt(n)` and is independent of the contrasts. Existing team
residuals plus this common mean have the original independent Normal prior.
A new team's residual can then be drawn from that same prior. Ignoring the
integrated-out mean would understate uncertainty relative to the new team.
All car effects are subsequently centred over the forecast's car field.

Fresh qualifying and race-day deviations preserve the learned shared driver
weekend component. Teammates share car effects; qualifying car-event variation
and the race car-day residual follow their distinct roles in the current model.
Each random stream attaches to its state identity and the supplied seed. Field
row order does not choose who gets a random draw, and extending the announced
path does not change earlier states.

## Information and implementation boundaries

The caller supplies an outcome-free snapshot for every intervening event at one
information origin. Each snapshot must carry the cumulative calendar and driver
participation counts; omitted fields, inconsistent counts and changing metadata
are rejected. This is conditional on the declared calendar and lineups. Source
verification and uncertainty about schedule or lineup changes remain separate
requirements. The catalogue's cross-season completeness limitation still applies.
Checking training event dates alone also cannot establish when archived results
became available; source verification must establish that information boundary.

The circuit loading uses the frozen training factor. For an unseen circuit it
inherits the existing qualifying design's no-adjustment convention. This does
not estimate circuit-factor uncertainty or supply the future outcome model's
circuit coefficients. Those limitations must be included in the final predictive
specification, alongside future weather, neutralisation and operations.

The result contains latent qualifying/race components and their observed-state
totals for the announced cars and teams. Those totals are for predictive work;
they are not the equal-machinery driver ranking. Outcome generation, qualifying
segment effects, context coefficients, condition integration, calibration and
baseline predictors remain to be connected and validated. The canonical
counterfactual also still needs its accepted producer and common scenario.

No fit is loaded or certified here. Every result leaves convergence, source
verification, independent evidence and publication flags false. It records the
training arrays, consumed parameter arrays, implementation hash and supplied
seed. The [bounded archive reader](overall_posterior_archive.md) now shares the
fitter's source, receipt and checksum checks. Accepted-fit integration and
intended-runner memory/runtime remain to be verified; the resource checks use
synthetic parameter arrays.

## Bounded processing

`iter_forecast_states` processes contiguous draw batches while retaining the
common chain axis. It validates and hashes all consumed parameters in bounded
chunks before yielding any forecast. Nonfinite or out-of-support parameters in
the final draw therefore fail before any batch is yielded. The digest remains
compatible with the original array identity, including for strided inputs and
read-only memory maps.

Only the latest driver and car walks are reconstructed in each batch. The
forecast no longer needs an array of every historical state for every draw.
The separate `reconstruct_history` diagnostic helper still materializes that
history and should only receive deliberately small inputs.

Each batch replays the original identity-keyed random stream and takes its draw
slice. This avoids treating a variable-length Normal or Student-t random-number
algorithm as though raw generator steps corresponded to returned samples. The
working stream contains only the chain and draw axes, without a historical-state
axis. Batching preserves the innovations and the model; it does not reduce the
draw count. Small floating-point differences from matrix arithmetic are covered
by the numerical equivalence checks.

The iterator labels every draw range and the full posterior shape. Consumers
must account for all ranges and release processed outputs to keep output memory
bounded. A later arithmetic or processing failure must invalidate the whole
forecast; successfully consuming an initial batch is insufficient.
`forecast_states` remains a convenience wrapper that collects the whole
forecast; its retained output memory grows with the forecast horizon. Neither
interface itself verifies or loads a compressed fitted archive. Callers can use
the mapped archive context, supplying the expected run and training-array
identities and completing that context successfully before committing an output.
The canonical producer still needs this integration and its acceptance checks.

[`overall_forecast_resources.py`](../analysis/overall_forecast_resources.py)
measures a synthetic workload at the committed training-data shape and the
largest draw count in the existing fixed retry ladder. It requires the user's
memory and CPU caps, uses temporary uncompressed memory maps, checks complete
draw coverage, and retains no driver-state table. It runs no fit or predictive
evaluation. Local resource evidence cannot establish readiness on the intended
CI runner or the memory required by compressed-archive loading.

The [local resource record](../outputs/championship/development/overall-forecast-batch-checks/resources.json)
shows complete processing of that workload under the enforced scope limits. Its
peak resident memory covers the whole process, including data construction and
workload creation. The old-history allocation figure is a calculated lower bound,
not a measured baseline peak or a speed comparison. The temporary workload maps
were removed after the check; only its identity, coverage and measurements remain.

## Verification

[`test_overall_forecast_states.py`](../tests/test_overall_forecast_states.py)
compares reconstructed history and future components against the existing joint
model conditioned on identical innovations. The cases cover same-season updates,
a regulation reset and an unseen team. Separate checks exercise field centring,
newcomer uncertainty, the car-season mean, missing participation, preserved joint
draw indices, field permutations, prefix consistency and invalid input rejection.

The [verification record](../outputs/championship/development/overall-forecast-state-checks/verification.json)
identifies the tested source and full-suite report. It also preserves the initial
distribution-property assertion failure and the subsequent passing focused run.

The subsequent [batch verification](../outputs/championship/development/overall-forecast-batch-checks/verification.json)
records bounded validation, original-stream equivalence, terminal reconstruction,
complete draw coverage and the full local suite on committed source. It links the
resource workload separately from scientific acceptance.

The subsequent [archive verification](../outputs/championship/development/overall-archive-checks/verification.json)
exercises compressed loading, validation of unused sites, mutation rejection and
cleanup. Its local synthetic format fixture produces the same state checksum as
the uncompressed workload. Its convergence field is a test fixture, not fit
evidence; no resource check establishes scientific acceptance or CI readiness.

These are probability-interface and software checks, not parameter recovery,
simulation-based calibration or evidence that the model predicts real races well.
The original numerical failure and the separate pending sampling proposal remain
unchanged. Acceptance still follows the full [execution ledger](overall_execution.md).
