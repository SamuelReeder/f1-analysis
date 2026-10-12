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
seed. Full-archive memory/runtime and efficient batching remain to be measured
and addressed before production use; the current checks use synthetic arrays.

## Verification

[`test_overall_forecast_states.py`](../tests/test_overall_forecast_states.py)
compares reconstructed history and future components against the existing joint
model conditioned on identical innovations. The cases cover same-season updates,
a regulation reset and an unseen team. Separate checks exercise field centring,
newcomer uncertainty, the car-season mean, missing participation, preserved joint
draw indices, field permutations, prefix consistency and invalid input rejection.

These are probability-interface and software checks, not parameter recovery,
simulation-based calibration or evidence that the model predicts real races well.
The original numerical failure and the separate pending sampling proposal remain
unchanged. Acceptance still follows the full [execution ledger](overall_execution.md).
