# Legacy baseline before qualifying: development definition

This definition precedes implementation and real-event evaluation of the
baseline's unknown-grid calculation. It extends the
[legacy forecast interface](overall_baselines.md), without changing either
fitted model or its priors. It is not a complete comparison registration or
permission to fit, score real events or publish a ranking.

## Conditional probability construction

Let `theta` be the separately fitted qualifying-only parameters, `beta` the
joint race-coefficient vector, and `F` the fixed legacy qualifying features.
Training features are medians of paired persistent-state sums from the
qualifying-only posterior. The race fit conditions on those fixed features.
The specified modular approximation is the product of the qualifying posterior
and the race-coefficient posterior conditional on the fixed training features.
It is not the successor's joint posterior, and is not a claim that the original
latent quantities are independent in a fully joint model.

For a future event, compute fixed future driver/car features using the complete
qualifying-only forecast draw set and the existing median policy. Generate each
qualifying trajectory from one paired qualifying parameter/state draw, including
shared team/segment effects, session scales and Student-t noise. Keep dependence
within that draw. The race baseline then integrates its conditional order
probability over every qualifying-grid trajectory and every joint race-coefficient
draw. Evaluate the Cartesian product in bounded blocks; do not pair fits by draw
index, truncate the larger fit, or average log probabilities.

Repeated identical grid trajectories may be aggregated with their integer
counts. This is exact for the empirical distribution and cannot drop a rare
grid. Reordering either posterior sample inventory cannot alter the integrated
probability. All coefficient coordinates remain paired within their fitted draw.
The same qualifying grid distribution supplies the secondary grid-only race
baseline; that race mapping uses only its grid coefficient and no driver/car
features. Its pre-qualifying forecast therefore still requires a grid model.

The result is a single-event marginal conditional order probability. Summing
such event scores is not the joint probability of a season with shared states.
The supplied running panel is retained explicitly. This baseline has no panel
probability model, so it does not reweight qualifying trajectories using the
later panel and supplies no retirement prediction. This conditional definition
and the common classification policy must be covered by the final comparison
registration; no historical exclusion rule is inherited automatically.

## Explicit development assumptions and remaining work

The initial kernel requires the existing explicit scenario
`qualifying_order_no_penalties`: every entrant sets a time, qualifies and starts
in simulated qualifying order. It uses the existing segment generator and an
explicit advancing-field schedule. This is a conditional development scenario,
not a completed official-grid policy. Penalties, eligibility, no-time sessions,
qualifying weather and differing formats remain requirements for the end-to-end
weekend evaluation. Real grids are rejected at the pre-qualifying information
boundary rather than inserted into this calculation.

One trajectory per qualifying draw is a development approximation only. The
grid simulation seed is explicit and uses a separate legacy observation stream;
batch size and field order must not change it. The complete forecast-median and
grid replication/precision policy must be registered before predictive scoring.
Integrating the full empirical product does not eliminate posterior or grid
Monte Carlo error, prove convergence, or establish calibration.

Synthetic checks will cover the fitted qualifying observation process,
batch/field invariance, hand-computable grid/coefficient mixtures, unequal sample
counts, sample permutations, normalized order probabilities, empty/singleton
panels and complete input rejection. They do not substitute for accepted fits,
as-of source verification, complete chronological comparisons or independent
future evidence.
