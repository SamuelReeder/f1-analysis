# Unknown race conditions

The [development specification](overall_condition_policy.json) is fixed before
inspecting its real-history condition counts or evaluating its forecasts. It
defines a concrete historical approximation, not an accepted predictive model
or an amendment to the failed numerical study. Fitted likelihoods, sampler
settings, original failures and acceptance rules remain unchanged.

Wetness and neutralisation are one categorical state, with the existing shared
design's definitions. Each eligible event contributes once, irrespective of
field size, clean-lap count or number of running finishers. The bounded risk
inventory supplies contexts, including wet events and events without running
finishers. Unknown context and events without eligible risk observations remain
listed. Missing evidence never becomes dry/green evidence. The existing outcome
window determines history; future observations cannot enter its counts.

For each circuit, the specified symmetric Dirichlet prior is updated by its
event counts. This is conjugacy: multiplying the prior density by categorical
observations adds category counts to its parameters, as derived in the
[University of Washington multinomial lecture](https://faculty.washington.edu/yenchic/20A_stat512/Lec7_Multinomial.html).
A circuit without eligible history retains that prior. There is no hidden
fallback to dry conditions, fitted shrinkage across circuits or data-selected
prior strength. This approximation does not represent seasonality, a weather
forecast or changes in safety-car deployment over time.

Conditional on the observed historical contexts, this frequency model and the
existing race-parameter model factor under their independent priors. Forecast
trajectories retain a sampled probability vector per circuit on the same
chain/draw axes as the race parameters. Repeated visits share that vector;
events draw distinct condition states, each shared by their entire field.
Wetness and neutralisation are drawn jointly. Replacing shared probabilities
with independent posterior-mean draws would lose dependence across visits.

For a single event's predictive density, frequency uncertainty can instead be
integrated analytically with its posterior mean. If `w[k]` is that mean,
`a[d,k]` the probability of a running panel and `b[d,k]` its conditional order
probability, the predictive order probability conditional on that panel is
`sum(d,k, w[k] * a[d,k] * b[d,k]) / sum(d,k, w[k] * a[d,k])`.
Integrate the joint mass before normalizing: the observed panel changes both
condition and parameter weights. This formula does not supply a season's joint
condition probability by multiplying separately integrated event means.

Development checks must cover event deduplication, joint state counts, wet and
no-finisher contexts, missing evidence, cutoff safety, unseen circuits,
analytical posterior moments and conditional scoring. Full weekend checks must
preserve paired draws, event/driver ordering, batch boundaries and persistent
circuit uncertainty. These checks do not establish calibration, historical
source availability or an accepted equal-car reference calendar. Exposure,
official grids, classification, qualifying weather and the final evaluation
inventory remain separate open requirements.
