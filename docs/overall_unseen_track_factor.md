# Qualifying track uncertainty at an unseen circuit

The [development policy](overall_track_factor_policy.json) specifies a population
distribution before evaluating forecasts. This is a new extrapolation assumption,
not a fitted fact about a new circuit and not a change to the existing fitted
likelihood, numerical study or acceptance rules.

The existing [factor builder](../f1rank/design.py) estimates a fixed circuit axis
from qualifying car deviations. It scales eligible factors by their spread but
does not subtract their mean. The new law retains that origin and unit: a fresh
standard Normal factor at an unseen circuit. It does not recenter the training
factors or reinterpret the qualifying car states. Even when a short training
history supplies no learned axis, this population scale remains an explicit
assumption, with uncertainty in the team loading retained.

The same circuit factor is used for every car and every future visit within a
posterior trajectory. Each car's paired team-season loading multiplies it. After
centering across the cars in the future field, that effect enters qualifying
pace and enters race pace through the paired qualifying/race car link. Drawing
independent track effects for each car, or averaging loadings before multiplying,
would lose the shared dependence.

For centered loadings `l[i]` held fixed, the new circuit effects have mean zero
and covariance `l[i] * l[j]`. Their relative effect has variance
`(l[i] - l[j])²`. These are algebraic consequences of the specified population
law, not empirical recovery results. The factor persists across seasons while
team-season loadings follow the existing model. Separate circuits receive
independent coordinates.

The forecast interface must require this policy for a new joint-model circuit;
it cannot silently select the former neutral-factor convention. Known circuits
retain their fitted fixed factors, including zeros from the existing data
availability rule. The separate race-only ablation intentionally removes the
qualifying-derived axis and continues to do so for new circuits.

This extension does not estimate uncertainty in the factors learned from the
training data, identify a physical circuit characteristic, or establish that a
Normal extrapolation is calibrated. Those limitations, together with the known
short-history convention, must remain explicit in the final protocol. Synthetic
equivalence, moment and invariance checks can verify implementation but cannot
accept this model or produce a driver ranking.
