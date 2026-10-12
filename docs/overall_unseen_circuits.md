# Forecasting outcome coefficients at an unseen circuit

This development policy extends the existing circuit prior in
[`overall/model.py`](../f1rank/overall/model.py). It does not change the fitted
likelihood, priors, numerical profile, retry procedure or acceptance criteria.
It must be identified explicitly in the final predictive registration before
evaluation. No real held-out scores or new fits have been inspected to choose it.

The policy identifier is `independent_gaussian_with_training_reference`.
It applies to the grid, pace and completion-hazard coefficients. For each joint
posterior draw, preserve the fitted base, population scale and uncentred circuit
coordinates. Draw a fresh standard Normal coordinate for each unseen circuit
and coefficient family. Reuse that coordinate whenever the same circuit appears
again in the forecast path. Different new circuits have distinct innovations;
all entrants at an event share the event's circuit coefficient.

For fitted scale `s`, raw training coordinates `z`, fitted base `b`, and a fresh
coordinate `z_new`, the unseen circuit's linear coefficient is

```
b + s * (z_new - mean(z)) + wet_effect * wet + neutralisation_effect * neutralised
```

The grid and pace coefficients use the existing exponential link; completion
uses the existing logistic link and stable log probabilities. The training mean
remains the reference. Adding future circuits must not recenter the fitted
circuits or reset a new circuit's effect to zero. The uncentred training mean is
not learned from the centred training likelihood, but its prior uncertainty
matters for a new circuit relative to the fitted field. It is retained in the
joint raw posterior coordinates, as required by this interface.

Conditional on a fixed scale, before observing outcomes, let `n` be the number
of training circuits. An unseen deviation has variance `s² * (1 + 1/n)`.
Distinct unseen deviations share covariance `s²/n` through the training mean.
A contrast between an unseen and a fitted circuit has variance `2s²`, as does
a contrast between distinct unseen circuits. These are algebraic consequences
of the existing independent standard Normal coordinates, not estimates from
racing data. Dropping the training-mean uncertainty gives a different forecast.

This is a conditional extension of the fitted model with its original reference
set. Rebuilding a larger centred model with the same independent base prior is
not automatically equivalent: recentering requires transforming the base and
preserving its induced dependence. No such refit or prior replacement is made.

The caller must select this policy explicitly. Omitting it at an unseen circuit
continues to reject the forecast. Seeded streams attach to the coefficient
family and circuit identity, preserving dependence across events, draw batches
and field permutations. Known-circuit predictions keep their existing formula.
The output identifies the policy and remains development evidence only.

This addresses outcome coefficients, not the whole unseen-circuit problem.
The qualifying-derived track factor still uses its existing no-adjustment
convention, which needs a final scientific policy. Unknown future weather and
neutralisation, official grid rules, completion/classification assumptions,
accepted-posterior integration and calibration also remain required. An accepted
driver ranking cannot be inferred from this extension's software checks.
