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

These coordinates are latent circuit uncertainty. A caller repeating outcome
simulations for a fixed outer latent state must hold them fixed; the lower-level
coefficient helper accepts explicit coordinates for that purpose. Changing the
trajectory seed draws a different population realization. The current iterator
still produces a single trajectory per joint draw and does not establish the
final nested replication or Monte Carlo precision policy.

This addresses outcome coefficients, not the whole unseen-circuit problem.
The separate [qualifying track-factor extension](overall_unseen_track_factor.md)
now propagates a shared new-circuit coordinate through paired car effects when
explicitly selected. The [historical condition law](overall_future_conditions.md)
supplies development integration of unknown wetness and neutralisation. Final
registration of those assumptions, official grid rules, completion/classification,
accepted-posterior integration and calibration remain required. An accepted
driver ranking cannot be inferred from this extension's software checks.

## Verification

The [tests](../tests/test_overall_unseen_circuits.py) compare with the actual joint
model's deterministic coefficients under an explicit reference change. Exact
Gaussian moment fixtures check the retained common-mean variance, covariance
between new circuits and relative uncertainty. Pipeline checks preserve repeated
visits, draw batches, field permutations and path prefixes; omitted policies,
malformed coordinates and nonfinite joint parameters fail before publication.
Known-circuit behavior is also compared with the existing default path.

The [verification record](../outputs/championship/development/overall-unseen-circuit-checks/verification.json)
links the committed test source and capped local resource measurement. The
resource workload uses synthetic parameters and a hypothetical circuit, including
checked compressed-archive loading. It measures no MCMC, accepted-posterior
behavior, real-event accuracy or intended-runner fitting cost. The original
fitting inputs and pending numerical proposal remain unchanged.
