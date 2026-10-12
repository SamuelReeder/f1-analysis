# Overall comparison baselines: development interfaces

The [model plan](overall_model_plan.md) fixes the primary comparison as the existing
grid-plus-qualifying race model, with grid-only and current-race-only secondary
comparisons. [The implementation](../analysis/overall_baselines.py) now provides
the primary and grid-only conditional race-order likelihoods and their predictive
arithmetic. It does not fit models, score real races or authorize publication.

The primary baseline preserves the definitions in
[`qualifying.features`](../f1rank/qualifying.py) and
[`championship.race_model`](../f1rank/championship.py):

- Driver features are the posterior median of each paired `skill + compat` sum.
- Car features are the posterior median of each paired `car + car_track` sum.
- The qualifying fit uses the legacy lineage compatibility definition in
  [`model`](../f1rank/model.py), separately from the successor's era definition.
- Race strengths combine negative log grid slot with car and driver features.
  Coefficients retain the legacy Normal priors. Grid-only retains just its grid
  coefficient, and requires no qualifying features.

The source implementation uses medians even though the old championship module's
introductory text says means. Summing marginal medians can also differ from taking
the median of the paired sum. The new helper reproduces the executable feature
definition. It must receive qualifying-only draws; substituting states updated
with race observations would change the primary baseline.

These are fixed qualifying features, as in the legacy model. Predictive arithmetic
retains and integrates the joint race-coefficient draws. It does not propagate
the qualifying-feature uncertainty discarded by the legacy plug-in policy. That
limitation must accompany the comparison; a version integrating qualifying draws
would be an additional explicitly specified arm, not a silent replacement.

The numerical likelihood uses the corrected Plackett–Luce implementation. This
preserves the statistical definition while avoiding the known order-arithmetic
failures; old cached scores are not relabeled as new evidence. Synthetic checks
compare prior sites and ordinary likelihood values directly with the legacy
implementation, then check normalized orders and predictive integration.

Inputs preserve the complete declared field, including entrants outside the
running panel. Grid slots must already follow the common complete-field pit-lane
policy. Running ranks are supplied explicitly; the adapter never selects finish
statuses, substitutes an average for a missing feature, or omits an empty running
panel. This is a conditional order model with no retirement probability. Its
empty-panel score therefore supplies no retirement validation. The common
classification-conditioning policy still needs to be frozen for every arm.

The arithmetic layer cannot establish convergence, as-of source availability,
matching training histories or observed/forecast event cutoffs. Those remain
mandatory checks in the evaluation controller and its registration. In
particular, the old exclusion of failed qualifying folds cannot be inherited as a
way to omit required successor comparisons. Missing or failed baseline fits must
remain visible and prevent a complete comparison.

Current-race-only still requires its own probability-model and prior definition.
The adapter rejects that arm instead of mapping it to legacy ratings-only. The
end-to-end pre-qualifying comparison also still requires a registered grid and
unknown-condition policy. No real-data scores, sampler changes or source-hashed
model changes are made by this development interface.
