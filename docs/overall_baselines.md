# Overall comparison baselines: development interfaces

The [model plan](overall_model_plan.md) fixes the primary comparison as the existing
grid-plus-qualifying race model, with grid-only and current-race-only secondary
comparisons. [The implementation](../analysis/overall_baselines.py) now provides
the primary and grid-only conditional race-order likelihoods and their predictive
arithmetic. The [legacy forecast adapter](../analysis/overall_legacy_forecast.py)
propagates qualifying-only future states and connects their fixed features to
the primary baseline's actual-grid race predictions. These interfaces do not
fit models, score real races or authorize publication.

## Qualifying forecast definition before evaluation

The legacy race baseline requires a separate qualifying-only posterior from the
standard model, with its original lineage compatibility. Its future driver and
car states must not be copied from the successor's race-updated posterior or
its era-based team effects. The adapter requires the standard fitting
metadata and matching qualifying arrays, rejects successor-specific posterior
sites, and leaves source/convergence verification to the checked fit loader.

The original `f1rank.fit.fit` exporter does not retain the sampled coordinates
needed by this adapter by default. Future registered baseline fits must pass
`extra_sites=FORECAST_EXTRA_SITES`, declared in the adapter, to retain the raw
driver walks, car innovations, track loadings and lineage effects. This changes
the saved variables only; it does not change the likelihood, priors or sampling
settings. Existing centred rating arrays cannot silently substitute for missing
uncentred states. An older archive lacking these coordinates is rejected and
remains unusable for this forecast path. No old fit has been relabeled or rerun.
The [bounded legacy reader](../analysis/overall_legacy_archive.py) preserves the
original metadata/design check and additionally requires explicit archive,
sidecar and full training-array identities. It validates every draw array and
scalar information field, including unselected members, and closes temporary
read-only maps on exit. Successful exit rechecks the original files and design.
It does not convert legacy metadata into a successor fit receipt or certify
source provenance, convergence or scientific acceptance. Those checks remain
required before an accepted baseline forecast.

The development resource workload in
[`overall_legacy_resources.py`](../analysis/overall_legacy_resources.py) uses the
committed cutoff/data dimensions with arbitrary synthetic values. It includes
historical state blocks that the forecast does not retain, checks every archived
member, and compares the complete future features and hypothetical actual-grid
order probabilities before and after compressed loading. Its draw shape comes
from the existing largest Overall retry solely to size the workload; it does not
choose future baseline fitting settings. The workload measures the local archive
and forecast path, not MCMC, accepted-fit diagnostics or intended-runner capacity.
The [archive and resource record](../outputs/championship/development/overall-legacy-archive-checks/verification.json)
links its local measurement and synthetic I/O checks to their committed source.

Propagate the original uncentred driver and persistent-car walks from their last
training states, then centre on the complete announced future field. Retain the
existing event/season gap scales, experience/age trend, car carryover and reset
rule. A new driver, team or driver-lineage pair receives the original population
distribution. Returning to a lineage reuses its driver-team effect even across
an era boundary, as specified by the legacy likelihood. New team-season track
loadings receive the existing prior.

Known circuits use the fixed factor learned from qualifying training data.
An unseen circuit retains the original legacy no-adjustment convention, recorded
explicitly in its output. This is a disclosed baseline limitation; adopting the
successor's new population extrapolation here would change the specified primary
baseline. No future qualifying results or circuit factors estimated with future
observations may enter the forecast.

Future transients can generate qualifying observations, but they do not enter
the legacy race features. Compute the driver feature as the median of paired
future skill plus lineage-compatibility draws, and the car feature as the median
of paired persistent-car plus track-effect draws. Every posterior draw must be
accounted for before taking those medians; sums of marginal medians are not
equivalent. This preserves the legacy plug-in definition rather than propagating
qualifying uncertainty through the race coefficients.

The forecast field and intervening lineups/calendar are declared outcome-free
inputs at one origin. The adapter must preserve new entrants, team transfers,
missing qualifying observations, path prefixes and batch/field invariance.
Random streams identify state, driver/team and event so iteration order cannot
change the forecast. The numerical forecast replication/precision policy and
source-vintage verification remain required for the final evaluation; a single
development trajectory per qualifying draw does not establish them.

The actual-grid adapter reads grid slots only from the forecast catalogue,
including its complete-field pit-lane convention. Later response input supplies
only running positions and must retain every declared driver, including those
outside the running panel. It rejects a pre-qualifying origin instead of inventing
a grid. It keeps qualifying-fit and race-coefficient identities separately: the
legacy fixed feature medians do not require equal posterior draw counts or
arbitrary pairing between those fits. End-to-end pre-qualifying grid integration
remains a separate open requirement. Its
[development probability definition](overall_legacy_prequalifying.md) specifies
how to integrate the grid and coefficient distributions without arbitrary
pairing, while retaining the remaining official-grid and precision requirements.

The [forecast verification record](../outputs/championship/development/overall-legacy-forecast-checks/verification.json)
links synthetic original-model trace comparisons, lineage persistence, newcomer
and missing-qualifying cases, unchanged features under transient variation,
batch/field/prefix invariance, future-result rejection and a hand-computable
actual-grid order probability. It also records checks of the shared catalogue
and forecast path used by the successor. It establishes no accepted fit,
real-race accuracy, forecast-median precision or runner resource profile.

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

The separately implemented [current-race-only arm](overall_race_only.md) removes
the qualifying evidence channel from the successor while retaining its race
likelihoods and state prior family. The legacy adapter rejects that arm instead
of mapping it to legacy ratings-only. Fitting integration and the end-to-end
pre-qualifying comparison still require a registered grid and unknown-condition
policy. No real-data scores or sampler changes are made by these interfaces.

The [verification record](../outputs/championship/development/overall-baseline-checks/verification.json)
links the focused and full local Python checks to their committed implementation.
Those are synthetic/software checks, not comparative predictive evidence.
