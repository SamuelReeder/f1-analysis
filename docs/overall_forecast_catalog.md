# Overall forecast input catalogue

[`overall_forecast_catalog.py`](../analysis/overall_forecast_catalog.py) maps a
declared future field to the training history without reading the target
weekend's observations. It is development infrastructure for the
[evaluation plan](overall_model_plan.md), not a predictor or accepted evidence
of improvement.

## Information available at the origin

The caller must supply a timestamped snapshot with its source reference, event
schedule, scheduled distance, complete declared field, and intervening calendar.
The catalogue supports the plan's pre-qualifying and actual-grid information
sets. The actual-grid input requires an explicit starting position for every
entrant. Pre-qualifying input rejects grid positions. Both reject realized
qualifying times, race results, completed distance, weather and neutralisation.
Future conditions stay unknown here. The separate
[historical condition model](overall_future_conditions.md) integrates them in
development forecasts; its calibration and final registration remain required.

Dates must carry timezones. Training events must precede the forecast origin,
which must satisfy the declared session schedule. The catalogue binds its
training arrays, normalized snapshot and implementation with content hashes.
Changing held-out tables cannot supply a feature through this interface.

A source marked `archived_asof` must claim capture before the origin. A
`retrospective_reconstruction` remains labeled as such. These are caller claims:
the catalogue does not retrieve the source, verify its timestamp, certify that
the field was complete, or establish that an archive was previously untouched.
Every output leaves source verification and independent-evidence flags false.
The evaluation runner must verify and preserve those materials separately.

The supplied calendar must include intervening events. Within-season missing
rounds are rejected, but the checker cannot establish a preceding season's last
round from event identifiers alone. Calendar completeness, including across
seasons, therefore requires a verified source. Declared intervening driver
participation supplies experience counts; it is not inferred from future results.

## Retaining the complete field and its uncertainty

Every declared entrant remains in the catalogue, including drivers without a
qualifying measurement. Unseen drivers require explicit biographical and prior
experience metadata. Missing history is represented by missing state references,
never an assigned average rating or a dropped entrant. An unseen driver may
have prior racing experience; unknown history does not imply rookie experience.

Entries identify the relevant driver history, team/era stint, car history,
circuit and elapsed event/season gaps. Teammates share a car identity. Transfers,
new teams and regulation boundaries preserve which states must be generated
instead of reused. A pit-lane grid feature uses the full declared field before
any result-based exclusions, matching the corrected training design.

The returned historical indices are not future ratings. A predictor must
reconstruct the underlying uncentred states, propagate their joint uncertainty
through the registered transitions and intervening calendar, and then centre on
the forecast field. Comparing separately centred states from different past
events would introduce an arbitrary difference between fields. New driver, car,
team/era and circuit states also require their declared population uncertainty.
That propagation is separate from this catalogue; the development implementation
is described in [forecast states](overall_forecast_states.md).

The same field/time validation now supports a separately identified legacy
qualifying-only catalogue. It binds only that qualifying design's arrays, with
the `legacy_qualifying` arm identity, rather than filling in missing race
observations or relabeling a successor catalogue. Its
[baseline forecaster](overall_baselines.md) preserves the original lineage
effect definition. Source verification remains false for both catalogue paths.

## Evidence and remaining work

Synthetic boundary checks live in
[`test_overall_forecast_catalog.py`](../tests/test_overall_forecast_catalog.py).
The grid-feature regression and input audit are retained under
[`overall-forecast-catalog-checks`](../outputs/championship/development/overall-forecast-catalog-checks/).
The original failed numerical profile remains unchanged. Its synthetic arrays
and truths are reproduced by the audit despite the training-input correction;
the still-pending coordinate experiment uses the updated source fingerprint.

The [verification record](../outputs/championship/development/overall-forecast-catalog-checks/verification.json)
ties the focused and full-suite reports to their source revisions and preserves
the original failing regression. It is software evidence only.

Source collection and verification, accepted-posterior integration, condition
calibration and final registration, baseline integration and the frozen evaluation inventory remain
required. The catalogue itself propagates no states and scores no predictions.
The separate state implementation has synthetic checks; a full predictor and
its scientific validation remain required. Neither module publishes a ranking
or establishes production readiness.
