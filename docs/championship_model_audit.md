# What drives the provisional ranking, and what needs work

Audit dated 2026-10-08, using the frozen data through `2026-15`. This diagnosis
does not establish a best-driver ranking or change any completed validation.
The existing combined publication gate remains failed.

Hadjar's high position is mainly inherited from the qualifying estimate and its
current-team effect. The added career race-pace residual reduces his estimate.
There is real qualifying evidence behind the result in this dataset, but the
overall model does not adequately represent all the things implied by “best
driver.” Its independent current-season race model tells a materially different
story. The strongest next direction is to reconcile those models and propagate
their uncertainty, rather than change settings until a preferred ranking appears.

The [audit plan](championship_model_audit_plan.md) and
[script](../analysis/championship_model_audit.py) were committed before running
the probes. Numerical evidence is in
[`outputs/analysis/championship_model_audit/`](../outputs/analysis/championship_model_audit/).
The source revision and input hashes are recorded there. No new fits were run.

## The observed cause of Hadjar's position

The simulation reproduced the existing preview to floating-point precision:
the maximum expected-points discrepancy was `4.44e-16`. No missing current-driver
race residual, driver-order mismatch, or reversed coefficient sign was found in
this path. All the preset simulation seeds, `0` through `4`, put Hadjar second by
mean points. His position is not explained by choosing a fortunate simulation seed.
Sources: [summary](../outputs/analysis/championship_model_audit/summary.json) and
[seed results](../outputs/analysis/championship_model_audit/simulation_seeds.csv).

The following probes hold the fitted race coefficients fixed and use identical
random numbers. They diagnose the existing simulation; they are not refitted
alternatives, causal effects, or predictive validation.

| Probe | Hadjar's order by mean points | Expected points per race |
|---|---:|---:|
| Original preview | 2 | 6.832 |
| Remove everyone's current-team qualifying effect | 7 | 4.980 |
| Give only Hadjar an average team effect | 6 | 4.904 |
| Remove the extra race-pace component for everybody | 2 | 7.232 |
| Give everybody average qualifying ability | 15 | 4.338 |
| Use posterior-mean qualifying and race ability | 2 | 7.191 |
| Remove independent car-weekend noise from simulated qualifying | 2 | 6.965 |
| Give only Hadjar average qualifying ability | 14 | 3.939 |
| Give only Hadjar average extra race pace | 2 | 7.153 |

Source: [all drivers and probes](../outputs/analysis/championship_model_audit/scenarios.csv).
The portable probe uses paired random numbers; the original preview's portable
table advances the random stream after its headline simulation, so its mean
points differ slightly. Neither table was replaced.

The qualifying model assigns Hadjar a portable-skill median of `+0.056%` and a
current-team-effect median of `+0.126%`. Those marginal medians are not an exact
additive decomposition of the combined median. The team-effect interval runs
from `-0.073%` to `+0.331%`. His skill and team-effect draws have correlation
`-0.507`; his in-team pace and current-car pace have correlation `-0.568`.
These are examples of the uncertainty in separating driver, team and car.
They do not show that the team effect is fictitious or that its causal source is
car handling. Source: [components](../outputs/analysis/championship_model_audit/driver_components.csv).

There is an important countercheck: already-completed qualifying sensitivity
fits put Hadjar between second and fourth by in-team median. Refitting the model
without team effects puts him third, whereas simply removing those effects from
the fitted simulation puts him seventh overall. These answer different questions.
We cannot infer that removing a term and refitting would produce the same result
as a fixed-coefficient probe. Source: `existing_qualifying_structural_sensitivity`
in the audit summary.

The recorded observations also explain why the qualifying estimate is high:

- Against Lawson in `2025`, Hadjar was faster in the event-averaged matched
  qualifying segments at `16` of `21` comparable events; the mean advantage was
  `0.187%`.
- Against Verstappen in `2026`, he was faster at `5` of `10` comparable qualifying
  events, with an overall mean deficit of `0.052%`.
- The adjusted dry-race comparisons are different: against Verstappen in `2026`,
  Hadjar was slower in all `8` available paired races, with a mean deficit of
  `0.542%`.

These are retrospective, equally weighted event comparisons, not forecasts or
new significance tests. They use the model's included segments and the existing
adjusted race data; they do not cover every entry or establish why a gap occurred.
The full event records and aggregation definitions are in
[qualifying comparisons](../outputs/analysis/championship_model_audit/hadjar_qualifying_event_gaps.csv),
[race comparisons](../outputs/analysis/championship_model_audit/hadjar_race_event_gaps.csv),
and the audit script.

## Main model limitations, in order of practical importance

### The overall model does not use the newer current-season race estimate

`championship_preview.py` imports only `u` from `racemulti`: a career-level race
residual conditional on qualifying. The race model itself also has driver-season
and driver-race effects (`e` and `w`), but those do not enter the simulation.
Current qualifying plus a lifetime residual can miss a driver's changing race
performance. Historical coefficients also pool different eras into the same
grid-to-finish relationship.

The separate `race_total_model` estimates current-season total dry-race pace
without qualifying inputs. Its published driver table puts Hadjar ninth by median,
based on `9` usable current-season races. Its held-out driver-prediction check
compares teammate gaps and passed; that does not establish cross-team causal
rankings or validate using it inside a championship model. The disagreement is a
reason to investigate this missing input, not proof that ninth is his true rank.
Sources: [components](../outputs/analysis/championship_model_audit/driver_components.csv),
[`pace.json`](../outputs/race_total/pace.json),
[`racemulti.py`](../f1rank/racemulti.py),
and [`race_total_model.py`](../f1rank/race_total_model.py).

A proposed successor should let qualifying predict the grid and current-season
race pace predict finishing performance, estimating their conditional weights.
It must avoid adding total race pace on top of its already counted qualifying
component without accounting for their relationship. This is a development
proposal; no such model has been fitted or scored by this audit.

### “Equal car” still retains uncertain team-associated advantages

The headline explicitly keeps each driver's current-team effect. That effect can
absorb adaptation, equipment differences, team support, role and selection, as
well as car handling. Equalizing a car intercept does not remove those influences.
The result therefore answers a narrower scenario than universally transferable
driver ability. The portable result is also weakly identified; changing the label
or removing team effects is not enough to establish a causal ranking.

The posterior correlations above and the paired probes quantify the sensitivity.
Future work should evaluate recovery of driver/car/team effects and uncertainty
around team changes, rather than impose a preferred driver order.

### Uncertainty is passed between stages incompletely

`qualifying.features()` supplies median qualifying features to the race models.
The finishing model is fitted with posterior-mean race qualities, then simulation
draws qualifying ability, race qualities and finishing coefficients separately.
Quality covariance from the same saved fit is preserved where applicable, but
cross-model dependence is not. The race residual was itself estimated conditional
on qualifying, so these components are not independent measurements of ability.

This is a modular approximation, not a fully joint posterior. The audit does not
measure the direction or magnitude of resulting interval error. Training the final
stage on full-data fitted features also differs from the season-ahead feature
policy used in validation. A successor needs either joint inference or an explicit
uncertainty-passing procedure with predictions made from genuinely earlier data.

Qualifying enters both the grid and race-strength terms. That is not automatically
a double-counting bug: their coefficients are estimated conditional on each other.
Whether the resulting combination predicts well is an empirical question, and the
combined publication test still failed.

### The simulated qualifying grid is not generated by the fitted observation model

`model.py` fits Student-t noise with segment-varying scale and a car-segment effect.
`championship.simulate()` instead adds Gaussian form and Gaussian noise using
`sqrt(sd_car_event**2 + sigma0**2)`. It does not reproduce the fitted segment-scale
variation, heavy tails, shared car-segment effects or the qualifying session
progression. The fitted median degrees of freedom is `3.889` and the median
segment log-scale variation is `0.564`, so these terms are not absent in the fit.
Source: `qualifying_hyperparameters` in the audit summary.

Independent car-weekend noise is an intentional “equal on average” assumption in
the championship code, although its introductory description suggests a common
car term. The scenario should state this consistently. Removing that noise leaves
Hadjar second in the fixed probe, so it does not explain his high position.
The full simulated-grid distribution still needs predictive calibration.

### Race context and several driver qualities are missing

The race stage uses a common grid coefficient and a static ranking distribution.
It has no circuit-specific overtaking difficulty, safety-car intervention, explicit
passing/defending process, pit strategy, tyre strategy or weather sequence.
The simulation holds current driver ability fixed through the season rather than
advancing experience, aging and form trajectories. Qualifying weekend form affects
the grid but has no direct shared race-day effect beyond that grid.

Retirement is an independent yes/no draw with a common fixed probability here.
All retirees receive zero simulated points; completion distance and classification
are not modeled. Driver-specific incident rates were withheld after their gate
failed. Degradation, consistency and first-lap effects are also absent from this
fixed preview, and battle effects had no usable draws. Those are limitations of
scope, not reasons to override failed inclusion tests. Sector times have not been
tested; adding more inputs is not automatically an improvement.

The lap models already account for several tyre, traffic and lap-number effects.
Those are not wholly missing from the project. The missing part is carrying
relevant current driver effects and race context into the finishing simulation
with a validated interpretation.

## Ranking presentation and a concrete implementation issue

Expected points order is not the same as certainty about driver ability. In the
original simulation Hadjar outscores Sainz in `53.15%` of seasons and Norris in
`53.2%`; his median championship finish is fifth. Those are conditional simulated
season probabilities, not probabilities of being intrinsically better. The public
wording should foreground uncertainty and the scenario, not “second-best driver.”
Source: [pairwise outcomes](../outputs/analysis/championship_model_audit/hadjar_pairwise.csv).

There is a concrete tiebreak limitation: championship rank is assigned by sorting
total points, without a sporting tiebreak rule. The resulting tie resolution
depends on array ordering and sort behavior. The original simulation has `17`
seasons tied for the highest points total out of `2,000` (`0.85%`). This affects
title shares and rank summaries; it cannot explain Hadjar's second place by mean
points. A future simulator should retain the race-result information needed for
an explicit tiebreak, or disclose tied outcomes rather than arbitrarily rank them.
Source: audit summary and `championship_preview.py`'s rank calculation.

## Recommended next work

1. Define the intended estimate explicitly: current race ability, current-team
   performance, or transferable equal-car ability. Preserve all existing results.
2. Develop a separate successor that combines qualifying-derived grids with
   current-season race pace and propagates their dependence and uncertainty.
   Address grid-noise and tiebreak consistency in that separate implementation.
3. Audit predictions by driver experience, team switches, circuit and season;
   examine calibration and driver/car separation. Do not filter out inconvenient
   races or select thresholds after inspecting these diagnostics.
4. Only then add individually motivated race mechanisms, evaluated against a fixed
   baseline. Freeze the comparison before scoring, report failures, and retain
   independent future evidence for any claim of reliable improvement.

This audit changes no model package, data, frozen study, dashboard publication or
registered gate. The evidence and report are committed locally; pushing them
requires the user's approval.

Verification: the complete Python suite passed `165` tests. Both the audit and
original-preview manifests verified, and the committed dashboard publication
receipt still passed. See [verification.json](../outputs/analysis/championship_model_audit/verification.json).
This audit checked code and existing model inputs; it did not independently
revalidate the original timing feeds or obtain new predictive evidence.
