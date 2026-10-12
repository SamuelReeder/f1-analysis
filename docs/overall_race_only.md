# Current-race-only comparison definition

This specifies the secondary `current_race_only` arm named in the
[Overall plan](overall_model_plan.md), before fitting this arm or inspecting any
of its real-data predictions. It is an evidence ablation of the successor, not
the old career racing residual or legacy ratings-only baseline. It cannot rescue
a failed primary comparison or publish another driver table.

## Included and omitted evidence

The arm uses the same cutoff-first entrant/event/car history and eligible race
laps, grids, running finishing orders, completion observations and race contexts
as the joint candidate. All race likelihoods and their priors remain those of
[`overall.model`](../f1rank/overall/model.py). Actual grids are permitted race-stage
conditioning information for every arm; a race-only comparison does not mean
ignoring the grid.

Qualifying lap times, segment availability and finishing positions do not enter
this arm. Its builder does not open the qualifying-times table. It retains entry
metadata such as driver identity, team, age and experience. Empty qualifying
observation and session arrays provide no fictitious measurements.

The qualifying-derived circuit factor is also removed: its coefficients are zero
throughout training and forecasts. Merely masking qualifying's observation
likelihood would still leak qualifying results through that factor, which
[`design.circuit_factors`](../f1rank/design.py) estimates from qualifying pace.
Race-context circuit effects remain, since they are learned from race evidence.
The comparison therefore measures the qualifying information channel, including
its derived track adjustment, rather than only the qualifying likelihood term
conditional on an already learned qualifying covariate.

## State and prior definition

The inherited latent-state family remains intact. Current race ability still
combines the shared driver walk and an evolving race residual, with the original
learned link, team nuisance, car effects and weekend dependence. Each sampling
distribution is inherited unchanged. In this arm the shared component is
inferred only through race observations; its internal name does not imply that
qualifying observations have been used.

Qualifying-specific scales with no race role remain prior-only. Segment-indexed
sites have no coordinates when the observation block is empty. The diagnostic
report records those empty sites rather than treating them as varying posterior
parameters or throwing a reshape error. Existing coordinates retain the same
convergence checks; an archive containing only empty or constant coordinates
cannot pass. No threshold, retry setting or previous fit result changes.

This choice makes the role of qualifying evidence explicit while retaining the
candidate's state prior family. It may expose weakly identified latent components;
that is a property to report and evaluate under the registered recovery and
numerical rules, not grounds to fix a link or narrow a prior after a result.

## Implementation and validation boundary

[`overall_race_only`](../analysis/overall_race_only.py) provides the separate
builder and probability model. The builder preserves the race blocks and removes
qualifying-derived inputs before data transformations. The model defensively
removes those inputs again. The forecast catalogue requires the explicit arm
identity for an empty qualifying history and rejects qualifying observations or
a restored qualifying-derived circuit factor in that arm. Training-array
identities distinguish its inputs from a joint-candidate fit.

Synthetic checks cover the observation boundary, inherited density, race-factor
sensitivity, finite gradients, matched future states and conditional forecast
batches. Synthetic forecast parameter values are software fixtures, not a
converged fit. A race-only prediction must eventually use its own checked fit;
joint-candidate draws cannot stand in for that posterior.

The arm still needs fitting integration, the common frozen event inventory,
source-vintage checks, complete recovery and calibration, and comparative scores.
Its pre-qualifying forecasts would include prior-only qualifying noise, so they
require evaluation under the common registered grid policy as well. The current
conditional forecast interface does not settle official-grid, unknown-condition,
lapping, exposure or classification policy. This definition authorizes no fit,
new profile run or publication, and leaves the original failed profile preserved.
