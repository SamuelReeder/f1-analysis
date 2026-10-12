# Calibration sensitivity planning

`analysis/overall_calibration_power.py` calculates sensitivity before choosing
the full recovery study's size. It does not read fitted posterior ranks, select
a new seed after a fit, or replace the original failed numerical experiment.
Its planning grid, error budget, illustrative power target and synthetic error
patterns are committed before running the calculation. They are planning
assumptions, not a frozen calibration registration or acceptance decision.

## Inventory and probability calculation

The inventory comes from the current design builders and the complete recovery
quantity definitions. It includes every current driver/car contrast, context
coefficient, observation likelihood and finite-simulation ranking quantity.
No response is generated or scored to choose quantities. Fixed-truth stress
cases remain in the workload inventory and outside uniform-rank tests. The
Bonferroni family counts every prior-predictive case, quantity and nontrivial
rank-CDF cutoff, exactly as the existing summary does.

At a cutoff, the count among independent replicates is binomial. The null
probability follows the uniform discrete-rank reference. The alternative
probability is the cumulative mass of a specified synthetic rank distribution.
The calculation enumerates the same two-sided
[SciPy binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
and strict adjusted threshold as the evaluator, then sums the alternative
binomial probability of all rejected counts.

CDF cutoffs within a quantity are dependent. The maximum of their rejection
probabilities is a lower bound on detecting an error in that quantity; their
sum, capped at unity, is an upper bound. These are not exact whole-family power
calculations, and driver-pair dependence is not turned into extra trials. The
lower bound also guarantees at least that chance of a family rejection when
that quantity has the stated error distribution.

## Fixed planning patterns and interpretation

The grid includes a uniform reference and mixtures concentrating additional
mass at the lower end, both ends or middle of the rank support. It also includes
the exact binary data-ignoring example from the recovery tests, where the
likelihood quantity detects an inference algorithm that returns the prior.
These are controlled sensitivity patterns, not measured defects in the F1
posterior or comprehensive descriptions of possible inference errors.

The output preserves every grid size and alternative. It identifies the first
grid size whose lower bound meets the illustrative power target, if any, and
reports balanced inventory work counts. This does not select the eventual
study size or authorize those fits. Work counts are arithmetic, not a measured
runtime prediction. The local ranking-functional resource measurement covers a
synthetic scenario; full inference cost and intended-runner resources remain
unknown.

The calculation assumes complete, independent replicates. The new
[sampling-stream checks](overall_sampling_streams.md) enforce distinct
predeclared generation and sampling keys. They do not by themselves prove
accurate inference. A missing/failed case still prevents calibration acceptance;
power cannot be claimed by discarding it.

Full numerical sensitivity for close rankings, error-pattern relevance,
study-size selection, frozen protocol and actual execution remain required.
The separate coordinate study still awaits approval. No planning probability
establishes recovery, real-data calibration or permission to publish a ranking.

## Recorded planning result

The [committed calculation](../outputs/championship/development/overall-calibration-power/planning.json)
counts **9,496** rank-CDF tests in the complete prior-predictive family. With its
illustrative **5%** family error budget and **80%** lower-bound power target, the
earliest tested grid sizes are:

| Extra rank mass placed at | 10% mixture | 25% mixture |
|---|---:|---:|
| Lower end | 1,600 | 200 |
| Both ends | 6,400 | 1,600 |
| Middle | 6,400 | 800 |

These are independent replicates of the affected case. A mixture replaces the
stated fraction of a uniform rank distribution with mass at the indicated
positions; it is not a percentage error in a driver's ability. The exact binary
prior-only example first meets the illustrative target at **100** replicates.
The uniform reference correctly has no grid size meeting that detection target.
These are the earliest points on the declared grid, not exact minimum sample
sizes or an approved inventory. Every grid result is retained in the artifact.

This calculation shows why a small computational profile cannot establish
sensitivity to subtle calibration errors across the full quantity family. It
does not justify removing difficult quantities, weakening a gate or treating a
large study as affordable before its fitting cost is measured. Actual study
size remains unselected.

The [verification record](../outputs/championship/development/overall-calibration-power/verification.json)
preserves the initial calculation and a reproduced floating-point overshoot in
a long sum of binomial probabilities. The corrected arithmetic uses binomial
tail functions and enforces probability bounds within rounding tolerance.
Inventory, alternatives, grid, error budget, rejection regions and the earliest
grid sizes are verified unchanged. The exact small enumeration, null error
budget and saturation checks pass. This numerical correction changes no model,
sampling setting or scientific acceptance rule.
