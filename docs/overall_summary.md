# Canonical Overall summary: paired simulation aggregation

This definition precedes implementation and evaluation of the successor's
summary calculation. It implements part of the canonical producer described in
the [Overall plan](overall_model_plan.md), not a new public ranking. It neither
changes the fitted probability model nor authorizes fitting or publication.

## Inputs and coverage

Keep the grouped posterior chain and draw axes. Every draw must receive the
same declared number of independent simulated seasons, on the same reference
calendar, with the complete field. Divide each season's total by the common
number of Grands Prix before calculating expected points. Do not weight drivers
or posterior draws differently because a simulation failed or finished sooner.

Consume bounded blocks of draws and consecutive replicate ranges. Reject a
missing, repeated or out-of-order replicate range; only finalize when every
declared draw has its full workload. Each block must identify the same fit,
posterior, scenario, simulation procedure and field as the accumulator. These
identity checks establish consistency, not authentic scientific provenance;
the future canonical controller must verify the artifacts behind them.
Bind the summary implementation, definition and display contract before
accumulation and reject source changes before finalizing a result.

The baseline input includes paired season points, title shares and sporting rank
intervals. Validate tied rank groups and their fractional title shares. A tied
group occupies its full sporting interval and splits title probability equally.
Rank groups must agree with championship points, and the field cannot receive
more points than the common scoring allocation. Equal points may still be
separated by sporting countback; unequal points cannot share a tied rank group.
Names or input order never resolve a sporting tie. Use lower-rank samples for
the lower season-rank endpoint, upper-rank samples for the upper endpoint, and
tie midpoints for the reported median. Quantiles use the same linear convention
as the existing summary, including when a quantile falls between observations.

## Explanatory probes

For each driver and joint posterior draw, replace that driver's portable
qualifying state with the current field mean in that draw, its portable total
race state with the corresponding field mean, or both. Other drivers, every
observation/context parameter and the reference scenario stay unchanged. The
field mean includes the selected driver. Use the same simulation random inputs
and retain the affected driver's points from every paired baseline/probe season.

The accumulator consumes all declared probes together with their baseline
seasons. The loss is baseline points minus probe points, divided by the same
race count. It may be negative. It is an explanatory intervention in this
conditional model, not a causal attribution of innate talent. Separate losses
must not be added: neutralizing correlated components can have interacting
effects. This definition requires no learned component weights.

The paired block generator invokes the existing simulator for the baseline and
replays its event randomness for every intervention through the
[equivalent paired kernel](overall_paired_simulation.md). This reuses qualifying
grids and omits unused probe championship countback, while preserving all paired
point totals. Its random stream address contains the fixed seed,
chain, draw-block start and replicate-block start; these coordinates differ
between blocks and match within paired interventions. The complete block layout
uses fixed-width unsigned stream words enforced by the generator. Oversized
coordinates are rejected before simulation, rather than allowing ambiguous
variable-width integer encodings at the random-stream boundary. The layout
and seed must be fixed in the future simulation registration. Changing the layout
can change random draws and requires a different simulation identity. This
generator inherits the existing simulator's unresolved completion assumptions
and raises on failures; it never substitutes another season.

## Separate uncertainty calculations

Within each posterior draw, retain points means, variances and the full driver
covariance matrix, plus the means and variances of paired probe losses. Combine
blocks using centered moment updates. This keeps conditional Monte Carlo error
for a driver difference or explanatory loss paired; adding marginal errors as
though they were independent is incorrect.

Expected-point intervals describe the distribution of conditional means across
posterior draws. Season-rank intervals and title probabilities describe simulated
outcomes across draws and seasons. Report conditional simulation errors
separately from posterior variation and from MCMC integration error. The
posterior standard deviation calculation retains the existing subtraction of
estimated simulation-mean variance; it is not an MCMC error estimate.

The existing points-error limit is retained. Before presenting an expected-point
interval as sufficiently precise, check both the estimated conditional error
and the bounded-variable error guarantee already described in
[ranking recovery](overall_ranking_recovery.md). For points in `[0, B]`, the
conditional standard error bound is `B / (2 sqrt(R))`; for a paired difference
in `[-B, B]`, it is `B / sqrt(R)`. Here `B` is the maximum of the committed points
table and `R` is the fixed replicate count. Do not increase the workload, change
seeds or drop draws after inspecting precision. Insufficient precision leaves
the corresponding interval unavailable. These are development precision
reports, not a replacement for the final registered acceptance procedure.

Pairwise comparisons use all pairs and the same posterior means, preserving
joint-draw differences and conditional covariance. Estimated probabilities of
positive and exactly tied finite means are labeled as finite-simulation
quantities. They do not establish the ordering of arbitrarily close true
expected values. Rank sensitivity and MCMC diagnostics remain required.

## Remaining integration

The summary must be checked against direct calculations on small known arrays,
including exact ties, perfectly correlated driver outcomes, unequal replicate
blocks, field/draw permutations, paired explanatory losses and incomplete
inventory rejection. Bounded accumulation avoids keeping the entire nested
simulation cube in memory; it does not reduce the declared simulation workload.

The posterior adapter, accepted common scenario,
scientific evidence checker, exporter and refresh controller must still be
connected before this can produce the canonical payload. The original failed
numerical profile remains failed. No synthetic array or caller-supplied success
flag can establish scientific acceptance; the development result always leaves
publication disallowed.
