# Conditional simulation uncertainty in expected-point ordering

This definition precedes its implementation and fixed saved-fixture analysis.
It supplies the missing near-tie sensitivity report for the complete grouped
simulation moments. It does not change a model, score, convergence criterion or
acceptance gate. It cannot establish functional MCMC convergence.

## Simultaneous conditional mean intervals

For each fixed posterior draw, independent replicated seasons produce points
per Grand Prix in `[0, B]`, where `B` is the maximum of the committed scoring
table. A paired explanation loss is in `[-B, B]`. Dependence between entrants,
between paired interventions or between posterior draws is allowed. Independence
of the replicated seasons conditional on each draw is required; mean-zero error
or a sample variance alone is insufficient.

The bounded-independent-variable inequality in
[Hoeffding's original paper](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)
gives `P(|estimated_mean - conditional_mean| > e) <= 2 exp(-2 R e² / w²)`
for replicate count `R` and outcome range width `w`. Allocate a declared error
probability `delta` over the complete family of `K` means. Taking
`e = w sqrt(log(2 K / delta) / (2 R))` and a union bound gives simultaneous
coverage at least `1 - delta`, conditional on the retained posterior draws.

For a complete controller companion, `K` includes every chain, draw, driver and
required explanation as well as the baseline means. Use width `B` for points
and `2 B` for losses. Intersect each interval with its known support. Preserve
all draws and each draw's replicate count; do not select a different error
probability, add replicates or drop a near tie after inspecting results.

This is a confidence bound on inner simulation error for a fixed trace. It is
neither a posterior credible interval nor an MCMC error bound. It makes no
assumption that posterior draws are independent or representative. The consumer
must keep those distinctions explicit even if all intervals are narrow.

## Propagate intervals to all comparisons

For each pair at each draw, subtract opposite endpoints to bound the exact
conditional expected-point difference. The fraction with a strictly positive
lower difference bounds the empirical fraction favoring the first driver from
below. The fraction with a strictly positive upper difference bounds it from
above. Retain equality bounds and both directions. Report these fractions for
each chain and the complete trace; they are not authenticated posterior
probabilities. Equality is forced only when both point intervals coincide.

For each driver and draw, the best possible rank is one plus the number of
competitors certainly ahead; the worst is the field size minus the number
certainly behind. These bounds include unresolved ties. Aggregate bounds on
the rank CDF by using the worst rank for its lower CDF and best rank for its
upper CDF. They cover every tie-respecting ordering, not a chosen tiebreak.
Keep expected-point ranks distinct from the existing simulated-season ranks.

Also average interval endpoints across the complete retained trace to bound
its mean scores and pairwise differences. This checks sensitivity of the
canonical expected-score ordering to simulation error. It does not measure the
error from a finite or nonconverged posterior trace. No precision, convergence,
scientific-evidence or publication flag becomes true in this report.

## Integration and checks

Read only a completed, fingerprint-matched grouped companion using the existing
checked loader. Bind the diagnostic's definition, implementation, scoring
contract, companion reader, complete input receipt/summary/archive and declared
error probability in the new report. Recheck inputs and sources before writing
to a new output directory. Do not modify or reinterpret the simulation receipt.

Check the formulas with an exhaustive bounded Bernoulli experiment rather than
claiming empirical coverage from a handful of successes. Verify interval
propagation against all vertices of small boxes, exact ties, zero-width
intervals, chain disagreement and field permutations. The connected test must
read a full controller-produced companion and reject changed/failed inputs.

Apply the fixed procedure to both already committed noise-audit cases, using
their complete saved traces and original replicate count, without new random
draws. For this points-only fixture the family contains every baseline mean;
there are no invented explanation observations. Set its error probability to
one minus the existing contract interval probability before running the audit.
Preserve both cases regardless of the sensitivity bounds. This is a descriptive
reanalysis of known synthetic examples, not new validation or evidence about a
real driver. Full functional MCMC diagnostics and prospective precision
registration remain required.
