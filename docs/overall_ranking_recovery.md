# Ranking functionals for development recovery

`analysis/overall_ranking_recovery.py` connects known parameter values to the
existing equal-machinery/team simulator for the unfinished
[recovery study](overall_recovery_checks.md). It does not fit a model, publish a
driver table or supply the missing final weekend assumptions.

## The function being checked

The input is a complete current field and a joint parameter value. Portable
qualifying and total race states enter together with their shared weekend,
observation-noise, grid, pace and hazard parameters. Fitted car/team effects do
not enter the counterfactual driver score. Their estimation uncertainty still
affects the joint posterior from which parameter values must come.

The protocol fixes the complete reference events, context indices, distances,
race-control assumptions, qualifying format, simulator seed and workload. Every
parameter value uses the same workload, lane layout and seed. Repeated known
states batch independent simulated seasons; those lanes are simulation work,
not posterior samples or extra MCMC chains.

The resulting test quantities cover every driver's simulated mean points per
GP, all paired point differences, the lower/upper rank of that mean, and the
shared best-driver indicator. Exact ties in these finite means remain shared;
driver names do not break them. These are ranks of mean performance, distinct
from the simulator's championship countback for individual season outcomes.
Paired numerical errors use the same simulated seasons and retain their
covariance.

These quantities are explicitly named `finite_simulation`. Conditional on its
fixed simulation procedure, each is a deterministic function of the supplied
parameters. The known truth and selected posterior parameters therefore receive
the same function in the SBC comparison, including its randomized tie handling.
This checks a numerical approximation to the target; it is not a claim to know
exact expected points or arbitrarily close drivers' true ordering.

## Numerical precision and failed trials

The existing points-MCSE limit is inherited from
`f1rank.overall.simulation.MAX_EXPECTED_POINTS_MCSE`; it is not relaxed for this
work. The evaluator records estimated per-driver and paired MCSE. It also checks
a conservative bound before accepting the points integration: a season's average
GP points lie in `[0, B]`, where `B` is the maximum of the frozen points table.
For `R` independent seasons, the standard error is at most `B / (2 sqrt(R))`.
Indeed, `E[X²] <= B E[X]`, so `Var(X) <= E[X](B-E[X]) <= B²/4`.

Both the estimated per-driver error and this bound must meet the inherited
limit. A tiny sample with accidentally constant results therefore cannot count
as precise. The workload and seed must be fixed before evaluation; a failed
precision check does not trigger more simulation or a different seed. This new
development check does not retrospectively change any published gate. Its
resource cost must be measured before a complete study is frozen.

Precision in points does not guarantee precision in near-tie ranking indicators.
The report leaves `establishes_rank_precision` false. Full expected-point and
ranking recovery still require a registered numerical-sensitivity assessment,
appropriate study power and accepted posterior inference. MCMC integration
error is separate from the conditional simulation error reported here.

## Connection to recovery and preserved limits

An optional `ranking_functionals` section in a future recovery registration
contains the evaluator source hash and the protocol for every case. The complete
quantity inventory must include its finite-simulation quantities. The checked
consumer verifies the registered source and preserves the numerical reports for
the truth and every selected joint posterior draw. Missing reports, a different
protocol or failed points precision prevent a complete aggregate calibration
result. Existing parameter-only development checks cannot claim full recovery.

The existing simulator still rejects an outcome with no full-distance finisher;
this evaluator preserves that failure. It neither conditions it away nor replaces
the failed season. The common all-cause retirement hazard, lapping, shortened
events, correlated incidents and forecast-time conditions remain unfinished
scientific requirements. The complete recovery study, accepted-posterior
integration, chronological/prospective comparisons and publication checks also
remain open. No full study was launched by implementing this evaluator.

Software checks include an analytically solvable field, consistency with the
joint-posterior target adapter, permutation invariance, shared finite ties,
same-randomness monotonicity, paired numerical error and rejection of incomplete
or unregistered scenarios. Fixed synthetic inputs and cache-format receipts are
test fixtures, not driver evidence.

The [verification record](../outputs/championship/development/overall-ranking-recovery-checks/verification.json)
preserves the initial development checks, passing focused suite on committed
source and a capped local resource measurement. The workload uses the committed
field/calendar dimensions with arbitrary synthetic parameters, explicit full
racing distance and no retirements. It verifies actual scope limits and records
elapsed time and peak process memory. It does not measure a full posterior,
complete recovery study or intended CI runner. Hashed model/data sources and the
pending numerical proposal are unchanged; existing stale-publication failures
still require regeneration.
