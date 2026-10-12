# Expected-point ordering: simulation sensitivity

The new diagnostic reports how inner simulation error can affect expected-point
comparisons, while retaining every chain, draw and driver. It reads a completed
grouped simulation companion and binds its source and input fingerprints. It
does not establish posterior convergence or permit a ranking to be published.

The [definition](overall_rank_sensitivity.md),
[implementation](../analysis/overall_rank_sensitivity.py) and
[verification](../outputs/championship/development/overall-rank-sensitivity-checks/verification.json)
record the scope and evidence. The focused suite passes 84 tests, including
exact binomial tail calculations, every vertex of a grouped interval box,
complete ties, field permutations, resolved comparisons and the checked
controller-to-diagnostic path. The earlier overlapping suite is preserved;
its count must not be added to the final suite's count.

## Saved synthetic examples

The [fixed audit](../outputs/championship/development/overall-rank-sensitivity-checks/audit/audit.json)
reuses both original saved cases and their complete traces. It generates no new
observations or simulation replicates. The declared error probability comes from
the existing contract before execution, and the original replication is retained.

| Saved case | Exact-mean checker | Noisy-mean checker | Conditional ordering sensitivity |
|---|---|---|---|
| separated_chain | failed | passed | unresolved |
| mixed_control | passed | passed | unresolved |

Each saved example has a simultaneous family of 6,400 baseline means. The
pointwise radius is 0.121242 points per Grand Prix under the declared family
bound. Both examples retain pairwise empirical-fraction bounds from 0 to 1,
in each chain and the complete trace. The known exact means fall inside the
intervals in these examples; that observation is not a proof of general coverage.
The mathematical bound requires independent replicated seasons conditional on
each fixed draw and the stated scoring bounds.

The mixed control's unresolved ordering is not a newly declared convergence
failure. Numerical convergence of a trace and sensitivity of a near-tie ordering
to inner simulation error answer different questions. Conversely, the separated
case's passing noisy checker still cannot accept its failed exact trace. No
threshold, seed, replicate count or original verdict was changed.

These bounds are conservative. A wide result records a limitation of the fixed
workload; it is not permission to increase replication after inspecting results.
Every report keeps functional MCMC precision, ranking precision, scientific
acceptance and publication flags false. Expected-point ordering remains distinct
from sporting ranks and title probabilities in simulated seasons.

## Local resource check and remaining work

The diagnostic also processes the largest dimensions in the existing grouped
storage fixture: 4 chains, 3,200 draws per chain and 22 anonymous entrants. The
baseline and required explanation means form a family of 1,126,400 quantities.
The reported replicate count is an input to the bounds; those seasons were not
simulated in this measurement.

Input construction and diagnostic computation take 0.259 recorded seconds. The
whole audit process reports maximum resident memory of 256,220 KiB under the
3G memory and 200% CPU scope caps. These are local synthetic diagnostic resources,
not full posterior simulation, fitting or intended-runner measurements.

Functional MCMC diagnostics still require a registered procedure that accounts
for noisy conditional means. Full nested-simulation precision and resources,
known-truth recovery and calibration, completion/shortened-event assumptions,
scientific validation and accepted producer integration remain open. This work
changes none of the production model, input data or pending weekend-coordinate
proposal, and launches no fit.
