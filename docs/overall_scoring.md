# Overall predictive scoring contract

[`overall_scoring_contract.json`](overall_scoring_contract.json) fixes scoring
definitions before successor predictive results exist. It is not the complete
evaluation or prospective registration, and does not authorize real-data fitting
or publication. The frozen event inventory, model and baseline implementations,
forecast-time information and condition policy, calibration acceptance rules,
full recovery inventory and independent horizon remain required.

The implementation is
[`overall_predictive_scores.py`](../analysis/overall_predictive_scores.py).
It lives outside the model source while the registered computational profile
runs. It neither reads that run's posterior nor changes its inputs.

## Scores and shared uncertainty

The primary comparison evaluates the observed running-finisher order, conditional
on its declared information set, using posterior predictive log probability.
Higher is better. Likelihood is integrated over the posterior before taking its
log. Related conditional factors are multiplied within the same joint draw before
integration; multiplying separately averaged probabilities discards dependence.
Posterior, draw and event identities must match exactly. These interface checks
do not independently prove provenance; the future evaluation runner must verify
the actual saved draws and training inputs.

Ordinal diagnostics use log probability and the sum of squared differences
between predicted and observed cumulative probabilities. Lower ranked probability
score is better. Retirement diagnostics use Bernoulli log probability and squared
probability error; lower Brier score is better. These proper scoring rules follow
[Gneiting and Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

For discrete calibration, the probability integral transform uses the predicted
mass below the observed category plus a supplied uniform fraction of that
category's mass. The caller must record the randomization. This diagnostic does
not itself define an acceptance test or justify treating dependent drivers and
races as independent observations. No calibration success is claimed here.

Invalid probability vectors are rejected, rather than normalized to conceal
bad output. An impossible observed outcome retains a negative-infinite log score.
An infinite comparison cannot pass this contract; it is recorded as unresolved
without dropping its event or replacing it with a finite penalty.

## Complete chronological comparisons

Each comparison requires the full inventory and all declared arms. Every record
must match its event's target, forecast origin, strictly earlier training cutoff
and field. Missing, duplicate, failed or mismatched records prevent a complete
comparison. Different forecast origins cannot be pooled into a single result.
The running-finisher target still needs a separate completion/retirement check;
success on survivors does not establish the complete weekend prediction.

The primary baseline is the existing grid-plus-qualifying model. Grid-only and
current-race-only comparisons are descriptive and cannot rescue primary failure.
The season-block bootstrap and primary improvement rule are inherited unchanged
from `championship.paired_seasons`; a regression test compares their results
directly, including unequal event counts between seasons. This scoring contract
does not inherit the old candidate's future registration date or failed-fold
exclusions.

Even a complete primary comparison that passes returns
`publication_allowed: false`. Acceptance still requires the remaining numerical,
recovery, calibration, independent-evidence and production checks in the
[execution ledger](overall_execution.md). Synthetic examples test the arithmetic
and failure handling; they are not predictive results for real drivers.
