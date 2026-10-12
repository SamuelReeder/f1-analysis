# Development recovery and calibration checks

`analysis/overall_recovery_checks.py` prepares known-truth evaluation for the
unfinished [Overall plan](overall_model_plan.md). It has no fitting entrypoint,
does not freeze a full study, and cannot authorize a ranking. The original
numerical-profile failures remain recorded, and the separate coordinate-study
proposal still awaits approval.

## Quantities and evidence boundaries

The evaluator observes the existing synthetic generator's latent draws without
adding random sites or changing its responses. It checks current-field pairwise
driver, teammate and car contrasts, context coefficients, each observation
likelihood, and the joint observation likelihood. Priors are excluded from that
last quantity. Every quantity is evaluated on the same generated responses at
the generating parameters and at posterior draws. Missing parameters cannot be
silently redrawn, and invalid draws cannot be discarded.

Parameter ranks alone can miss an algorithm that returns the prior regardless
of the data. Data-dependent quantities help detect that failure. The exact
binary example in the tests demonstrates this distinction without relying on a
randomized test outcome. This choice follows
[Modrák and colleagues](https://doi.org/10.1214/23-BA1404).

The dominant-teammate generator fixes latent values, so its role is a stress
test. It is not included in tests against uniform prior-predictive ranks. The
other generator cases draw their parameters from the model prior, conditional
on the specified network, observation availability, race-order panel and
exposure. These are checks of conditional inference; they do not validate an
end-to-end weekend model or the real-world adequacy of those conditioning
assumptions. All cases remain in the required inventory.

All-draw state interval summaries retain same-draw contrasts and the existing
`recovery_metrics` interval convention. Coverage for a fixed stress truth is
descriptive; nominal average coverage requires a suitable ensemble. Likelihood
quantities are reported through their ranks rather than state intervals.

## Rank checks and complete accounting

Following the SBC construction in
[Talts and colleagues](https://arxiv.org/abs/1804.06788), a generating truth and
exact independent posterior draws have a uniform discrete rank under the
prior-predictive experiment. Correlated iterations within a chain do not supply
that reference distribution. This evaluator selects one draw per independent
chain, independently of its values, and shares the selected indices across all
quantities. Reproducible tie randomization includes the truth in the tied group.

This deliberately coarse rank policy does not establish adequate power or
independent, well-mixed chains. A full registration must justify its study size
and sensitivity before fitting. Passing convergence diagnostics does not prove
exact posterior sampling. Likelihood quantities improve the range of checks;
they do not detect every possible failure.

The checked archive consumer requires a committed, frozen recovery-calibration
registration, a complete case/quantity inventory, and the registered evaluator
source. It verifies the generated-array identity, experiment specification,
settings, catalogue, run key, current model source and the complete compressed
posterior through the shared archive reader. Registration, evaluator, receipt
and archive changes during consumption prevent a successful result. The direct
arithmetic interface is explicitly unverified.

The summary requires every declared replicate. Missing, running, failed or
unverified results prevent aggregate calibration tests on a selected subset.
For each prior-predictive case and quantity, it tests each nontrivial discrete
rank-CDF cutoff with an exact two-sided
[binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
The replicate is the sample unit; correlated driver pairs are not pooled into
extra trials. A fixed Bonferroni family spans the declared cases, quantities and
cutoffs, using a family error probability supplied by the future registration.
Non-rejection is reported as no detected miscalibration, never full recovery or
publication acceptance.

## Work still required

The actual recovery-calibration study has not been registered or run. Its
replicate counts, generation seeds, error budget, power justification, model and
sampler sources, failure accounting and ranking functionals must be fixed before
outcomes are examined. The pending numerical experiment cannot substitute for
that study. Resource measurements, the unresolved numerical case, expected-point
and ranking recovery, chronological predictive comparisons and prospective
evidence remain required by the plan.

Software fixtures use explicit generated values and cache-format receipts. They
are not fitted evidence, including when a fixture exercises the successful
archive-reader path. Existing stale-publication failures also remain open; this
development evaluator does not regenerate the published payload.

The [verification record](../outputs/championship/development/overall-recovery-checks/verification.json)
links the preserved initial development output and passing focused checks against
committed source. The latter covers the evaluator, shared archive reader and fit
interfaces. Hashed model/data inputs and the pending numerical proposal are
unchanged. The full local suite was not repeated for this isolated addition;
its previously recorded stale-export failures remain explicit rollout blockers.
