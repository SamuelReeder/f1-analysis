# Matched precision comparison: substantial discrepancies at stuck-chain points

The [frozen numerical comparison](overall_network_precision_protocol.json)
completed on the original generating point and every specified failed-chain
endpoint. The original float32 potential and full-gradient reports reproduce
exactly. Changing execution precision while holding the unconstrained coordinates,
observations and original control rotation fixed produces substantial discrepancies
at some of the stuck-chain points. This supplies a concrete numerical issue to
investigate; it does not yet establish which calculation is accurate or why the
original sampler stalled.

The [verification record](../outputs/championship/development/overall-network-precision-v1-execution/verification.json)
retains all **33** points and **59,466** gradient coordinates per point, with the
complete differences in the [comparison result](../outputs/championship/development/overall-network-precision-v1/compare/result.json).
The table summarizes every point in each group. Relative gradient difference
means the Euclidean norm of the full gradient difference divided by the float64
gradient norm; it is a descriptive discrepancy, not an acceptance criterion.

| Point group | Maximum absolute potential difference | Maximum relative full-gradient difference | Maximum absolute gradient-coordinate difference |
|---|---:|---:|---:|
| Original generating point | 0.009428 | 0.000038699 | 0.083927 |
| Attempt 1 endpoints | 803.866138 | 0.063068702 | 907.734059 |
| Attempt 2 endpoints | 340.722742 | 0.006235834 | 102.465796 |
| Attempt 3 endpoints | 178.107690 | 0.025671406 | 79.524750 |
| Attempt 4 endpoints | 0.298364 | 0.000428601 | 2.637552 |

The earlier computation-equivalence audit tested the generating points and
established agreement between coordinate implementations at those points. It
did not establish precision throughout a sampling trajectory. The new results
show why that distinction matters.

## A specific calculation to investigate

The largest derivative discrepancies in the earlier attempts involve the
qualifying and lap Student-t degrees-of-freedom parameters, `nu` and
`overall_lap_nu`. At the first saved point of the first chain of attempt **1**,
the reconstructed qualifying degrees of freedom are **74,545.4921875**, while
the original generating value is **8.039385795593262**. All values and other
parameters remain in the complete records; these failed-chain locations are not
posterior estimates.

The installed Student-t implementation subtracts log-gamma terms in its
normalizer, consistent with NumPyro's
[published implementation](https://num.pyro.ai/en/stable/_modules/numpyro/distributions/continuous.html#StudentT.log_prob).
This makes the normalizer and its derivative a specific next numerical target.
An algebraically equivalent formulation using the log-beta function is worth
checking against an independent high-precision reference; JAX documents
[`betaln`](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.special.betaln.html).
Neither the source inspection nor this whole-model comparison alone attributes
the full discrepancy to that calculation.

The next check should isolate the Student-t probability calculation, retain every
registered point, test values and derivatives against an independent reference,
and verify any equivalent implementation throughout the declared numerical domain
before integrating it. Preserve the same likelihood, priors, support and failed
experiments. A source correction would require fresh registered numerical and
scientific evidence before use in a published ranking.

## Resource record and limits

| Stage | Internal seconds | Separate external elapsed | Peak RSS (KiB) |
|---|---:|---:|---:|
| Prepare fixed inputs | 65.338 | 1:06.08 | 1,057,656 |
| Float32 evaluation | 8.347 | 0:08.19 | 747,376 |
| Float64 evaluation | 8.060 | 0:07.41 | 786,096 |
| Complete comparison | 2.671 | 0:02.90 | 91,144 |

Each process verifies its own active **3G** memory and **200%** CPU scope with
nice **19**, and checks that no sibling heavy job is active. The clocks remain
separate measurements. The [focused software record](../outputs/championship/development/overall-network-precision-checks-v1/verification.json)
records **26** passing checks. These are local calculation and software resources,
not full-fit or intended-runner measurements.

The fixed coordinates were reconstructed once from saved constrained values;
rounding can prevent recovery of the sampler's original unconstrained state.
The comparison does not supply the missing warmup trajectories, step sizes, mass
matrices or original sampler energies. It does not establish that float64 fitting
would converge or fit within the memory budget. No MCMC, model assumption, retry
rule, gate, public ranking or deployment changed. The [full plan](overall_model_plan.md)
remains incomplete.
