# Full-network diagnosis: severe chain stagnation

The failed computational profile has a more specific problem than disagreement
between reasonably exploring driver/car estimates. Many saved parameter chains
barely move, and some scalar parameters are exactly constant within a chain.
This affects the separate retirement-risk block as well as qualifying and lap
parameters. The original failed convergence decisions remain unchanged.

The [complete diagnostic result](../outputs/championship/development/overall-network-diagnosis-v1/result.json)
and [compact verification record](../outputs/championship/development/overall-network-diagnosis-v1-execution/verification.json)
retain all **4** exhausted attempts, all **45** stochastic scalar parameters and
the first and last complete parameter points from every chain. The following
counts include every scalar parameter, using its saved floating-point values.
Chain positions follow their original archive order.

| Attempt | Exactly constant scalar parameters by chain | Distinct retirement-intercept values by chain |
|---|---|---|
| 1 | 28, 21, 20, 0 | 1, 1, 1, 789 |
| 2 | 18, 0, 26, 20 | 1, 1427, 1, 1 |
| 3 | 3, 1, 5, 9 | 267, 1139, 1, 1 |
| 4 | 0, 0, 5, 1 | 2, 3, 1, 2 |

An absence of exactly constant scalars does not establish useful movement. In
the final attempt, the retirement intercept has at most **3** distinct values
within each **3,200**-draw chain. The full record includes ranges, adjacent
movement and quarter means so this is not inferred from endpoints alone.

The lap autocorrelation is especially revealing. In attempt **3**, every chain
retains a single value throughout its **3,200** saved draws, close to the model's
upper bound of **0.95**. In the final attempt, the corresponding distinct-value
counts are **138, 226, 1, 1**. The original bound comes from the unchanged
[`overall_lap_rho` prior](../f1rank/overall/model.py). These observations suggest
investigating initialization, integration and numerical precision. They do not
prove that the prior bound is wrong or authorize changing it.

## Endpoint calculations and their limits

The original generating observations and parameters reproduce their fingerprints.
Every saved archive matches its recorded fingerprint and complete field
inventory. The equivalent batched model evaluates finite potentials and complete
gradients at all **32** reconstructed chain endpoints and at the original
generating point. These checks use every stochastic parameter coordinate, not
only the scalar subset used for movement descriptions.

The endpoints are reconstructed by applying the inverse transform to saved
constrained values. Floating-point rounding can prevent recovering the sampler's
original unconstrained state. These are therefore freshly evaluated gradients
at reconstructed points, not recovered sampler-gradient logs. Finite gradients
also do not establish good geometry or an adequate integration step size.

For context, the generating-point potential is **355,513.375**. In the first
attempt, final reconstructed potentials are **434,902.21875**, **423,005.875**,
**424,960.53125** and **352,612.59375**. The generating point is neither an expected
posterior mode nor a target energy. These disparate failed-chain locations,
together with the movement records, motivate investigation; no chain may be
selected as a replacement posterior.

The original run did not save warmup trajectories, adapted step sizes, mass
matrices or sampler extras. Recorded acceptance probabilities and tree-depth
saturation remain unverified metadata. The available traces cannot determine
whether finite precision, adaptation, initialization or geometry caused the
stagnation. Future numerical experiments should retain those sampler-state
diagnostics; NumPyro exposes energy and adaptation state through its
[documented HMC state](https://num.pyro.ai/en/stable/_modules/numpyro/infer/hmc.html).

## Measured execution and next investigation

The read-only diagnosis completed with **72.671 seconds** of internal elapsed
time and **1,248,080 KiB** peak resident memory. The separate external log records
**1:06.82** elapsed; the differing clocks remain separate measurements. The active
scope record confirms **3G** memory and **200%** CPU limits. The
[reader verification](../outputs/championship/development/overall-network-diagnosis-checks-v1/verification.json)
records **15** passing software checks. Its scope snapshot was taken after exit
and does not establish active properties.

Next, compare numerical precision at the same original observations and fixed
parameter points before considering another long fit. Preserve all points,
original failed attempts, model assumptions and acceptance rules. Freeze the
comparison definition before calculating its results; it must be a separately
identified numerical investigation, with no MCMC or ranking output. An observed
precision difference would still need a causal check before attributing the
sampling failure to it. The complete [scientific and production plan](overall_model_plan.md)
remains open.
