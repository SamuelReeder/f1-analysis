# Grouped simulation moments and functional diagnostics

This definition precedes implementation and the fixed synthetic audit. It adds
diagnostic inputs for the existing canonical simulation, not a new model,
acceptance threshold, ranking product or permission to fit real data.

## Preserve the necessary quantities

After complete simulation and successful summary validation, retain chain and
within-chain draw axes for the per-driver expected-point estimates, unbiased
conditional season-point covariance, paired explanation-loss means and unbiased
conditional loss variances. All quantities use points per Grand Prix, with the
existing common race count. Save the full replicate counts, complete original
field order and explanation-axis labels. Dividing conditional covariance or
variance by the replicate count gives the estimated simulation-mean error;
the stored conditional season variance is not itself that mean error.

Write a numeric archive and its metadata only after the checked posterior reader
has exited successfully and the controller's source/input checks pass. Bind
the archive fingerprint, fit/posterior/scenario/simulation identities, field,
axes, units and sources in the completed summary and receipt. A partial or
failed execution cannot supply accepted diagnostic moments. The checked reader
must reject changed files, inconsistent identities, incomplete counts, unknown
members, invalid axes or nonfinite values. This verifies an artifact's internal
consistency; its existence does not authenticate scientific evidence.

The archive is a diagnostic companion, not a checkpoint or permission to resume
or retry a failed simulation. No draw, replicate, event or entrant is dropped.
No public dashboard payload is created by it.

## Why pooled results are insufficient

The existing numerical checker operates on grouped chains. Diagnostics for a
parameter need not describe a different nonlinear function of that parameter.
The primary references are the
[rank-normalized diagnostic paper](https://arxiv.org/abs/1903.08008) and
[Stan's posterior-analysis documentation](https://mc-stan.org/docs/reference-manual/analysis.html).
They motivate monitoring the quantities of interest; they do not validate this
project's particular nested simulator or its numerical acceptance procedure.

For a conditional function `f(theta)`, an independently simulated estimate can
be written `f(theta) + e`, with conditional mean-zero error. By total variance,
the marginal variance adds the expected conditional error variance. Independent
errors do not add corresponding cross-draw covariance. Thus extra simulation
noise can dilute the apparent dependence or chain separation in a trace. This
is an inference from the error decomposition, to be examined in the fixed
synthetic audit below, not a claim attributed to the references.

Do not promote the existing `mcmc_precision_verified` or
`rank_precision_established` flags merely because a checker passes on noisy
conditional means. Keep parameter convergence, conditional simulation error,
functional MCMC error and near-tie sensitivity separate. Grouped moments enable
further analysis; a complete registered functional-precision procedure and its
validation remain required before scientific acceptance or publication.

## Fixed synthetic noise audit

Use the initial chain/draw shape from the committed fitting settings, the
unchanged numerical checker and the existing paired-difference range bound to
choose the fixed replicate count. Construct an exactly solvable strict race
order between anonymous entrants: the first driver wins with probability `p`,
otherwise the second wins. Award the original points table's winning and
runner-up scores. The exact conditional means follow directly from `p`, and
binomial sampling generates the same count law as independent repeated races.
This is a synthetic diagnostic fixture, not a canonical-simulator run or a fit.

Include both a deliberately separated-chain probability trace and a mixed
control. Freeze all trace formulas and random seeds in
`analysis/overall_functional_noise_audit.py` before execution. Compare the
original checker on exact conditional means and their finite-simulation
estimates. Record both outcomes, conditional variances, error bounds and input
fingerprints. Keep all cases even if the expected masking pattern is absent.
Do not change a seed, trace scale, sample count or threshold after evaluation.
No result reclassifies the original failed numerical profile or authorizes the
pending coordinate experiment.
