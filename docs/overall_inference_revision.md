# Proposed next numerical experiment

The original [profile](overall_profile_results.md) remains failed. The proposed
next experiment tests equivalent sampling coordinates for the lap-control
parameters. It changes no prior, likelihood, driver score, convergence threshold,
sampling seed or retry limit. The user approved the separate experiment on
2026-10-12. The [frozen registration](overall_rotation_profile_protocol.json)
preserves the prepared inputs, settings, gates and inventory. The original
[proposal](overall_rotation_profile_proposal.json) retains its historical pending
state and cannot itself pass the runner's registration checks. Approval to run
the experiment does not establish a successful correction or authorize a ranking.

## Evidence for investigating the sampling coordinates

The [geometry diagnosis](../outputs/championship/development/overall-profile-geometry-v1/review.json)
reproduces the original synthetic observations and truths exactly, then records
the generating scales and saved control correlations for every attempt and chain.
It also checks that the original probability-model and generator sources match.
It does not fit the model or estimate real drivers.

Lap trend, compound effects and tyre-age slopes are strongly coupled in the
synthetic designs. The controls' local curvature differs across directions,
which is consistent with difficult sampling. This is a diagnosis, not proof of
the failure's cause. The full matrices are retained; a numerical design-rank
calculation on rounded inputs is not evidence of scientific identification.
The confounded profile's failure and the diagnostic failures in driver contrasts
are unchanged. Choosing a favorable earlier retry remains prohibited.

## Exact change being proposed

For each race, let `b` contain all its fixed-scale Gaussian lap controls, `D` the
diagonal matrix of their existing prior scales, and `X` their design matrix. The
existing prior is `b = D z`, where `z` has independent standard Normal components.
Compute an orthogonal eigenbasis `V` of `(X D)' (X D)` and instead sample
`z_new` from the same standard Normal prior, with `b = D V z_new`.

An orthogonal rotation preserves the standard Normal distribution. Consequently
the original `b` prior and likelihood remain the same. The change-of-variables
log Jacobian is the sum of the logs of the existing prior scales. No direction
is removed, even when the design contains unused or collinear columns. The
original control names are recovered as deterministic outputs, and all other
latent states and observation models are unchanged.

The basis uses design columns and fixed prior scales only. It does not use lap
responses, generated truths, posterior samples, diagnostic magnitudes or driver
rankings. There is no choice among transformations after a new fit. This follows
the general purpose of
[NumPyro reparameterizers](https://num.pyro.ai/en/stable/reparam.html): changing
sampling geometry while retaining the intended distribution. The prototype is
[overall_control_rotation.py](../analysis/overall_control_rotation.py).

## Experiment boundaries

The proposal retains the original study inventory, synthetic seeds and exact
observation-array hashes. The old generator supplies the same observations and
truths; changing sampling coordinates must not redraw an easier problem. All
original numerical settings and diagnostics remain fixed, including the retry
ladder. The checker will examine both recovered original controls and the new
coordinates. Convergence in a new run would not reclassify the old run.

The [runner](../analysis/overall_rotation_profile.py) checks committed source,
model/data fingerprints and the parent registration. It uses a new output/cache
identity and refuses an existing experiment directory. Every study is accounted
for even if another fails. Failed draws stay unavailable to production. Source
changes, exceptions and interrupted runs require inspection; they do not grant
an automatic settings change or another seed search.

If approved, the experiment uses the resource caps recorded in the proposal and
the existing shared CPU slice. Progress monitoring follows the user's requested
interval with one waiter. No real-data fit, driver table, push or publication is
authorized by this experiment, regardless of its result.

The mathematical checks cover the original control covariance, invertibility,
unused directions, response-independent construction, full joint density after
the Jacobian adjustment, observation-factor agreement, and finite gradients.
Registration checks reject missing approval and altered seeds, settings, gates
or model inputs. Their [verification record](../outputs/championship/development/overall-control-rotation-checks/verification.json)
identifies the tested source and passing reports. These are software and equivalence checks;
actual convergence, complete recovery and predictive validation remain open.

The later [race-order arithmetic correction](overall_candidate.md#race-order-arithmetic-correction)
changes the successor's evaluated likelihood implementation and source identity,
while retaining the specified Plackett–Luce distribution. The proposal remains
pending approval with that source identity. Its input audit must reproduce the
original observations and truths exactly. A future run would evaluate the
corrected arithmetic and rotated coordinates together; it could not attribute a
convergence change to either one alone or relabel the original failed profile.
The [new input audit](../outputs/championship/development/overall-order-numerics-checks/input-audit.json)
confirms that the corrected source reproduces the registered observations and
truths. Historical geometry and grid-audit programs deliberately require their
original model source; their committed reports remain records of those revisions.

## Runner accounting before execution

The pending runner must enforce the matched-truth condition directly, in addition
to the earlier input audits. Bind the original profile's committed truth files
in the proposed registration and compare the complete generated truth record
before starting each study. A changed or missing truth must stop that study
before sampling; it cannot be replaced by a new generating value.

For every returned sampling attempt, attach the unchanged diagnostic check for
every current-field qualifying and race contrast, retaining chain means and
within-chain standard deviations. Include failed attempts as well as successful
ones. These are the comparison reports already promised by the proposal. They
do not replace or modify the original attempt's convergence decision, trigger
additional retries, or supply a new acceptance criterion. Preserve a separate
failure in these reported diagnostics even if the original parameter check
passes. Software fixtures for this reporting path are not MCMC results.

These safeguards change the runner's implementation fingerprint. They do not
change the model input identity, study inventory, coordinates, truths, seeds,
sampling settings, numerical gates or pending approval status. No experiment is
launched by implementing them.
