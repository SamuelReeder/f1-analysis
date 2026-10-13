# Complete-design recovery preflight: result

Every required design passed the fixed density-equivalence and finite-gradient
preflight under the declared local resource caps. This includes the complete
real transfer-network design. All original latent directions and observed
probability factors were retained. This is one supplied generating parameter
per design; no posterior was fitted and no calibration or ranking accepted.

The [definition](overall_recovery_preflight.md),
[corrected frozen protocol](overall_recovery_preflight_v2_protocol.json),
[complete review](../outputs/championship/development/overall-recovery-preflight-v2/review.json)
and [raw summary](../outputs/championship/development/overall-recovery-preflight-v2/summary.json)
record the complete inventory and its source identities.

## Complete inventory and resources

Every row completed all density, original-state, observation-factor and full
unconstrained-gradient checks. Original, transformed and gradient coordinate
counts agree in each case.

| Design | Qualifying observations | Lap observations | Latent coordinates | Compiled value/gradient seconds | Child monotonic seconds | GNU wall time | Process peak RSS (KiB) |
|---|---:|---:|---:|---:|---:|---|---:|
| connected | 216 | 1,152 | 903 | 3.587 | 18.186 | 0:17.52 | 930,796 |
| confounded | 216 | 1,152 | 897 | 3.636 | 17.853 | 0:16.70 | 923,060 |
| sparse_rookie | 183 | 992 | 849 | 3.589 | 18.485 | 0:17.84 | 940,676 |
| dominant_teammate | 216 | 1,152 | 903 | 3.569 | 18.069 | 0:16.92 | 923,640 |
| era_boundary | 216 | 1,152 | 909 | 3.641 | 18.075 | 0:17.40 | 927,920 |
| real_network | 15,656 | 122,556 | 59,466 | 19.283 | 43.103 | 0:31.90 | 1,768,420 |

Each independent child verified its actual `f1.slice` scope, MemoryMax 3G,
CPUQuota 200% and nice 19 before numerical work. Cases were processed
sequentially. The original shared slice configuration was not modified.
Raw process resource output is retained separately for every row.

The compiled phase includes compilation and evaluation of one full potential
and gradient. Its duration is not a per-transition sampler measurement. Process
RSS describes this preflight, not multi-chain fitting or posterior archive work.
Child monotonic durations disagree with GNU wall durations. Both clocks remain
in the record without an established cause or an extrapolated full-fit runtime.
The real-network measurement retains the complete original design and its
conditioning assumptions; it does not imply that real driver effects have been
identified or that the model describes real racing adequately.

## Probability checks and preserved failure

The constrained joint density includes the unchanged original priors and
likelihood. Its comparison includes the existing lap-control transformation's
log absolute Jacobian; the conditional weekend transformation contributes zero.
Every original state and each observed log-probability factor was compared using
the previously specified tolerances. The full compiled gradient used the
installed NumPyro support transforms and included every unconstrained coordinate.
Complete shapes, fingerprints, discrepancies and runtime identities are in the
per-case result files.

The [initial run](../outputs/championship/development/overall-recovery-preflight-v1/summary.json)
failed while recording generated-array metadata, before density or gradient
checks. The recorder recognized NumPy arrays but attempted to serialize JAX
arrays directly. All child failures and logs remain preserved. The correction
records shape, dtype and byte count for both array types without serializing
response values. The separate corrected registration keeps the original
cases, generation seeds, model, coordinate method, tolerances and caps. No
scientific criterion was changed in response to a result.

The corrected focused software suite passes 41 tests, including reproduction
of the actual generated-array metadata failure, source/protocol boundaries,
complete failure accounting and the original coordinate-equivalence checks.
The earlier 40-test run overlaps and is not added to that count. Both raw outputs
and source identities are retained in the
[software verification](../outputs/championship/development/overall-recovery-preflight-checks/correction-verification.json).
These checks are not a new full repository test run or production acceptance.

## Remaining work

The existing successful numerical profile establishes convergence only for its
original registered generated datasets. This preflight adds necessary full-design
computation checks; it does not establish convergence for the dominant-teammate,
regulation-boundary or real-network designs. Their actual inference cost and
behavior remain unmeasured. No failed fit or historical result has been relabeled.

Full recovery and simulation-based calibration require a complete frozen study,
justified size and power, independent generation/sampling streams, all required
quantity and ranking checks, and complete failure accounting. The already
inspected preflight datasets cannot be called untouched calibration evidence.
Functional MCMC and near-tie precision, full nested-simulation resources,
scientific weekend assumptions, predictive validation and production integration
also remain required. No new ranking or public payload was produced.
