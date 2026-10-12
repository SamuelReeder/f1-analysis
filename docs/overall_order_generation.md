# Preserving race-order perturbations

The race-order generator now preserves the actual Gumbel perturbations when
ranking entrants. The original expression added each perturbation to an absolute
strength and then sorted the rounded sum. At large common strengths, the added
variation could disappear. This affected the joint likelihood's synthetic
generator, the conditional forecast kernel and the equal-opportunity race
simulation. It is distinct from the earlier probability-normalizer correction.

The implementation in [`order.py`](../f1rank/overall/order.py) retains the rounded
sum and its roundoff term, sorting lexicographically by those components. It uses
the TwoSum construction described in
[Shewchuk's arithmetic paper](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf).
This avoids changing the random stream or replacing the lost draw with a new
tie-breaking draw. Merely subtracting a global maximum is insufficient when a
lower group also contains tied strengths far below that maximum.

The JAX generator uses
[optimization barriers](https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html)
to preserve the intermediate rounding steps under compilation. It handles native
leading batch axes; the rank generator is discrete and is not differentiated.
The fitted likelihood continues to use its separately tested relative suffix
normalizers. No new random site, prior, numerical gate or retry choice is added.

Excluded entrants are placed last. Exact mathematical ties retain the caller's
declared tie convention: stable input order in the joint generator and
counterfactual simulator, and the existing identity-keyed tie uniforms in the
conditional forecast. Those uniforms no longer replace a distinct Gumbel order
merely because its rounded absolute sums coincide.

The NumPy path rejects nonfinite included inputs or compensated-sum overflow.
The compiled JAX path emits invalid indices for that row, and the recovery
consumer rejects them before indexing observations. Ignoring a sentinel and
allowing negative indices to select an entrant would manufacture an observation.

These helpers preserve supplied floating-point inputs; they cannot recover
information already lost when latent strengths were constructed. Tests exercise
normal arithmetic scales on the local CPU, including compiled JAX, and compare
with exact rational sums of the supplied inputs. They do not establish arbitrary
precision for the entire model or behavior on every accelerator/backend.

The regression fixtures exercise the actual joint generator, conditional
forecast and counterfactual race points. Additional checks cover separated tied
groups, masks, permutations, exact tie conventions, generated first-place
frequencies, invalid arithmetic and the recovery consumer's rejection path.
The original numerical-profile inputs must still match their frozen identities.
The profile failure remains recorded, the proposed sampling study remains
unapproved, and full recovery, calibration and production integration remain open.

The [verification record](../outputs/championship/development/overall-order-generation-checks/verification.json)
identifies the tested source, preserved failure reports, full local suite and
compressed synthetic workload. The [input audit](../outputs/championship/development/overall-order-generation-checks/input-audit.json)
reproduces the original registered observations and truths exactly. The resource
workload tests the local forecast path with synthetic parameters; it does not
measure MCMC fitting or establish readiness on the intended CI runner.
