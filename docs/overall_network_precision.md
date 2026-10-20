# Fixed-coordinate precision comparison

Investigate numerical precision after the saved-network diagnosis found severe
chain stagnation. This is a post hoc calculation comparison, not another fit or
a new acceptance gate. The exhausted profile and its failures remain unchanged.

Freeze the implementation, original research checkout and evidence identities
before preparing the workload. Use the original generating point followed by the
first and last saved point of every chain in every exhausted network attempt.
Regenerate the original observations and generating parameters once in the
original float32 environment and require their original fingerprints. Verify
each original archive and complete field inventory using the checked diagnostic
reader. Reconstruct unconstrained coordinates once in float32 and retain them
without alteration. This inverse transform cannot recover information lost when
the sampler originally saved constrained coordinates.

Persist those exact coordinates and observations in a numerical cache with a
recorded checksum. Both evaluation processes read that cache. The float64 process
lifts the same floating-point values; it does not redraw synthetic observations,
reinitialize parameters or independently invert the saved constrained values.
Discrete values, shapes and point labels remain fixed. Build the control rotation
from the original NumPy arrays before lifting anything and require its original
fingerprint. Conversion of these fixed coefficients to each execution precision
is part of the comparison; a different fitted or recomputed basis is forbidden.

Evaluate the joint potential and complete gradient in separate float32 and
float64 processes, together with every stochastic scalar's constrained value.
Require output dtypes to match the selected precision. The float32 reports must
be compared exactly with the prior saved-network diagnosis. Any mismatch remains
in the record and prevents treating the comparison as a reproduced baseline.

Retain every field's absolute difference, gradient difference norm, relative norm
when its denominator is nonzero, and largest-difference coordinate with its values
in both calculations. Include empty fields explicitly. Compare every fixed point;
do not select a point, parameter or threshold after seeing differences. These are
descriptive numerical discrepancies, with no pass/fail tolerance, inferred
posterior accuracy or claim that float64 would converge.

Use sequential capped analysis scopes with the standing memory, CPU and priority
limits. Keep the same explicit numerical environment except for the required
JAX precision flag. Record progress, failures and separate resource measurements
for preparation, both evaluations and comparison. Immutable cache hashes bind the
complete numeric inputs and raw gradients, which are diagnostic data rather than
publication posteriors. No MCMC, prior, likelihood, retry rule, gate, ranking or
public pointer changes follow from this investigation. Any subsequent fitting
proposal needs its own reviewed and frozen procedure.
