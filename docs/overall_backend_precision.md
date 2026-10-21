# Fixed-input backend precision and repeatability diagnosis

Investigate the completed CPU/GPU full-network gradient disagreement. This is a
post hoc numerical diagnosis, not another fit or a replacement acceptance gate.
The original screen and all failed coordinates remain unchanged. The independent
package under `analysis/backend_precision/` consumes the existing experiment;
it does not modify its runner, model, data, registered source inventory or runtime.

Freeze this implementation and every original preparation/evaluation/comparison
record before execution. Use every original case and fixed point, including
all archived failed attempts. Never select only coordinates that improve.
The [runner](../analysis/backend_precision/__init__.py) declares each execution
arm and every comparison pair. Re-evaluate the original single-precision CPU
and GPU calculations using the original evaluation function. Evaluate each
backend in double precision in separate fresh-process repetitions as well.

Reuse the same serialized observations and unconstrained parameter coordinates.
Lift their existing floating-point values without regenerating observations,
parameters, seeds or the inverse transform. Keep discrete arrays unchanged.
Build the original NumPy coordinate map before lifting data and require its
fingerprint. Its fixed coefficients are converted to the execution precision;
a new basis estimated from lifted or fitted data is forbidden.

The double-precision operation has the same potential, complete gradient and
complete retained-state outputs as the original evaluation. Require their
original shapes, declared execution precision and complete point inventory.
Record actual backend, precision, matrix-product configuration, package versions,
resources and stage timings. Explicitly unset the optional matrix-precision
override and require the default configuration. The original experiment did not
record this optional configuration; exact baseline reproduction must therefore
be tested rather than assumed.

Compare every field of the original and repeated single-precision calculations
both numerically and bitwise. Preserve mismatches, including run-to-run variation;
a mismatch prevents describing the original calculation as reproduced. Report
all declared within-backend, cross-backend and cross-precision differences, their
norms and largest-difference coordinates and values. Do not average repetitions
or replace an unfavorable run. Apply the original unchanged screen budgets to
the declared double-precision cross-backend pair as a descriptive check only.
Neither agreement nor higher precision establishes exact truth or posterior
accuracy. Every declared arm must have a terminal status, including failures.

Run sequentially under the existing capped analysis scopes, keeping the GPU pool
and admission rule unchanged. Separate compilation, execution, transfer, archive
and external process measurements. Fixed-point timings cannot establish sampler
mixing, fitting performance, or an overnight refresh. No MCMC, modified prior,
new model, relaxed gate, accepted ranking or publication follows from this study.
Any candidate arithmetic or precision change needs a separate prospective
implementation and sampling comparison after the diagnostic evidence is reviewed.
