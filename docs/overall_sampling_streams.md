# Separate sampling streams for a future calibration study

The existing numerical profile deliberately keeps the same fixed sampling
settings and retry seeds for each design. It is a computational experiment,
not a simulation-based calibration study. Its exhausted case and original
registration remain unchanged.

A full calibration study needs independent replicate randomness as well as
separate generating truths. Repeating the same posterior draws across trials
can break the uniform-rank reference even for a simple correct sampler. The
exact Uniform example in `tests/test_overall_sampling_streams.py` shows that,
conditional on a reused set of draws, rank probabilities are their interval
spacings. Independent replicate draws average over those spacings. The
[Stan SBC implementation](https://mc-stan.org/rstan/reference/sbc.html) likewise
uses a sequence of sampling seeds to preserve independence between runs.

## Before any full calibration fits

The fitter now requires a `sampling_seeds` table for a new
`recovery_calibration` registration. It names every study and every attempt in
the unchanged retry ladder. All generation and sampling keys must be distinct,
valid unsigned sampler keys; omissions, reuse and integer wraparound fail
before fitting or cache creation. These separate pseudorandom streams are an
implementation requirement, not proof of accurate inference.

The declared first seed becomes that study's base settings and cache identity.
Every attempt records its actual declared seed. The complete frozen registration
hash protects unused later attempts too. Changing the registration during a run
records an exception; changing it before loading a cache cannot relabel that
cache. The checked recovery consumer uses the same per-study settings when
verifying the archive.

The ordinary numerical-profile path retains its original schedule. Supplying
the new table to another registration purpose is rejected. Warm-up, draw counts,
chain count, target acceptance, priors, likelihoods, convergence gates and retry
limits are unchanged. This is support for a future fully registered study;
there is no search for seeds after inspecting its results. No calibration
registration or fit is created by this change.

## Evidence and remaining work

The failing runner regression preserves the earlier behavior with an explicit
sampling stub, not MCMC. Corrected checks cover complete retry execution,
collision/range rejection, immutable caches and the original profile path.
Source hashes change, so affected model/publication artifacts require fresh
evidence. Original synthetic observations and truths must still reproduce
exactly for the separate pending coordinate study; its approval remains pending.

This fixes a prerequisite of the proposed power calculation. Full study size,
power against declared error patterns, numerical-sensitivity assessment,
complete registration and actual inference remain unfinished. The development
fixtures cannot establish calibration, ranking recovery or publication readiness.

The [verification record](../outputs/championship/development/overall-sampling-stream-checks/verification.json)
links the failing regression, passing focused checks, full local suite and exact
original-input audit. The full suite still fails the previously recorded
stale-qualifying-export integration checks. The sampler implementation, base
settings, probability-model sources and gates are verified unchanged; only the
new calibration path uses a separately declared seed inventory. The original
profile's observations and truths reproduce exactly. Its failure is preserved,
and the updated pending source identity grants no approval.
