# Model attempts through separate checkpoint workers

The development adapter in `analysis/checkpoint_attempt/` connects the verified
chunk sampler, bounded final archives, CPU diagnostics and sequential executor.
The fixed retry controller retains the existing fit settings, seeds, numerical
thresholds and retry ladder. It advances only after complete failed numerical
diagnostics. A worker crash, missing output or changed artifact is an execution
failure and cannot select another numerical attempt. Every failed posterior
remains in its own attempt directory.

The sampling worker preserves the declared complete collector and sampler
extras. Its final result binds the frozen attempt specification, checkpoint tip,
runtime identity, sealed inventory and numeric archive hashes. If interruption
occurs after archive sealing but before the result was written, recovery compares
every archived value with its original checkpoint chunks before reconciling the
artifact. Completed stages are verified without running their numerical work.
The CPU diagnostic worker starts after the sampling interpreter has exited.
It reads the same sealed artifact, retains every parameter and current-driver
contrast report, and recounts sampler extras. The validator checks diagnostic
site coverage and recomputes the existing gate from the saved metric summaries.
It does not replace failed diagnostics with another criterion.

The [attempt adapter checks](../outputs/championship/development/overall-checkpoint-attempt-checks-v1/stdout.txt)
pass **13** tests, including separate real sampler/archive and diagnostic workers,
interrupted sealing, altered numeric bytes, diagnostic coverage and fixed retry
selection. The [full-model binding checks](../outputs/championship/development/overall-checkpoint-attempt-profile-checks-v1/stdout.txt)
pass **7** tests for complete collector axes, unchanged settings/gates, explicit
worker environments, unique attempt paths and authoritative prior-service state.
Both recorded executions confirm their checked source files were unchanged.
These are focused engineering checks. A new [computational registration](overall_checkpoint_attempt_profile.md)
is required before full-model fitting. The fixtures cannot establish convergence,
recovery, calibration or production turnaround. The existing scientific registrations and public ranking remain
unchanged. Full synthetic fitting, complete calibration, predictive evaluation,
canonical publication and the measured end-to-end refresh are still required by
`overall_model_plan.md`.
