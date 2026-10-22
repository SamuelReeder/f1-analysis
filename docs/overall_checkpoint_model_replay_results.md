# Complete-model replay execution status

The [registered engineering procedure](overall_checkpoint_model_replay_protocol.json)
has now completed its six declared designs. It is a short implementation check,
not a complete fit, convergence result or runtime acceptance benchmark.

## Initial service launch failed before sampling

The [initial execution](../outputs/championship/development/overall-checkpoint-model-replay-v1/execution.json)
failed at GPU admission because the systemd service's command path did not
resolve `nvidia-smi`. Its [terminal record](../outputs/championship/development/overall-checkpoint-model-replay-v1/terminal.json)
verifies the same service invocation had stopped, accounts for **12** failed
initial phases and **12** blocked dependent phases, and confirms that sampling
had not started and no checkpoint cache existed. Its complete
[comparison record](../outputs/championship/development/overall-checkpoint-model-replay-v1/comparison.json)
therefore reports every design as unavailable. This is an execution failure;
there are no convergence diagnostics or measured sampling costs from that launch.

## Corrected full-design engineering replay

The corrected [execution ledger](../outputs/championship/development/overall-checkpoint-model-replay-execution-v2/execution.json)
records all four expected phases for each of six cases: connected, confounded,
sparse rookie, dominant teammate, era boundary and the real network. The cases
cover uninterrupted sampling, intentional stops during warmup and retained
draws, and fresh-process resume. The [terminal record](../outputs/championship/development/overall-checkpoint-model-replay-execution-v2/terminal.json)
confirms every phase was processed and every worker exit matched its registered
outcome. The service unit and journal had already been collected when its final
runner exit was checked, so that outer exit remains unobserved.

The [numerical comparison](../outputs/championship/development/overall-checkpoint-model-replay-execution-v2/comparison.json)
passed for all six cases. Each case's posterior and sampler-extras archives have
matching field inventories, dtypes, shapes and bitwise values between the
uninterrupted and resumed result. The comparison run record and capped resource
measurement are saved [here](../outputs/championship/development/overall-checkpoint-model-replay-execution-v2/comparison-fixed-path-v1.json).
An earlier comparison invocation failed before numerical comparison because it
used a relative output path; that failure and its resource record are preserved
in the same execution directory. It did not change model code or settings.

This replay used the protocol's small fixed engineering workload. It did not run
a production-length fit, evaluate convergence, measure a representative refresh,
or authorize publication. The service terminal record explicitly keeps
`full_fit_executed`, `scientific_acceptance` and `publication_allowed` false.
The replay shows that the tested checkpoint/resume path preserves its declared
small-run values; it cannot show that a full update finishes by the next morning.
