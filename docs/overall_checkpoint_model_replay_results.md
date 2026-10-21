# Complete-model replay execution status

The [registered engineering procedure](overall_checkpoint_model_replay_protocol.json)
has not established full-model checkpoint replay yet. It is a short implementation
check, not a complete fit, convergence result or runtime acceptance benchmark.

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

## Corrected service environment

The correction explicitly includes `/usr/lib/wsl/lib` in the service's command
path. A separate capped [service preflight](../outputs/championship/development/overall-checkpoint-service-preflight-v1/run.json)
then succeeded. Its [output](../outputs/championship/development/overall-checkpoint-service-preflight-v1/stdout.txt)
verifies utility resolution, actual GPU backend, double precision, registered
allocator admission and host resource limits inside a systemd service.

The [subsequent launch](../outputs/championship/development/overall-checkpoint-model-replay-launch-v2/launch.json)
uses the same frozen worker code, model, observations, engineering settings and
comparison rule in a new execution directory. It records the original failed
execution and successful service preflight. The original failure is preserved.
No model or fitting-source edit was required. The new execution still requires
terminal accounting, committed raw artifacts and complete numerical comparison.

A successful future replay would establish engineering behavior for the declared
workload. Complete fitting, runtime measurement, scientific acceptance and the
next-morning refresh remain separate outstanding requirements.
