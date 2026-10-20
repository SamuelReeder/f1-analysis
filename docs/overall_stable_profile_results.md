# Corrected computational profile: preflight passed and fitting started

The [complete new preflight](../outputs/championship/development/overall-stable-preflight-v1/summary.json)
passed every original case. Each regenerated observation, parameter and truth
fingerprint matches the previously inspected problem. Every complete collector
inventory matches the original retained fields, shapes and dtypes. The coordinate
equivalence checks and complete generating-point gradients passed under the
existing criteria. These checks performed no MCMC.

| Case | Gradient coordinates | Internal seconds | Separate external elapsed | Peak RSS (KiB) |
|---|---:|---:|---:|---:|
| Connected | 903 | 16.227 | 0:16.81 | 926,396 |
| Confounded | 897 | 15.775 | 0:16.35 | 915,844 |
| Sparse rookie | 849 | 16.273 | 0:16.87 | 934,808 |
| Dominant teammate | 903 | 15.880 | 0:16.47 | 919,332 |
| Era boundary | 909 | 15.853 | 0:16.45 | 924,724 |
| Real network | 59,466 | 26.005 | 0:26.71 | 1,300,888 |

The per-case result and resource files linked from the summary supply these
separate timing measurements. Each case ran sequentially in the standing **3G**,
**200%** CPU analysis scope with nice **19**. The retained real-network inventory
contains **93** sites, with a maximum-attempt logical payload of **7,494,144,000**
bytes under the fixed ladder. Logical payload is not peak fitting memory or a
compressed-file estimate.

The [preflight/storage checks](../outputs/championship/development/overall-stable-preflight-checks-v1/pytest.txt)
record **33** passes, and the separate
[profile controller checks](../outputs/championship/development/overall-stable-profile-checks-v1/pytest.txt)
record **17** passes. They exercise inventory preservation, changed-input rejection,
failed or killed child accounting and refusal to replace existing runs.

## Frozen fit and launch

The [new registration](overall_stable_profile_protocol.json) binds the corrected
production method, complete current source/data identity, passing preflight,
runtime sources and original full case inventory. Generating seeds, sampling
seeds, fit settings, fixed retry ladder and numerical thresholds remain unchanged.
The prior exhausted profile remains failed and is not resumed.

The [initial admission record](../outputs/championship/development/overall-stable-profile-v1-launch/admission.json)
confirmed the new service running and the connected case in its fitting stage.
It verified the **13G** service memory cap, shared **400%** CPU cap, CPU weight
**10** and nice **19**. The [committed launcher](../outputs/championship/development/overall-stable-profile-v1-launch/launch.sh)
refuses to overlap active fit or heavy analysis units. A single locked waiter
checks at the user's **20-minute** interval and reports a failed case or terminal
service state. The admission record describes launch, not subsequent completion.

Do not change registered model, data, runtime or runner sources while this fit
runs. Preserve all attempts and source identities. Completed numerical results
will be recorded here after the run; no convergence or publication result is
claimed by this launch record. The [full plan](overall_model_plan.md), including
complete scientific validation, accepted canonical publication, refresh and
approved rollout, remains incomplete.
