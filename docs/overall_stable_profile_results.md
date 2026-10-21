# Corrected computational profile: fitting completed

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

## Completed computation and audit status

The [terminal summary](../outputs/championship/development/overall-stable-profile-v1/summary.json)
records every registered case as converged. The
[terminal observation](../outputs/championship/development/overall-stable-profile-v1-terminal/observation.json)
preserves the launcher completion and original output inventory. The
[metadata review](../outputs/championship/development/overall-stable-profile-review-v1/review.json)
finds the complete records consistent, including failed attempts. The
[independent saved-array audit](../outputs/championship/development/overall-stable-profile-diagnostics-v1/audit.json)
reproduces every attempt's complete parameter and current-driver contrast
reports exactly. It also recounts the saved divergences, acceptance probabilities
and tree-depth-limit fractions, with no discrepancies or changed input files.
This preserves the original failed attempts as well as the successful retries.
The following tables report the original committed fit outputs.

This is a computational study with synthetic observations and known truth.
The real-network case uses the historical driver/team structure but synthetic
responses; its named drivers are not actual driver-rating results. Full recovery,
simulation-based calibration, predictive evaluation and canonical publication
remain incomplete. No Overall ranking can be published from this profile.

## Recorded runtime

The per-case `result.json` and adjacent `<case>.resources.txt` files supply the
following measurements. Internal duration includes the case's fitting, retries,
diagnostics and result work. External elapsed is the separate process timer.

| Case | Internal seconds | External elapsed | Peak RSS (KiB) |
|---|---:|---:|---:|
| [connected](../outputs/championship/development/overall-stable-profile-v1/connected/result.json) | 401.771 | 6:42.76 | 1,671,036 |
| [confounded](../outputs/championship/development/overall-stable-profile-v1/confounded/result.json) | 251.938 | 4:13.05 | 1,648,440 |
| [sparse_rookie](../outputs/championship/development/overall-stable-profile-v1/sparse_rookie/result.json) | 297.603 | 4:58.81 | 1,678,572 |
| [dominant_teammate](../outputs/championship/development/overall-stable-profile-v1/dominant_teammate/result.json) | 99.669 | 1:40.60 | 1,651,808 |
| [era_boundary](../outputs/championship/development/overall-stable-profile-v1/era_boundary/result.json) | 88.705 | 1:29.22 | 1,664,320 |
| [real_network](../outputs/championship/development/overall-stable-profile-v1/real_network/result.json) | 110,272.281 | 29:04:57 | 12,766,068 |

The [launcher resource record](../outputs/championship/development/overall-stable-profile-v1-launch/resources.txt)
reports external elapsed **29:24:05** and peak RSS **12,766,068 KiB**. Internal
and external clocks disagree, particularly for the real-network case; they are
preserved separately and are not a matched performance comparison. The launch
allocation remained the registered **13G** memory cap and shared **400%** CPU cap.
This workload does not measure an accepted post-race refresh or establish
next-morning publication.

## Every attempt and retry

These values come from each linked result's `receipt.attempts[].diagnostics`.
The registered rank R-hat, effective sample size, Monte Carlo error and divergence
rules were unchanged. Extremes below summarize all reported parameter families;
the full reports retain each parameter family and current-driver contrast.
Failed attempts remain failed even when a later registered retry converged.

| Case / attempt | Outcome | Max R-hat | Min bulk ESS | Min tail ESS | Max MCSE/SD | Divergences | Tree-limit fraction | Internal attempt seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| connected / 1 | failed | 1.011439 | 746.330 | 597.775 | 0.036864 | 0 | 0.00 | 145.751 |
| connected / 2 | converged | 1.005745 | 1348.960 | 813.097 | 0.026847 | 0 | 0.00 | 245.567 |
| confounded / 1 | converged | 1.009116 | 500.803 | 850.112 | 0.044688 | 0 | 0.00 | 242.503 |
| sparse_rookie / 1 | failed | 1.010498 | 504.556 | 441.954 | 0.043762 | 0 | 0.00 | 101.683 |
| sparse_rookie / 2 | converged | 1.005944 | 1299.778 | 1260.498 | 0.040317 | 2 | 0.00 | 185.180 |
| dominant_teammate / 1 | converged | 1.009162 | 610.349 | 470.876 | 0.039565 | 0 | 0.00 | 90.305 |
| era_boundary / 1 | converged | 1.009570 | 560.669 | 667.125 | 0.037727 | 0 | 0.00 | 79.755 |
| real_network / 1 | failed | 1.047499 | 136.708 | 64.289 | 0.138661 | 0 | 0.63 | 14,988.008 |
| real_network / 2 | failed | 1.011543 | 408.160 | 642.712 | 0.050533 | 0 | 1.00 | 33,403.986 |
| real_network / 3 | converged | 1.006593 | 750.474 | 1134.929 | 0.037405 | 0 | 1.00 | 61,554.801 |

Recorded parameter-family failures:

- connected, attempt 1: `car_step_z`.
- sparse_rookie, attempt 1: `sd_track_load`.
- real_network, attempt 1: `car_jump_z`, `car_segment`, `car_segment_z`, `car_step_z`, `car_total`, `compat`, `driver_form`, `driver_form_z`, `overall_car_contrast`, `overall_car_link`, `overall_day_z`, `overall_race_skill`, `overall_sd_drift_race`, `overall_sd_drift_season`, `overall_sd_grid_circuit`, `overall_walk_z`, `overall_weekend_complement_z`, `overall_weekend_qualifying_scale`, `sd_car_event`, `sd_car_segment`, `sd_compat`, `sd_driver_form`, `sigma0`, `skill_z`, `track_load`, `track_load_z`.
- real_network, attempt 2: `overall_weekend_qualifying_scale`, `sd_driver_form`.

The full-network later attempts reach the recorded tree-depth limit on every
retained transition. This indicates expensive sampling and motivates the separate
runtime work; it is not a retrospective new convergence rule. No depth setting,
threshold or retry seed was changed after these results. The recorded
current-driver contrast diagnostics passed for every attempt; those contrasts
are separate from the parameter gate and do not erase its failures.

## Recorded synthetic truth intervals

Each result's `recovery` rows compare known pairwise synthetic truth with the
saved `q05` to `q95` interval. The table counts misses in those original rows;
it does not independently recompute the interval estimates.

| Case | Skill interval misses / comparisons | Race-skill interval misses / comparisons |
|---|---:|---:|
| connected | 2 / 15 | 4 / 15 |
| confounded | 0 / 15 | 0 / 15 |
| sparse_rookie | 0 / 15 | 0 / 15 |
| dominant_teammate | 0 / 15 | 0 / 15 |
| era_boundary | 0 / 15 | 2 / 15 |
| real_network | 45 / 231 | 22 / 231 |

There is a single synthetic realization per declared case, and pairwise
comparisons share drivers. These counts do not establish interval calibration,
full recovery or simulation-based calibration, and no new pass/fail threshold is
inferred from them. Expected-point and ranking-functional acceptance is still
required by the [full plan](overall_model_plan.md).
