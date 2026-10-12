# Overall numerical profile: recorded result

Source: [complete profile review](../outputs/championship/development/overall-computation-profile-v1/review.json) and its linked study receipts. This is a completed synthetic numerical profile, not full recovery, SBC, real-data calibration or a driver ranking.

Every registered study was processed. The confounded case exhausted the fixed retry ladder. The connected and sparse-rookie cases converged after retries. The original failure remains failed; no threshold or sampler setting was changed to rescue it.

| Study | Attempt | Converged | Worst rank R-hat | Minimum bulk ESS | Minimum tail ESS | Divergences | Fitter elapsed seconds |
|---|---:|---|---:|---:|---:|---:|---:|
| connected | 1 | No | 1.010872 | 457.71 | 308.06 | 1 | 193.21 |
| connected | 2 | Yes | 1.004066 | 1167.65 | 1221.28 | 0 | 369.44 |
| confounded | 1 | No | 1.136581 | 25.93 | 118.81 | 0 | 204.52 |
| confounded | 2 | No | 1.023518 | 98.76 | 135.26 | 0 | 385.78 |
| confounded | 3 | No | 1.033097 | 105.42 | 314.33 | 0 | 676.78 |
| confounded | 4 | No | 1.056273 | 64.37 | 151.52 | 0 | 689.72 |
| sparse_rookie | 1 | No | 1.013512 | 853.04 | 970.78 | 2 | 73.99 |
| sparse_rookie | 2 | Yes | 1.005785 | 1287.22 | 2178.20 | 0 | 118.35 |

The durations above use the fitter’s monotonic clock. UTC timestamps and service resource measurements are retained separately in the review. They are different recorded measurements and are not substituted for one another.

Tree-depth saturation was substantial in the connected and confounded studies. It was a registered reported diagnostic, not an additional acceptance gate; the original converged statuses are not retrospectively changed. The confounded failure still requires investigation. The cases used different registered prior draws, so these results do not isolate the causal effect of teammate connectivity.

The unknown-weather source correction was applied only after this service terminated. The [input audit](../outputs/championship/development/weather-boundary-checks/audit.json) found no all-null rainfall event in the committed inputs. That bug is not an explanation or a cure for the profile failure. The old receipts remain tied to their original source; corrected-source work requires fresh registered evidence.

Next steps are read-only diagnosis of the saved attempts, completion of the scientific and evaluation contracts, and fresh registered numerical/recovery evidence for any subsequent source. The missing dominant-teammate, era-boundary and real-network studies remain required. No real-data ranking or publication follows from this profile.

## Post hoc contrast diagnosis

The [saved-draw diagnosis](../outputs/championship/development/overall-profile-diagnosis-v1/review.json) examines every attempt and every current qualifying/race driver pair. It verifies the original checker and reproduces the saved state diagnostics before deriving contrasts. Qualifying comparisons also fail in the final confounded attempt. Some earlier attempts have cleaner contrast diagnostics, but their complete fits failed; selecting those attempts would not meet the registered requirement. The failure therefore cannot be dismissed as affecting only irrelevant parameters.

This analysis is explicitly post hoc. It neither reclassifies the original attempts nor turns their failed draws into driver estimates or calibrated uncertainty. Archive checksums and all contrast diagnostics are preserved in its result file.

## Post hoc control geometry

The [geometry diagnosis](../outputs/championship/development/overall-profile-geometry-v1/review.json)
reproduces the original synthetic arrays and truths and retains the control
correlations from every saved attempt and chain. It identifies strong coupling
among lap controls as a reason to investigate sampling coordinates; it does not
establish the cause of the failure or a successful repair.

The separate [inference proposal](overall_inference_revision.md) preserves the
original prior and likelihood while rotating those controls. Density and software
checks were recorded before the separate numerical experiment. The user approved
that experiment on 2026-10-12; its [completed result](overall_rotation_profile_results.md)
again failed the confounded case. The original profile remains failed. The later
[weekend diagnosis](overall_weekend_diagnosis_results.md) preserves all attempts
and narrows the next investigation without accepting a failed result.
