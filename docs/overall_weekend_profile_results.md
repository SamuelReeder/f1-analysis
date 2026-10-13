# Conditional weekend-coordinate experiment: result

The separately approved experiment **passed its registered numerical profile**.
Every registered study converged under the unchanged diagnostics and fixed retry
ladder, including the previously failing confounded case. All original and new
coordinate diagnostics, and all current-driver qualifying/race comparisons in
each final attempt, passed. This resolves this computational profile's immediate
blocker. It does not establish complete recovery, calibration, identification,
ranking precision or predictive validity.

The [frozen registration](overall_weekend_profile_protocol.json),
[complete review](../outputs/championship/development/overall-weekend-coordinate-profile-v1/review.json),
[raw summary](../outputs/championship/development/overall-weekend-coordinate-profile-v1/summary.json)
and [saved-draw verification](../outputs/championship/development/overall-weekend-coordinate-profile-v1/archive-verification.json)
retain the complete inventory. Both earlier failed experiments remain unchanged.
No real-driver fit or ranking was produced, and the prototype remains outside
the production model.

## Study outcomes

| Study | Result | Attempts | Final R-hat maximum | Final divergences | Recorded study seconds |
|---|---|---:|---:|---:|---:|
| connected | converged | 2 | 1.004060 | 0 | 657.92 |
| confounded | converged | 3 | 1.007069 | 0 | 1549.86 |
| sparse_rookie | converged | 2 | 1.004279 | 2 | 234.32 |

Study seconds are the fitter's recorded monotonic durations, including attempts
and archive work. They are not substituted for elapsed wall time.

## Every attempt

| Study | Attempt | R-hat maximum | Bulk ESS minimum | Tail ESS minimum | MCSE/SD maximum | Divergences | Depth-limit fraction | Recorded seconds | Passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| connected | 1 | 1.010274 | 301.48 | 151.73 | 0.066658 | 0 | 0.999062 | 231.42 | no |
| connected | 2 | 1.004060 | 1521.17 | 1479.73 | 0.025780 | 0 | 1.000000 | 424.74 | yes |
| confounded | 1 | 1.076948 | 41.58 | 26.61 | 0.147778 | 0 | 1.000000 | 249.84 | no |
| confounded | 2 | 1.017700 | 195.27 | 225.96 | 0.068446 | 0 | 1.000000 | 478.40 | no |
| confounded | 3 | 1.007069 | 755.10 | 621.17 | 0.037042 | 0 | 1.000000 | 817.49 | yes |
| sparse_rookie | 1 | 1.010825 | 917.94 | 973.34 | 0.034550 | 0 | 0.000000 | 90.55 | no |
| sparse_rookie | 2 | 1.004279 | 1593.27 | 1506.02 | 0.025377 | 2 | 0.000000 | 142.07 | yes |

The original rules require rank-normalized R-hat below 1.01, bulk and tail ESS
at least 400, mean MCSE/SD at most 0.05, and at most 0.001 divergences per draw.
Every retry used its previously fixed seed and settings. All failed attempts
remain retained as diagnostic records; none was selected to replace a failure.

The connected initial failure was in `sigma_tau`. The confounded initial attempts
failed multiple original and transformed states; the complete parameter lists
are in the review. Its first attempt also failed driver comparisons. Its next
attempt passed the comparisons but still failed the full parameter checks,
so sampling continued according to the existing ladder. The sparse-rookie
initial failure was in `car_step_z` and `overall_walk_z`.

The final connected and confounded attempts reached the tree-depth limit on
every recorded sampling transition. That is a continuing computational
limitation. It was not a registered rejection criterion, and the original
convergence decision has not been changed or weakened. This profile does not
establish reliable inference for other datasets or future fits.

## Known-truth descriptions remain incomplete calibration evidence

The original runner also reports its fixed pairwise truth checks. The following
counts include every current-driver contrast in each study, using the already
implemented 90% posterior interval:

| Study | Qualifying truths inside interval | Race truths inside interval |
|---|---:|---:|
| connected | 15 / 15 | 15 / 15 |
| confounded | 15 / 15 | 15 / 15 |
| sparse_rookie | 9 / 15 | 10 / 15 |

These are correlated comparisons within individual generated datasets, not
independent calibration replicates or a new acceptance gate. The sparse-rookie
misses are preserved. Neither the complete coverage in the other cases nor
computational convergence establishes calibrated uncertainty. Full known-truth
recovery and simulation-based calibration remain required under a separate
complete registration; do not choose new truths or settings to remove these
misses.

## Integrity and resources

The original observations, complete generating truths, model/data snapshot,
coordinate implementation, study inventory, seeds, sampler settings, diagnostic
gates and retry ladder were checked. The read-only collector verifies the
complete attempt and comparison inventories and records every archive's byte
identity. Saved-draw verification rereads every numeric member and exactly
reproduces every retained parameter diagnostic and current-driver comparison
report, including all failed attempts. It generates no new samples.

The service used the registered 6G memory cap, shared 800% CPU slice, CPU weight
10 and nice 19. Its cgroup memory peak was 2,006,269,952 bytes.
GNU time recorded wall duration 37:26.73 and process maximum resident
memory 1,756,424 KiB. The complete
[resource record](../outputs/championship/development/overall-weekend-coordinate-profile-v1/resources.json)
and [raw resource output](../outputs/championship/development/overall-weekend-coordinate-profile-v1/resource-output.txt)
are preserved. There was no crash or OOM in this experiment.

The summed fitter durations are 2442.10 seconds, whereas GNU time
reports 2246.73 wall seconds. Both measurements are retained without
an established explanation for their difference. Per-study resident-memory
maxima are cumulative within the process. This local synthetic measurement
does not verify intended-runner fitting/refresh resources or explain the earlier
unrelated OOM.

## Meaning and next work

The previous lap-control rotation experiment exhausted its confounded retry
ladder. The conditional weekend transform now satisfies the same checks on that
same generated dataset, without changing its probability model. This matched
result supports the computational approach on this inventory; it is not a
proof that the observational data separate real driver and car effects.

Next work is the complete recovery/calibration protocol and numerical validation
of ranking functionals, including sparse, confounded and transfer-network cases.
It also requires a justified complete study size and resource plan before new
fits. The broader weekend assumptions, chronological prediction, independent
prospective evidence, accepted producer and refresh/rollout checks remain open.
This result authorizes no automatic additional fit or deployment. The canonical
Overall ranking remains unavailable, with no fallback to these synthetic draws.
