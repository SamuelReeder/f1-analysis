# Matched coordinate experiment: result

The separately approved experiment **failed its computational requirement**.
Every registered study and permitted retry was processed. The connected and
sparse-rookie studies converged; the confounded study exhausted its fixed ladder.
This was a recorded convergence failure, with no crash or OOM in this experiment.
The original failed profile remains unchanged. No real-driver ranking was fitted
or published.

The [frozen registration](overall_rotation_profile_protocol.json),
[complete review](../outputs/championship/development/overall-control-rotation-profile-v1/review.json)
and [raw summary](../outputs/championship/development/overall-control-rotation-profile-v1/summary.json)
bind the inputs, all attempts, diagnostics, resources and evidence boundaries.

## Study outcomes

| Study | Result | Attempts | Final R-hat maximum | Final divergences | Recorded study seconds |
|---|---|---:|---:|---:|---:|
| connected | converged | 3 | 1.003974 | 2 | 1341.59 |
| confounded | failed | 4 | 1.057654 | 0 | 2295.81 |
| sparse_rookie | converged | 2 | 1.006236 | 3 | 220.25 |

The study seconds are the fitter's recorded monotonic durations, including its
attempts and archive work. They are retained separately from elapsed wall time.

## Every attempt

| Study | Attempt | R-hat maximum | Bulk ESS minimum | Tail ESS minimum | Divergences | Depth-limit fraction | Recorded seconds | Passed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| connected | 1 | 1.014167 | 592.39 | 827.20 | 0 | 1.000000 | 223.06 | no |
| connected | 2 | 1.006658 | 541.38 | 299.47 | 0 | 0.705469 | 411.74 | no |
| connected | 3 | 1.003974 | 2051.34 | 2184.45 | 2 | 0.782734 | 702.99 | yes |
| confounded | 1 | 1.123708 | 24.19 | 55.78 | 0 | 1.000000 | 240.88 | no |
| confounded | 2 | 1.089441 | 42.56 | 181.37 | 0 | 1.000000 | 465.99 | no |
| confounded | 3 | 1.022706 | 179.56 | 323.20 | 0 | 1.000000 | 800.59 | no |
| confounded | 4 | 1.057654 | 69.60 | 70.00 | 0 | 1.000000 | 782.46 | no |
| sparse_rookie | 1 | 1.010235 | 957.75 | 1232.85 | 1 | 0.000000 | 84.64 | no |
| sparse_rookie | 2 | 1.006236 | 1627.31 | 2107.29 | 3 | 0.000000 | 134.05 | yes |

The registered rules require rank-normalized R-hat below 1.01, bulk and tail
ESS at least 400, mean MCSE/SD at most 0.05, and at most 0.001 divergences per
draw. The unchanged retry ladder and seeds are recorded for every row in the
review. Passing R-hat alone cannot rescue a failure of another criterion.

The final confounded attempt had maximum R-hat 1.057654, minimum bulk ESS
69.60, minimum tail ESS 70.00 and maximum mean MCSE/SD 0.123368. It had no
divergences, but every transition reached the registered depth limit. Its
largest R-hat was in `overall_sd_day`, the race weekend-variation scale.
Qualifying form and its scale also failed. These identify diagnostics needing
investigation, not an established cause of the failure.

Every attempt includes all current-field qualifying and race contrasts. The
final confounded attempt failed the qualifying contrasts between `synthetic_4`
and `synthetic_1`, and between `synthetic_5` and `synthetic_1`. Its earlier third
attempt passed the contrast checks but still failed the original parameter
diagnostics. That attempt cannot be selected to replace the exhausted result.
All contrasts in the final connected and sparse-rookie attempts passed.

## Integrity and resources

Generated observations and the complete generating truths match the original
registered studies exactly. The model/data snapshot, original truth files,
coordinate implementation, seeds, settings, gates and full inventory were checked.
Every saved attempt archive has a retained fingerprint; failed archives remain
diagnostic records and cannot enter a production posterior.

The service used the registered 6G memory cap, shared 800% CPU slice and nice 19.
Its recorded cgroup memory peak was 1,958,191,104 bytes.
GNU time reported elapsed wall time 58:54.02 and process maximum resident memory
1,745,684 KiB; its [raw output](../outputs/championship/development/overall-control-rotation-profile-v1/resource-output.txt)
is preserved. Per-study resident-memory maxima are cumulative within the process.
The summed receipt durations are 3857.65 seconds,
whereas GNU time reports 3534.02 wall seconds. Both measurements remain in
the evidence without an established explanation for their difference. This local
synthetic run does not verify full fitting or refresh resources on the intended
CI runner, and it does not explain the earlier unrelated OOM.

## Interpretation and next work

The original profile also failed the confounded study. The new experiment does
not establish a numerical repair. Corrected arithmetic and rotated coordinates
were evaluated together, so the observed differences cannot be attributed to
the rotation alone.

The next diagnostic step can inspect the existing saved draws for relationships
among weekend scales, shared form and driver/car effects, covering every attempt
and chain. That analysis would require no new fit. It must preserve the failure
and cannot select a favorable retry or change a gate. No additional sampling or
automatic retuning is authorized by this result.

Full recovery and SBC, calibrated weekend assumptions, ranking precision,
chronological prediction, independent prospective evidence and accepted
publication remain incomplete. The canonical Overall ranking remains withheld.
