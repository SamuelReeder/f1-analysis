# Saved weekend effects: diagnostic result

The completed post hoc analysis points to difficulty separating the qualifying
and race-only contributions to weekend form. It does not establish a cause or a
successful numerical repair. The confounded experiment remains failed, and its
draws cannot supply a driver ranking.

The [definition](overall_weekend_diagnosis.md),
[complete results](../outputs/championship/development/overall-weekend-diagnosis-v2/results.json)
and [review](../outputs/championship/development/overall-weekend-diagnosis-v2/review.json)
cover all 3 studies, 9 attempts, 4 chains per attempt and 15 current-driver pairs
per attempt. The original diagnostics for every retained saved site reproduce.
Generated observations and truths match, archive fingerprints are checked, and
every archive member is validated. No model, prior, gate or sampling setting
changed; there was no new fit.

## Total form is more stable than its decomposition

The conditional race-form marginal scale is `hypot(b * sf, sr)`, where `sf` is
qualifying form scale, `sr` is independent race form scale and `b` is the shared
weekend loading. This is a property of the specified population covariance,
not a posterior estimate of a driver's performance.

In every confounded attempt, that combined scale has better numerical
diagnostics than the independent race scale and the implied qualifying/race
correlation. All rows below come from the complete results; no attempt is chosen
as a replacement fit.

| Confounded attempt | Independent race scale R-hat | Marginal race scale R-hat | Implied form correlation R-hat | Total race-form RMS R-hat |
|---|---:|---:|---:|---:|
| 1 | 1.123708 | 1.002969 | 1.145931 | 1.002038 |
| 2 | 1.089441 | 1.003616 | 1.108755 | 1.004604 |
| 3 | 1.022706 | 1.000901 | 1.028545 | 1.000842 |
| 4 | 1.057654 | 1.006635 | 1.064101 | 1.009319 |

The final attempt's independent race-scale chain means range from 0.1695 to
0.2605, while the combined scale means range from 0.3656 to 0.3734. These are
descriptions of disagreeing chains, not calibrated posterior estimates. Better
diagnostics on the total cannot rescue a failed component or the whole fit.

The recorded scale/innovation RMS correlations do not support a simple claim
that every failure is a strong inverse relationship between a scale and its
standardized innovation. For example, the converged sparse-rookie study has a
stronger inverse race-scale/innovation relationship than the confounded study.
Differences between these studies cannot isolate teammate connectivity because
they have different generating draws.

## Driver/car separation also remains uncertain

All current pairwise total race-location checks pass in every saved attempt.
This is a diagnostic statement about their sum, not recovery of its driver and
car components. In the final confounded attempt, qualifying ability fails for
`synthetic_4` minus `synthetic_1` and `synthetic_5` minus `synthetic_1`.
Qualifying form fails for 10 pairs, qualifying car for 8, qualifying base
location for 6, and race form for 1. The complete review retains the failed
identities for every attempt and quantity.

For those two qualifying-ability comparisons, the final attempt's within-chain
driver/car correlations range from about -0.720 to -0.568 for qualifying and
-0.933 to -0.912 for racing. The full matrices and all other pairs are retained.
The components can compensate even when their total is sampled more reliably.
Correlation alone proves neither non-identification nor the cause of failed
sampling. Successful numerical diagnostics would still require recovery and
predictive evidence before claiming equal-car driver separation.

## Software, resources and next decision

The final focused software run passes 38 tests. The first run's hand-calculated
centering fixture was incorrect; its failed report is preserved and the
implementation was unchanged when the fixture was corrected. The first analysis
completed its calculations but could not serialize a NumPy scalar in roundoff
metadata. Its failure is preserved, with no partial result accepted. Converting
that metadata to a JSON scalar allowed the complete read-only analysis to finish.

The completed diagnosis records 44.158 seconds and process maximum resident
memory 890,468 KiB under the 3G memory and 200% CPU scope caps. These are local
analysis measurements, not full-fit or intended-runner resource verification.

The next computational proposal can investigate coordinates aligned with total
race form and its complementary qualifying direction. It must retain the exact
joint prior and likelihood, all dimensions, original observations and truths,
and the full fixed diagnostic/retry procedure. Such a proposal needs equivalence
checks and a separate frozen experiment before fitting. This diagnosis does not
authorize a rerun or changing model assumptions to obtain a preferred ranking.
