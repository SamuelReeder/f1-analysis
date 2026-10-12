# Saved-draw diagnosis of weekend effects

This is a post hoc diagnosis of the completed coordinate experiment, specified
before reading its saved draws for this analysis. It cannot accept a failed fit,
change a gate, select a preferred retry, or authorize new sampling. Preserve all
studies, attempts, chains and current-field driver pairs.

The model's qualifying form is the event-centred vector `sf * zq`. Race form is
the event-centred vector `b * qualifying_form + sr * zr`, with independent
standard Normal innovations conditional on the hyperparameters. Before event
centring, the qualifying variance is `sf²`, race variance is `b² sf² + sr²`, and
cross-covariance is `b sf²`. Thus the race marginal scale is `hypot(b * sf, sr)`
and the qualifying/race correlation is `b * sf / hypot(b * sf, sr)`.
Event centring applies the same projection to both vectors. These are conditional
population covariances, not empirical posterior covariances of fitted states.

Retain the complete within-chain and pooled correlation matrices for the declared
weekend scales, shared loading, persistent-state scales/loadings, and observation
noise controls. Also retain the marginal race-form scale, cross-covariance,
correlation, independent-race variance fraction, and realised RMS of centred
standard innovations and their scaled effects. Freeze the complete site list in
the implementation before execution. Report chain means, standard deviations
and the existing interval summaries; never turn chain disagreement into a
precise posterior estimate. Pooled correlations can reflect disagreement between
chains, so always retain within-chain results separately.

For every current driver pair, examine qualifying and race ability, team, car
and weekend-form contrasts and their same-draw sums. The qualifying sum excludes
session intercepts, segment effects and observation noise; it is a base latent
location, not a complete qualifying forecast. The race sum is the total latent
pace state in the joint likelihood. Preserve each pair's complete correlation
matrices and numerical diagnostics. These contrasts are diagnostic functions,
not driver ratings or a substitute acceptance rule.

Reconstruct stored form vectors from their scales, loadings, innovations and
event identifiers, with an explicit floating-point consistency tolerance. Before
deriving anything, reproduce the original numerical diagnostics for every
retained saved site. Require the committed review's archive fingerprints and
study identities, validate every numeric archive member, and recheck inputs after
reading. Use bounded temporary maps; failed archives stay outside the production
posterior loader. Preserve original verdicts and report every diagnostic failure.

The analysis is descriptive. Correlation, a small scale, or a chain shift does
not prove the cause of poor sampling. Different synthetic studies used different
generating draws, so their differences cannot isolate teammate connectivity.
Any subsequent sampling proposal needs its own separately frozen procedure.
