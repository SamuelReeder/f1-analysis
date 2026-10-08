# Equal-car preview audit plan

This is a diagnosis requested after inspecting the provisional Hadjar result.
It is exploratory development evidence, not a new validation or a selection of
settings. Preserve the existing models, data, fits, preview, gates and failed
studies. Do not run new fits, tune parameters, publish data, or push changes.

Before running the diagnostic, commit this plan and its analysis script.
Use a memory-capped scope with MemoryMax=3G and CPUQuota=200%.

The diagnostic will:

- Verify preview provenance and the qualifying design fingerprint. Reproduce
  the original in-team simulation using its seed, draw ordering, coefficients,
  retirement probabilities and race count. Stop if expected points do not match.
- Summarize all current drivers' qualifying skill, team effect, race residual,
  coefficient contributions, data coverage and posterior correlations. Compare
  the already published independent current-season dry-race estimates without
  inserting them into the championship or treating them as a new validation.
- Describe Hadjar's observed qualifying comparisons, using matched segments
  and then equal-weighted event averages, and existing adjusted race comparisons.
  These are descriptive, retrospective observations, not forecasts.
- Run the following fixed probes with the original coefficients and paired
  random numbers: the original simulation; portable qualifying for everybody;
  no extra race-pace contribution; field-average qualifying for everybody;
  posterior-mean qualifying and race ability; no independent car-weekend noise;
  and Hadjar alone assigned average team effect, average qualifying ability,
  or average extra race pace. No probe is a refitted or validated alternative.
- Repeat only the original model using seeds 0 through 4 and report every result
  to assess simulation variation. Never select a seed or new headline result.
- Report pairwise simulated championship outcomes and Monte Carlo error of mean
  point differences, separately from model uncertainty. Measure points ties;
  array-order ties in the existing preview are not a sporting tiebreak model.
- Inspect the code for alignment/sign errors, omitted uncertainty, noise-model
  mismatch, driver/car/team confounding, transfer across eras, and missing race
  mechanisms. Distinguish demonstrated behavior from untested explanations.

Write numerical evidence only under `outputs/analysis/championship_model_audit/`.
Record input hashes, source revision and elapsed time. A separate results report
may explain and prioritize findings. A surprising driver position is not, on its
own, evidence of a bug, and a diagnostic probe cannot authorize publication.
