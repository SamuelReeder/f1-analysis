# Championship diagnosis and follow-up

## Diagnostic plan, recorded before replaying the fits

The published experiment is complete and failed its combined gate. Its immutable
reference is `outputs/championship/summary.json` in commit `de8eee2`. The current
deployment choice contains race-specific pace, but a choice made using all the
historical results is not a fresh validation of that choice. Neither this document
nor the diagnostic replay changes the existing model, exclusions, retry ladder,
selection rule, bootstrap, or publication gate.

Run `python -m analysis.championship_diagnosis` using the existing checked inputs.
The analysis wraps the existing championship functions to capture their otherwise
discarded predictive scores and coefficient summaries. The model code and its
inputs remain unchanged. Commit the analysis and this plan before starting it.

The replay will:

- Reproduce the original combined selection procedure, including the original
  order of calls, random seeds, and MCMC convergence checks.
- Record every evaluated quality subset's race-level log predictive densities,
  the corresponding event identifiers, coefficient posterior summaries, and fit
  diagnostics. Record training and test coverage and quality uncertainty by fold.
- Compare the replay's choices, entry tests, and combined result with the committed
  reference. Numerical agreement is required within the tolerances defined in
  `analysis/championship_diagnosis.py`; disagreement stops interpretation.
- Decompose the unchanged outer test by season. Report the contribution of each
  season and the result when each season is omitted, for influence diagnosis only.
  No season is removed from a gate because it hurts the score.
- Describe the fixed race-specific-pace-only model on exactly the same outer
  seasons, using the same fits and scores. This is a post-selection historical
  comparison, not a replacement publication test. Its candidate was chosen after
  the original results were seen.

Outputs go under `outputs/analysis/championship_diagnosis/`. A completed result
includes input hashes, the source revision, elapsed time, the numerical replay
check, and an explicit exploratory-evidence label. The run must fail closed on
stale inputs, unconverged fits, nonfinite scores, mismatched event alignment, or
failure to reproduce the original result. It never writes championship outputs or
dashboard data.

## Evidence boundary for a follow-up

Use the diagnosis to choose a concrete modeling change with a plausible cause,
then freeze its specification and evaluation before fitting that new experiment.
Record all attempts and failures. Do not search seeds, exclusions, thresholds, or
candidate variants until a historical interval becomes positive.

Historical races used in the original experiment are development evidence for a
new candidate, including their already inspected aggregate scores. Repeating a
nested historical test does not make those outcomes unseen again. A positive
historical result alone cannot reverse the completed gate. Confirmatory evidence
must use outcomes not involved in the model-design decisions, with a preregistered
comparison and enough independent blocks to assess uncertainty. If such evidence
is not yet available, report that limitation and keep the Overall ranking withheld.

The intended eventual claim is an experimental equal-car championship with stated
assumptions and uncertainty. Predictive improvement alone does not establish a
universal or causal best-driver ranking.

## Completed diagnostic replay

The replay in commit `f1f8e33` reproduced the reference choices, entry tests, and
combined result. Its source revision was `dfd4088`. It took 848.630 seconds and
made 146 fits, with no retries. The Python suite passed 150 tests. Sources:
`outputs/analysis/championship_diagnosis/{summary,verification}.json`;
`evaluations.jsonl` preserves every evaluated subset's scores and diagnostics.

The original selection procedure remains failed: mean improvement +0.125434 per
race, season-block interval [-0.029811, +0.293439]. The predeclared descriptive
race-pace-only comparison on the same 207 outer races gives +0.157303, with interval
[+0.018981, +0.307308]. This candidate was chosen after the original experiment;
this positive historical interval is development evidence, not a new gate pass.

| Test season | Original selection: improvement per race | Fixed race pace: improvement per race |
|---|---:|---:|
| 2016 | +0.079992 | +0.284885 |
| 2017 | +0.098180 | +0.049816 |
| 2018 | -0.011620 | +0.021128 |
| 2019 | +0.277714 | +0.275711 |
| 2020 | +0.508298 | +0.508298 |
| 2022 | -0.053273 | -0.082432 |
| 2023 | +0.595160 | +0.582174 |
| 2024 | +0.078214 | +0.078428 |
| 2025 | -0.304933 | -0.157523 |
| 2026 | +0.107472 | +0.107472 |

The fixed candidate improves the early choice in 2016 and reduces the loss in
2025, when the original procedure also selected first-lap performance. It does
not remove all weak seasons: both procedures lose in 2022 and 2025. Omitting 2025
would make the original interval positive, but that is not a permitted exclusion.
Omitting 2023 makes the fixed candidate's interval [-0.007006, +0.236778], so the
apparent improvement is sensitive to the strongest season. These leave-one-season
calculations diagnose influence; they do not change either experiment's data.

Coefficient signs are consistent with the input definitions: race pace and
qualifying pace are positive for faster drivers, and their fitted race-strength
coefficients are positive. The recorded diagnostics do not indicate a convergence
failure to repair. There is evidence for simplifying selection, but no demonstrated
software error whose correction would reverse the completed gate.

The independent-data inventory is
`outputs/analysis/championship_evidence_inventory.json`. Earlier race results are
present, but matching earlier lap inputs are absent, and no deliberately reserved
test set has been established. The archive alone is not a confirmation dataset.
The risk from repeatedly choosing models using the same validation outcomes is
also described by [Cawley and Talbot](https://www.jmlr.org/beta/papers/v11/cawley10a.html).
