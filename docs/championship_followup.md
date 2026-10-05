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
