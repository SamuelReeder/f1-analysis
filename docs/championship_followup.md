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

The initial processed-data inventory is
`outputs/analysis/championship_evidence_inventory.json`. It did not find earlier lap
inputs in the processed tables. A subsequent raw-archive audit in commit `9f347e6`
found complete lap-event coverage in the already downloaded dump and verified that
the recorded championship benchmark excluded those earlier race outcomes. The
separate study is registered in [`championship_archive.md`](championship_archive.md).
Its scope is earlier-era validation; the audit cannot establish what humans may
have inspected outside the repository.
The risk from repeatedly choosing models using the same validation outcomes is
also described by [Cawley and Talbot](https://www.jmlr.org/beta/papers/v11/cawley10a.html).

## Registered follow-up and current limit

`docs/championship_followup_protocol.json`, committed in `c265e40`, freezes the
race-specific-pace-only candidate and the evaluator's source hashes. It keeps the
existing race likelihood, priors, season-ahead inputs, sampling defaults, retry
ladder, and season-block uncertainty calculation. There is no backward selection
or search over seeds. Grid-only and ratings-only comparisons are secondary and
cannot rescue a failed primary comparison with grid plus qualifying ratings.

The registration accepts only outcomes dated after 2026-10-05. Its fixed horizon
covers the remaining eligible races in 2026 and the seasons 2027 and 2028, with a
decision after 2028-12-31. This retains the existing minimum of three converged
season blocks. It does not repeatedly test intermediate scores or extend the
horizon after seeing a result. Races completed before registration are excluded,
even when their results have not yet been downloaded.

This is a substantial timing limit: **the prospective experiment cannot establish
improvement now**. The separately registered archive study provides a potential
earlier validation route using race outcomes excluded from the recorded benchmark.
That study has now stopped on an exhausted convergence ladder before predictive
scoring, as recorded below. The original failure and this future registration
remain immutable.

Run the evaluator as a memory-capped systemd user service when the horizon and
inputs are ready. Its command is:

```bash
.venv/bin/python -m analysis.championship_followup evaluate
```

It checks the original committed registration, frozen model code, complete race
coverage against freshly retrieved completed-season results, pre-season training
cutoffs, convergence, and unchanged input hashes during evaluation. Its research
outputs stay under `outputs/analysis/championship_followup/`; it cannot write the
canonical championship result or dashboard data. A passing future result would
still require integration, regeneration, and publication review.

The current preflight returned `awaiting_future_evidence`, with `gate: false`, and
started no fits. The full Python suite passed 157 tests. The portable dashboard
publication still verifies. Sources:
`outputs/analysis/championship_followup/{status,verification}.json`. No model source,
canonical racing output, or published ranking was changed by this follow-up.

## Archive study outcome

The separate archive study ended on 2026-10-05 with
`kind: exhausted_registered_fit`, `gate: false`, and
`publication_authorized: false`. Its terminal evidence is committed in `4d8eb23`:
`outputs/analysis/championship_archive/{failure,execution}.json`,
`execution.log`, `preparation.json`, and `failure.manifest.json`.
The frozen registration and its source/input hashes still match. The prepared lap
and weather hashes were verified after failure, and the canonical dashboard
publication still verifies. No model code or canonical racing result was changed.

The first test fold, 2007, exhausted the original race-pace retry ladder. Its
qualifying fit had passed on its final registered attempt:

| Qualifying attempt | R-hat maximum | Divergences | Draws | Converged |
|---|---:|---:|---:|---|
| 1 | 1.097074 | 7 | 1,600 | No |
| 2 | 1.051602 | 3 | 3,200 | No |
| 3 | 1.041013 | 235 | 6,400 | No |
| 4 | 1.018885 | 2 | 6,400 | Yes |

| Race-pace attempt | R-hat maximum | Divergences | Draws | Converged |
|---|---:|---:|---:|---|
| 1 | 1.026081 | 4 | 2,000 | No |
| 2 | 1.020425 | 3 | 2,000 | No |
| 3 | 1.036766 | 6 | 2,000 | No |

All race-pace R-hat values passed the unchanged threshold, but each attempt
exceeded the permitted divergence count of 2 for 2,000 draws. The registered
implementation is `f1rank/racemulti.py:fit`, which uses its original retry ladder;
it was explicitly frozen separately from the racing-quality `artifacts.RETRY`
ladder. Changing its settings now would not complete the registered experiment.

The process recorded 1,369.451 seconds elapsed. It made no race-stage fits and
completed no predictive folds. Thus there is no archive predictive interval to
interpret: this is a computational failure, not evidence that the candidate's
predictive effect is negative. The failed draws were not checkpointed for reuse.
No season was removed and no sampler setting, model, or acceptance criterion was
changed after the failure.

The archive plan and registration remain unchanged. The prospective study still
awaits its registered future evidence. Neither study currently establishes the
simpler model, and the original combined prediction test remains failed. A new
methodological study would need a justified specification and separately
registered evaluation; it cannot be presented as a passing rerun of this study.
The current evidence does not authorize an Overall ranking.

### Concrete next-study proposal, awaiting authorization

**Status update:** the user has now authorized this proposal. Its separate frozen
plan is [`championship_archive_conservative.md`](championship_archive_conservative.md).
The text below records the proposal that was approved; the failed study remains
unchanged.

The standing instruction says to record an exhausted fit and not rerun it with
different settings. Accordingly, no further archive fits have been launched.
The following is a proposal for a separate computational study, not an amendment
to the completed archive registration and not an active experiment:

- Keep the fixed race-pace candidate, likelihood, priors, archive seasons, input
  rules, baselines, convergence thresholds, and predictive decision rule unchanged.
- Before any new fit, commit a separate registration, source hashes, output
  directory, and cache namespace. Preserve the failed study and its diagnostics.
- Apply the same conservative sampler recipe to every archive race-pace fold:
  1,500 warm-up iterations, 2,000 draws per chain, four chains, target acceptance
  0.98, and seed 3. These settings are the final stage of the already committed
  `artifacts.RETRY` ladder applied to the original `racemulti.fit` defaults. There
  would be one attempt under this recipe, with no additional retries or seed search.
- Keep qualifying and race-stage fitting under their existing registered rules.
  Verify all qualifying and race-pace fits before computing predictive scores.
  Require every archive fold to converge; any exhausted fit terminates the new
  study without dropping its season. Evaluate the full predictive comparison only
  once, without inspecting partial scores to change the study.
- Implement the sampler wrapper in a separate analysis module so the frozen model
  sources, original registrations, canonical outputs, and dashboard remain intact.
  Verify that the wrapper uses exactly the existing likelihood and priors.
- Record a pass, failure, or inconclusive interval and retain the earlier-era scope.
  A successful calculation would still require the publication review described
  in the archive plan; it would not itself switch on the Overall table.

This proposal changes computation after observing a convergence failure. It needs
explicit authorization to depart from the standing stop rule. The existing
prospective study can instead remain the next evaluation route; its fixed horizon
and settings are already registered. No passing outcome is promised by either route.

## Authorized computational follow-up outcome

The user approved the separate conservative sampler study. Its implementation was
committed in `71661e6` and its immutable registration in `3539740`, before fitting.
The Python suite passed 162 tests in 46.98 seconds. The run used a 6 GiB memory cap,
with the existing shared CPU quota unchanged. Sources:
`outputs/analysis/championship_archive_conservative/verification.json` and
`docs/championship_archive_conservative_protocol.json`.

The study failed its first archive race-pace training fit. Its fixed attempt used
the registered seed 3, 1,500 warm-up iterations, 2,000 samples per chain, four
chains, and target acceptance 0.98. The maximum R-hat was 564,716.4375, with zero
divergences across 8,000 draws. Zero divergences did not establish convergence:
the extreme between-chain disagreement fails the unchanged R-hat requirement.
The process recorded 2,885.949 seconds elapsed; the pace attempt itself recorded
2,884.917 seconds. It reused the already accepted qualifying checkpoint, so the
qualifying attempt history in its summary is not a new set of qualifying fits.

No race-stage fit or predictive fold was completed. The failure and its frozen
evidence are committed in `0cee335`:
`outputs/analysis/championship_archive_conservative/{failure,execution}.json`,
`failure.manifest.json`, `execution.log`, and `preparation.json`. Registered
sources, inputs, and copied qualifying checkpoints still match. The original
dashboard publication also still verifies. This is another computational failure;
there is no predictive interval from this study to interpret or publish.

The wrapper calls the existing race-pace data adapter, likelihood, priors, and
prior-median initialization directly; its sampler settings match the authorized
registration. Source review has not established an implementation error that would
justify replacing the result. The saved failure contains the worst R-hat but not
the individual failed chain states, so it does not identify which parameter caused
the disagreement. A separate initialization diagnostic checks the frozen seed's
starting values and gradients without taking warm-up or posterior transitions.
It cannot produce a replacement passing fit or a ranking.

The initialization diagnostic in `outputs/analysis/championship_sampler_initialization/`
found finite initial parameters and gradients in every registered chain. It made
no sampling transitions. This does not explain the later chain disagreement or
establish that sampling was reliable; no concrete implementation defect has been
demonstrated by this check. The failed studies remain terminal.
