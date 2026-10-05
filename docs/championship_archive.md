# Fixed race pace: separate historical archive validation

This plan is fixed before archive fits or scores. The candidate is the same fixed
race-specific-pace model selected from the later-era development evidence. This
study neither alters the failed original championship experiment nor changes the
prospective registration.

## Why this archive

The audit in commit `9f347e6` found complete lap and qualifying event coverage for
the earlier archive in the already downloaded, hash-verified Jolpica dump. Every
recorded benchmark revision starts racing outcomes in 2010, and the recorded
championship revisions use that benchmark's race-order inputs. The earlier
qualifying observations were used in qualifying-window comparisons. Their presence
does not make race finishing orders a previously evaluated championship target.
The audit cannot establish what humans inspected outside the repository. This is
therefore a separately scored historical archive, not a claim of blinded or
prospectively collected outcomes.

Use the entire available earlier era: 2006 supplies the first training prefix;
2007, 2008, and 2009 are the held-out seasons. Later folds can train on earlier
archive seasons. These years follow the data boundary, not inspection of their
prediction errors. The opening fold has a shorter training history than the later
development folds; this limitation is retained rather than excluding it.

## Frozen model and input rules

- Fixed candidate: grid plus qualifying driver/car forecasts plus race-specific
  pace. Primary baseline: grid plus the same qualifying forecasts. Grid alone and
  ratings alone are secondary comparisons and cannot rescue a failed primary test.
- All qualifying and lap-model training observations precede each test season.
  Qualifying designs end with that test season and use the existing model, priors,
  data filters, and `jobs.ATTEMPTS`. Existing preprocessing excludes fuel-laden Q3
  observations in this era. No later fitted qualifying posterior is reused.
- The joint race-pace likelihood and its original sampling/retry defaults are
  unchanged. Older laps retain the existing weaker observation model: unknown
  compounds, inferred stops and tyre ages, stint offsets, and serially correlated
  errors. Refuelling and the earlier field are limitations of this transport test.
- Use the verified dump parser and existing old-lap framing and cleaning functions.
  Retirement windows use the original pre-FastF1 rule based on completed laps and
  administrative status. No learned cause classifier is needed by the lap mask.
- Weather uses the unchanged historical precipitation proxy and threshold, with
  responses cached separately and hashed before fits. Missing weather is an error.
- Admit a dry race exactly when stage A would produce a nonempty pair table: at
  least ten drivers survive the existing clean-lap rule, and at least one team has
  exactly two such drivers. Stage-A coefficient/bootstrap values are not inputs to
  the joint lap likelihood, so they need not be recomputed merely to decide
  membership. Admission uses no predictive score.
- Race-stage fitting and prediction reuse the existing championship functions,
  including integration of quality draws. Only the input adapter's first training
  season changes to the archive boundary. Unknown driver qualities retain the
  existing zero-mean rule.

## Decision, failures, and scope

Record every race's paired log predictive density, preserving event alignment.
Use the unchanged season-block bootstrap and minimum number of valid seasons in
`championship.paired_seasons`. All registered archive seasons must finish and
converge. If a registered fit exhausts its fixed attempts, record failure and stop;
do not remove its season or retune. Do not inspect partial scores to change the
candidate, inputs, dates, or threshold. Make the decision only after all folds.
Operational interruptions are recorded separately. They may resume using the same
committed protocol, unchanged inputs, and matching checked caches; an exhausted
convergence ladder is a terminal statistical failure, not an interruption.

A positive result would support the fixed candidate on this earlier archive. It
would not prove causal driver skill, erase the later-era failure, establish modern
era stability, or itself authorize a published ranking. Any integration must retain
the separate evidence and limitations and undergo publication review. A failed or
inconclusive result stays recorded.

`analysis/championship_archive.py register` records the source hashes, input hashes,
candidate, cutoffs, and rules in an immutable committed protocol before `run`.
Fits and weather caches stay under the ignored `outputs/fits/championship_archive/`.
Auditable summaries stay under `outputs/analysis/championship_archive/`. Canonical
model code, data, championship outputs, and dashboard data are not modified.
