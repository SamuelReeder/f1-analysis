# Showing the equal-car preview on the site

On 2026-10-23 the user asked for the Overall ratings to appear on the published
site now, labeled as unvalidated, rather than after the
[canonical Overall program](overall_model_plan.md) completes. That program has no
completed checklist item yet, and some items need races that have not happened.
This document records how the existing
[exploratory preview](championship_preview.md) is published. It changes no model,
fit, gate or validation result. The preview's own plan still governs how it was
computed. Its wording against publishing applied to the research run and is
superseded only for this labeled display. The plan file is a hashed input of the
preview, so it is left unedited.

## What is published

`dashboard/verify_publication.py record` writes `dashboard/public/data/overall_preview.json`
from `outputs/analysis/championship_preview/{summary.json,standings.csv,manifest.json}`:

- the headline (`in_team`) standings only: expected points per race, simulated
  title share and the 5th–95th percentile season rank range;
- its recording time, fit identifier and data cutoff (the Azerbaijan Grand Prix,
  2026-15, for the 2026-10-05 run);
- the original combined validation it did not pass, the two archive studies that
  stopped before scoring, and the race-pace convergence diagnostics.

The portable-skill rows and the HTML page are not published. The preview has no
contribution breakdown, so none is shown.

## Checks

- **Recording.** The file is rewritten only when `artifacts.require` confirms the
  preview's full provenance: data, code, posterior caches and its plan. Once
  those inputs change, for example after a new race, the committed file is kept
  with its own older cutoff. It is not silently refreshed.
- **Receipt.** `publication.json` records the file's SHA-256.
- **CI.** CI rebuilds the expected file from the committed preview outputs. The
  CSV and summary hashes must match the preview manifest, the summary must still
  say `gate: false`, and the standings must be internally consistent. Any
  difference stops the build.
- **Browser.** The site shows the preview only when the canonical result is not
  established. It rejects a preview that claims `validated`, a passing gate or
  empty standings. A validated canonical ranking always takes precedence.

## Presentation

The view is labeled **Preview · not validated**. It shows its own recording date
and data cutoff, and notes when other rankings use later data. It states the
held-out result from the recorded summary: +0.13 log score per race over 207
races (2016–2026), 95% interval −0.03 to +0.29. The interval includes zero.
Driver–team compatibility remains in the estimates. The simulation does not model
passing, traffic, strategy or weather lap by lap.

## Regenerating

`analysis/championship_preview.py` refuses to overwrite a recorded outcome.
Regenerating it with later data is a new run: move the old directory aside,
then run the script as a memory-capped service. Afterwards run
`dashboard/verify_publication.py record` on the fitting machine. The scheduled
refresh does not regenerate the preview, just as it does not regenerate the
racing qualities.
