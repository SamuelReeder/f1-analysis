# Refreshing corrected results

The production integration branch adds source-revision handling to the scheduled
refresh. This is pipeline behavior, not an amendment to a statistical model,
registered comparison, sampler setting or acceptance gate. Existing failed fits
and the frozen Overall computation checkout remain historical records.

## Detecting work

`f1rank.schedule.check` retains the new-race and missing-timing checks. When those
do not require work, it compares the fitting input inventory and bytes with the
last published receipt. This includes processed data, tracked source-data inputs,
model/extraction code, contracts and the Overall model definition. Editorial
reports alone do not trigger fitting.

If those inputs match the publication, the check reads the latest processed
season's qualifying and race-result responses using the same complete paginator
as fetch. It compares only events through the latest processed race. It ignores
driver-row order and future qualifying; a changed value, added driver or removed
completed result requires refresh. Cached raw bytes must also match the hashes
recorded by the processed-data build. The check does not replace raw responses
or update retrieval history.

An outage or inconsistent pagination fails the check. A truncated page, changed
total, duplicate driver or inconsistent event metadata cannot replace a complete
cached season. No published pointer changes during this check.

## Fetch, failures and retry

The dashboard refresh invokes fetch with `--refresh-latest`. In addition to the
current calendar year, it refetches the latest processed season, so an off-season
correction reaches the build even after New Year. The existing fit, export and
publication stages still follow the build.

The comparison uses the published receipt because raw responses and processed
tables can advance before a later fit fails. That partial progress cannot make
the correction disappear from the next schedule check. The last published
snapshot remains available while the failed refresh is recorded. A missing
receipt requires refresh; a malformed receipt or pointer mismatch requires
inspection rather than treating the data as unchanged.

## Preserving source versions

Before replacing a changed raw response, fetch preserves its exact old bytes in
`data/raw/jolpica/archive/`, addressed by the old content hash. It records this
preservation before advancing the active response. Existing archive bytes are
never overwritten, including when distinct revisions share a retrieval timestamp
or a previously seen version returns. A corrupt archive stops the operation.

If the old retrieval record is absent or does not match the bytes, its original
retrieval time is explicitly unknown. The time it was archived does not prove
that it was available at an earlier forecast origin. This change does not repair
the historical source-availability gap documented in
[Overall source vintages](overall_source_vintages.md).

Active responses and the raw manifest are replaced through complete temporary
files. The cache archive retains the version history and skips abandoned
temporary files. The checks exercise interrupted replacement, not physical
power-loss or filesystem durability guarantees.

## Verification and remaining integration

The [committed focused check record](../outputs/championship/development/overall-refresh-correction-checks-v1/verification.json)
contains **89 passed** and **2 deselected** tests, with their command, source
fingerprints, output and resource log. It includes unchanged and revised sources,
the real fetch/build path followed by a synthetic fit failure, subsequent retry,
raw-history preservation, pagination failures and cache restoration. Network
responses are synthetic. The deselected checks need regenerated local fit/export
artifacts; this record does not establish fresh-output or hosted-runner readiness.

Automatic upstream revision polling currently covers qualifying and race results
in the latest processed season. Older seasons, sprint responses and revised
FastF1 timing require an explicit source refresh. The accepted Overall producer,
its full refresh/cache dependencies and end-to-end validation remain required.
Changed hashed source requires regenerated publication artifacts before promotion.
No push, deployment or cache-release upload is performed by this implementation.
