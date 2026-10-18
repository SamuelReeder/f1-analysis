# Recorded source vintages and forecast-time evidence

The [committed inventory](../outputs/championship/development/overall-source-vintage-inventory-v1/inventory.json)
identifies the processed provenance manifests and generator source available for
the successor's forecast-origin review. Its input hashes and source revision bind
the inspected versions. The
[verification record](../outputs/championship/development/overall-source-vintage-inventory-v1/verification.json)
records a byte-identical repeat and checks the record inventories and file-time
flags against those committed manifests. Collection uses the Python standard
library and reads metadata and source text only.

| Manifest group | Metadata records | Recorded retrieval dates, UTC | File-time fallback explicitly recorded |
|---|---:|---|---:|
| Qualifying/results source files (`sources.json`, `jolpica`) | 42 | 2026-09-27 through 2026-09-28 | 42 |
| Race timing (`race_sources.json`, `races`) | 187 | 2026-09-28 | 0 |
| Sprint timing (`race_sources.json`, `sprints`) | 29 | 2026-09-28 | 0 |
| Telemetry (`race_sources.json`, `telemetry`) | 138 | 2026-09-28 | 0 |
| Practice (`practice_sources.json`, `sessions`) | 107 | 2026-10-02 | 0 |

These counts describe metadata entries. The groups overlap in events and must
not be added to claim distinct races. The inventory preserves every recorded
timestamp, missing-time record and file-time flag. The FastF1 groups omit the
file-time flag; their zero explicit fallback counts do not independently
authenticate their timestamps.

The inspected [qualifying builder](../f1rank/build.py) uses a raw manifest's
retrieval time when its hash matches; otherwise it records the raw file's
modification time with an explicit fallback flag. All inventoried Jolpica
entries use that fallback. A copied or restored file's modification time cannot
by itself establish when that content became available to a forecast.

The inspected [race extractor](../extract/race_extract.py),
[practice extractor](../extract/practice.py) and
[telemetry extractor](../extract/telemetry.py) record the current extraction
time and output hashes. These identify the recorded extraction versions without
establishing their availability before an earlier historical forecast origin.
The [Jolpica fetcher](../f1rank/fetch.py) also has an archive path for changed
responses.

A subsequent [local metadata inspection](../outputs/championship/development/overall-local-source-vintage-inventory-v1/record/inventory.json)
on 2026-10-18 found no Jolpica archive directory in either the active worktree or
the primary checkout's raw cache. The primary cache contains all 42 raw files
referenced by the processed qualifying/results manifest, but its raw retrieval
manifest contains no entries for those files. The worktree has no local Jolpica
raw directory. The [verification record](../outputs/championship/development/overall-local-source-vintage-inventory-v1/record/verification.json)
checks every filename and local presence observation against the committed
processed metadata and preserves a copy of the primary cache's raw manifest.
This inspection reads provenance metadata only. It does not hash raw responses,
search other archive locations or establish that older versions cannot exist
elsewhere. These inspected local paths supply no archived retrieval evidence
for the required historical forecast origins.

The [comparison-origin checks](overall_comparison_origins.md) already require
exact forecast times and common content identities. The remaining controller
work must bind those identities to preserved input bytes and evidence of their
availability at each declared origin. Training cutoffs alone do not establish
the vintage of later-corrected historical values. Supplemental files, official
grid announcements and other forecast inputs need the same origin-specific
review. Their inclusion in a current manifest is insufficient evidence of prior
availability.

This inventory leaves historical forecast availability unverified. It has not
reproduced raw content hashes, inspected observation arrays or provider revision
histories, or evaluated any forecast origin. It supplies neither a leakage
finding nor a passing as-of replay. Historical development, the separate
independent-evidence registration and scientific acceptance retain their
existing requirements. The running fit, frozen inputs and acceptance rules are
unchanged.
