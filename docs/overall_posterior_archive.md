# Bounded loading of a joint posterior

[`overall_posterior_archive.py`](../analysis/overall_posterior_archive.py) reads a
compressed posterior through the fitter's shared `checked_archive` context.
That context requires a converged receipt, the requested run identity when
supplied, current model/data sources and a matching archive checksum. It streams
the checksum instead of copying the compressed archive into memory. A successful
exit rechecks the receipt, source, open archive and file identity.

The existing full-array loader uses the same context. Its behavior remains
appropriate for callers that need every array in memory; the mapped reader is
for consumers that need bounded processing and only selected sites. No failed
or stale receipt becomes current through this refactor. The changed fitter
source requires regenerated evidence, as recorded in the
[candidate contract](overall_candidate.md#shared-archive-verification).

## Archive and array checks

Requested site names must be unique and present. The reader rejects duplicate
or invalid archive-member names and copies no arbitrary archive paths. It parses
the numeric array header, checks the common chain/draw axes and declared payload
length, and streams the decompressed payload in bounded chunks. Object arrays are
rejected without unpickling. Every array must contain finite values, including
sites the caller does not retain. An invalid unused site invalidates the archive.

Only requested sites are written to temporary arrays and opened as read-only
memory maps. Array order and byte order are preserved. The maps exist only
inside `mapped_checked_posterior`; they are closed and their temporary directory
is removed on normal exit or an exception. A caller must finish the context
successfully before committing any derived artifact. Partial processing and an
exception at context exit cannot advance a publication pointer.

The optional `expected_arrays_key` binds the receipt to the forecast design's
training arrays. The forecast implementation independently checks its consumed
parameters and preserves their common draw axes. A matching checksum or receipt
does not establish scientific acceptance: recovery, calibration, independent
predictive evidence and the publication contract remain required. A numerical
profile or software fixture cannot supply those conditions.

## Verification and resource boundary

[`test_overall_posterior_archive.py`](../tests/test_overall_posterior_archive.py)
uses explicitly synthetic cache-format fixtures. It checks round trips against
the full loader, exact forecast agreement, selected-site retention, finite-value
checks on unused sites, array layouts, bounded reads, cleanup and rejection of
changed sources, receipts, files and inputs. These fixtures exercise the format's
convergence field; they do not represent MCMC convergence.

The compressed mode of
[`overall_forecast_resources.py`](../analysis/overall_forecast_resources.py)
uses the committed data dimensions and existing maximum retry draw shape, filled
with synthetic parameter values. It writes a temporary compressed format fixture
with an additional unused history array, then reads and validates the whole
archive and forecasts in batches. Its convergence Boolean is a fixture field,
explicitly marked as such. No fitter, diagnostics study, real prediction or
driver ranking is run. The source/checksum checks themselves execute normally.

The local workload includes compression, loading, complete draw processing and
successful context exit. It verifies the enforced local memory and CPU limits.
It does not establish MCMC fitting resources, intended-CI-runner readiness or
end-to-end refresh behavior. Temporary workload files are discarded; only
identity, coverage, timing and resource measurements are retained.

The [verification record](../outputs/championship/development/overall-archive-checks/verification.json)
links the passing full suite and local compressed workload. The consumed
parameter identity and complete forecast-state checksum match the preceding
uncompressed workload. It separately verifies that the sampling functions and
settings were unchanged, and that the probability-model and generator sources
still match the original failed numerical profile. The pending experiment's
source fingerprint is current; its approval status remains pending.
