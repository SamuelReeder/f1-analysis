# Complete retained-fit storage accounting

The [frozen inventory](overall_fit_storage_protocol.json) completed for every
required recovery design. The [recorded outputs](../outputs/championship/development/overall-fit-storage-v1/summary.json)
use the actual installed NumPyro collector and postprocessor, with the unchanged
model, coordinate maps and original preflight inputs. This calculation applies
the collector to a supplied state; it runs no sampler transition.

Every design retains 93 array fields. The complete inventories agree with every
saved attempt of the approved weekend-coordinate experiment, including failed
attempts. There are 7 archive comparisons across connected, confounded and
sparse-rookie designs; the other designs have no saved posterior to compare.

| Design | Scalar values per draw | Largest attempt payload, bytes | All fixed attempts retained, bytes |
|---|---:|---:|---:|
| Connected | 1,923 | 98,457,600 | 270,758,400 |
| Confounded | 1,917 | 98,150,400 | 269,913,600 |
| Sparse rookie | 1,789 | 91,596,800 | 251,891,200 |
| Dominant teammate | 1,923 | 98,457,600 | 270,758,400 |
| Era boundary | 1,929 | 98,764,800 | 271,603,200 |
| Real transfer network | 146,370 | 7,494,144,000 | 20,608,896,000 |

These values come from each design's `result.json` under the linked output
directory. They are **logical array payloads**, not measured physical peak
memory or compressed archive sizes. They exclude sampler state, compilation,
temporary diagnostic arrays, copies and archive overhead. Some arrays may alias
in memory; logical bytes cannot establish whether a fit will exceed its cap.

The [real-network inventory](../outputs/championship/development/overall-fit-storage-v1/real_network/result.json)
records 585,480 bytes per draw. Its fixed attempts retain 1,873,536,000;
3,747,072,000; 7,494,144,000; and 7,494,144,000 bytes respectively. The largest
single site would occupy 892,928,000 bytes after the original checker's float64
conversion at the largest retry, before its further statistical working arrays.
This motivates the separately defined [bounded diagnostic computation](overall_bounded_diagnostics.md),
which retains every chain, draw and coordinate.

The inventory workers ran sequentially under the recorded 3G memory and 200%
CPU scopes, in the unchanged shared slice with nice 19. The
[software verification](../outputs/championship/development/overall-fit-storage-checks/verification.json)
records 52 passing final checks; these overlap the earlier 47 checks and must
not be added together. Per-worker resource logs preserve both monotonic duration
and external wall-clock measurements, which do not agree exactly. Neither clock
is replaced with an inferred runtime.

Full sampler memory and runtime remain unmeasured for the untested designs.
This inventory supplies no new posterior, recovery/calibration acceptance,
functional ranking precision or permission to publish a ranking.
