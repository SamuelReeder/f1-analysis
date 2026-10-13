# Bounded computation of the unchanged parameter diagnostics

The complete retained-array inventory shows why a finite-gradient preflight is
insufficient for full-fit memory planning. The current diagnostic checker casts
an entire site to float64 and constructs additional arrays for all its coordinates.
Compute the same coordinate-wise statistics in bounded groups instead. This is
a computational implementation change, not a new gate, model or sampling setting.

Keep every chain, iteration, parameter, coordinate, constant and empty site.
Use the original rank-normalized R-hat, bulk/tail ESS, mean MCSE/SD, divergence
handling, thresholds, deterministic-site policy, failure messages and report
structure from `f1rank.overall.diagnostics`. Preserve the original site order.
Only the independent trailing parameter-coordinate axis is partitioned; chain
and draw axes are never split into separate diagnostic problems.

Read coordinates directly from their original array layout, including read-only
memory maps and noncontiguous arrays. Do not reshape/cast an entire large site
into an extra in-memory copy. First validate all its coordinates and constant
status in bounded groups, then compute the existing statistics on each varying
group. Combine coordinate extrema to reproduce the original per-site report.
Retain the original whole-site behavior when any statistic is nonfinite. The
default coordinate-group size is an engineering constant fixed in the source;
it changes no statistical quantity or acceptance rule.

Verify complete report equality against the original checker on mixed numeric
arrays, scalar and multidimensional sites, noncontiguous layouts, missing/invalid
axes, nonfinite inputs, stochastic/deterministic constants, empty sites and
divergence failures. Recompute every saved attempt of the approved weekend profile
through the bounded implementation and require the entire original diagnostic
report to match, including failed attempts. This diagnostic-only reading never
promotes a failed archive into an accepted posterior.

Measure the bounded checker on the entire retained array inventory at the largest
registered retry dimensions for the real-network design. Use temporary memory
maps with all recorded shapes and dtypes. Fixed seeded synthetic values exercise
storage and diagnostics; they are not a joint model posterior and must not be
given a fit receipt. Keep all entries and all diagnostic results regardless of
whether the synthetic checker passes. Freeze the seed, formulas and workload in
the resource runner before execution. Check available disk space before allocation
and retain a failure if the complete workload cannot run under the scope cap.

Run checks and resource work sequentially in the standing 3G/200% `f1.slice` scope
with nice 19. Preserve source fingerprints, exact workload, construction and
diagnostic durations, process memory and the full result. Remove the temporary
arrays on normal or exception exit, retaining the compact evidence. An interrupted
process may leave ignored temporary data, which must be identified before cleanup.

This does not change the frozen numerical-profile runner or invalidate its saved
result. Integration belongs in a subsequent registered inference runner. The
resource fixture measures diagnostic working memory with mapped input; sampler
collection, possible host copies and full-fit memory remain separate requirements.
It supplies no new fit, calibration, functional-ranking convergence or publication.
