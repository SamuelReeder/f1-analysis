# Canonical simulation controller: development boundary

This definition precedes implementation and execution. The controller connects
the checked joint-posterior reader, fixed common calendar, paired explanatory
simulator and bounded [summary](overall_summary.md). It implements a dependency
of the canonical producer; it does not create another public Overall product.

The initial executable boundary accepts synthetic development fits only. Real
driver scoring still requires the complete scientific protocols specified by
the [candidate contract](overall_candidate.md). Neither numerical convergence
nor a caller-supplied success flag supplies that missing authorization. The
controller always leaves scientific acceptance and publication false.

## Frozen inputs

Before execution, commit a frozen development simulation protocol. It must bind
the exact fit receipt and posterior archive, training-array and complete catalog
identities, grouped posterior axes, full reference-calendar bytes, controller
and dependency sources, seed, draw-block size, replicate-block size, fixed
replicate count, qualifying format and every reference event's explicit
race-control assumption. These are simulation decisions, not fitted component
weights. A protocol change produces a different simulation identity and must
not be used to rerun a failed measurement until it appears favorable.

The design's current race field and its driver-entry mapping must agree with
the fit catalog and its training arrays. Missing qualifying observations do
not remove entrants from that field. No parameter may be filled with zero,
borrowed from another fit or paired with a differently shaped draw inventory.
The checked archive reader validates every saved member, including unused
members, and retains its source/receipt/archive checks through successful exit.

The common calendar contains every declared event with equal weight. Translate
its named circuit and observed wet/neutralisation category into the fitted
context axis using the design's circuit catalog. Never reuse numeric context
indices from a different training catalog, drop an unseen reference circuit or
insert an average coefficient. Keep all unresolved assumptions recorded in
the frozen reference artifact; translating it is not proof of calibration or
verification of its historical source vintages.

## Complete deterministic schedule

Visit all chains and all within-chain draws. Partition draws and season
replicates into consecutive bounded ranges without empty or singleton blocks,
as required by the existing simulator. When a final singleton would occur,
move a sample from the preceding block into that final block. Reject an
impossible declared partition instead of dropping or duplicating a sample.

Record the complete range schedule before simulation. Each random stream is
addressed by the fixed seed, chain, draw-range start and replicate-range start.
The controller supplies this address to the paired generator and supplies the
same coordinates to the accumulator. Only a baseline and its declared probes
share an address. The frozen layout is part of the simulation identity: a new
block size cannot be presented as the same numerical experiment.

All simulator state and observation sites come from matching slices of the
single checked posterior. Select the current driver-entry indices once and
retain the full fitted context axis. Paired explanatory probes use the same
field, parameters and scenario. No adaptation to observed points, diagnostic
failures, preferred drivers or estimated simulation precision is permitted.

## Failure and artifact behavior

An execution directory is immutable: refuse to reuse an existing directory,
including one with a failed or incomplete receipt. An incomplete file alone
does not establish that a process stopped. The operator must inspect the actual
process before planning recovery. This controller supplies no automatic retry,
resume, replacement season or fallback posterior.

Record the frozen inputs, complete schedule and a running receipt before
consuming the archive. Record completed blocks as computation progresses. On a
failure, preserve the exception and progress; never expose a partial summary as
a completed result. Do not write a result until every draw/replicate is present,
the archive context exits successfully, and the protocol, reference, design and
all registered sources have been rechecked. A completed development computation
still reports insufficient precision when its fixed workload misses the
existing precision checks. It never publishes that result.

The receipt distinguishes computational completion, conditional simulation
precision and scientific acceptance. The first two do not establish MCMC
precision, near-tie rank precision, recovery, predictive calibration or the
independent evidence needed by the canonical exporter.

## Required checks and remaining work

Check the calendar/context mapping, exact partitions, complete draw pairing,
deterministic stream schedule, direct versus controlled summaries, immutable
execution directories, preserved failures and source/archive/protocol mutation
rejection. Use explicitly synthetic archives and fields for these checks; they
are not evidence that any real fit converged.

The accepted real-data protocol/evidence controller, official-grid and completion
assumptions, calibrated reference scenario, full-workload resource measurement,
canonical payload/export and refresh/cache rollout remain required. The failed
original numerical profile and its separately pending experiment are unchanged.
