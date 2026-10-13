# Complete-design recovery computation preflight

The approved conditional weekend-coordinate profile now converges on its entire
inventory. A full recovery study also requires the dominant-teammate, regulation
boundary and real transfer-network designs. Before selecting that study's
replicate counts or requesting new fits, run this separate computation preflight
on every design in `f1rank.overall.recovery.SYNTHETIC_CASES`.

This is a fixed generation, probability-equivalence and gradient calculation,
not MCMC, calibration or a driver ranking. It cannot turn the earlier profile
into a complete recovery study. Its inspected synthetic datasets must not later
be represented as untouched calibration replicates.

## Fixed inventory and calculation

The separate frozen JSON protocol lists every case and its generation seed,
source fingerprints and scope limits before execution. Preserve the original
generator, all responses, parameter values and conditioning information. The
dominant-teammate case retains its fixed stress parameters. The real-network
case uses the complete original design and generated responses, without
subsampling observations, entrants or latent directions.

For each case, capture the original generating parameters through the existing
`generate_with_parameters` helper. Record the complete block shapes and byte
counts, catalogue dimensions, original truth/parameter/array fingerprints and
latent dimensions. Build the same data-only lap-control rotation and conditional
weekend transformation as the approved experiment. Transform the supplied
generating parameters; generate no replacement parameters.

Evaluate both complete constrained joint log densities. Check the transformed
density against the original plus the lap-control map's log absolute Jacobian;
the conditional weekend map has unit absolute determinant. Check every original
sample/deterministic state and each observed log-probability factor, preserving
all entries. Use the tolerances already present in
`tests/test_overall_weekend_coordinates.py`: density/factor absolute tolerance
0.005 and relative tolerance 0.000002; reconstructed-state absolute and relative
tolerances 0.000003. Record discrepancies, and retain a failure rather than
loosening a tolerance or choosing another seed.

Convert the transformed generating parameters to unconstrained coordinates with
the installed NumPyro support transforms. Compile and evaluate one complete
potential-energy value and reverse-mode gradient at those coordinates. Require
finite values and retain all latent directions. No warm-up, transition, posterior
sample, alternative initialization or retry is performed. Record runtime versions,
coordinate counts and the phase durations. This checks one supplied point per
design; it does not establish good posterior geometry or convergence.

## Resource and failure accounting

Process the complete inventory sequentially. Each case runs in its own systemd
user scope under `f1.slice`, with MemoryMax 3G, CPUQuota 200% and nice 19. Retain
the shared slice's existing CPU quota and weight. Record actual scope properties,
the full child output, process exit status, GNU time report and child progress.
No concurrent heavy check is launched. The parent only performs orchestration
and retains metadata; numerical work stays inside the capped child.

Every case is attempted once. Generation, equivalence, gradient, memory-limit or
other process failures remain explicit. Continue to account for the remaining
cases without rerunning a failure. An existing output directory is rejected.
Source or protocol changes invalidate the calculation rather than silently
starting a different experiment. All scientific/publication flags remain false.

The scope cap follows the standing rule for non-fitting work. A failure at this
cap would not prove that a fit cannot run under a separately registered larger
cap. A successful gradient would not measure multi-chain warm-up, sampling,
archive storage or the full nested simulator. Full study power, resources,
functional precision and an approved fitting registration remain required.
