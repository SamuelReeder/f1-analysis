# Full retained-fit storage inventory

This definition precedes the inventory calculation. The successful complete-design
preflight measured one potential and gradient, whereas fitting also retains
posterior arrays and evaluates diagnostics. Count that complete workload before
selecting resources for further inference. This performs no new MCMC and changes
no model, sampler, gate or registered preflight result.

## Exact retained array inventory

Use every case, original seed and model/coordinate source from the completed
recovery preflight. Regenerate only those already inspected inputs and require
their observation, generating-parameter and truth fingerprints to match their
saved preflight records. Transform the supplied parameters with the unchanged
control and weekend coordinate maps.

Apply NumPyro's actual collection/postprocessing function to that one supplied
unconstrained state, returning every sample and deterministic site. It is a
collector-format exercise, not a sampler transition or posterior draw. Verify
agreement with direct postprocessing. Bind the installed NumPyro implementation
files and runtime identity; do not assume another version retains the same fields.

Record every site's name, shape, dtype, element count and logical bytes. For each
attempt in the existing fixed retry ladder, multiply this inventory by its full
chain/draw shape. Retain every attempt in the accounting. Also report the sum
across the whole ladder, since failed archives remain on disk, and the largest
single site's float64 array size under the current diagnostic conversion.

These are logical array payload sizes, not measured physical peak memory or
compressed disk sizes. They omit sampler/gradient/compilation state, diagnostic
temporaries, array copies, compression buffers and archive headers. Aliased arrays
can share memory. Do not infer that adding or doubling logical bytes gives the
actual sampler peak, or that compression will provide a particular saving.

## Independent checks and resources

Compare the complete postprocessing inventory with the headers of every saved
attempt from the approved conditional weekend-coordinate profile, including
failures. Bind each archive's existing fingerprint; require identical field names,
per-draw shapes and dtypes, and complete registered chain/draw axes. Keep all
attempts in this check. A header check is not a new convergence certificate.

Run each design in a sequential systemd user scope with MemoryMax 3G, CPUQuota
200%, nice 19 and the unchanged shared `f1.slice`. Record the actual scope
properties, runtime, process resource output and complete success/failure inventory.
Do not allocate a full-size posterior or reduce any dimension to make this
calculation fit. Reject an existing output directory. Preserve a failure without
selecting a different seed or changing a numerical setting.

Check byte arithmetic against explicitly allocated small arrays, including
multidimensional and empty sites and different numeric dtypes. Check the archive
comparison with independently written NumPy archives, including missing, extra,
reordered and malformed inventories. The complete real-network accounting must
use the actual postprocessor, not a scaled small-field fixture.

This inventory informs the next implementation and resource decision. It does
not establish full-fit memory, convergence, recovery/calibration, functional
ranking precision, intended-runner resources or publication acceptance.
