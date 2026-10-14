# Saved-draw checks for the complete computational profile

The [diagnostic verifier](../analysis/overall_complete_profile_diagnostics.py)
has passing focused software checks in the
[prepared-check record](../outputs/championship/development/overall-prepared-profile-checks-v1/verification.json).
It has not run on the original archives. The profile was
[interrupted](overall_complete_profile_results.md), leaving no whole-run summary;
an interruption-aware review is needed before using that evidence. This verifier
does not modify the frozen fit or its settings.

First commit the complete original profile outputs, run the focused software
checks, and produce and commit the [metadata review](overall_complete_profile_review.md).
Then run this verifier in a separate capped scope under the existing resource
policy. The command accepts the committed review, a new output directory and
the name of the scope containing its process. It refuses to overwrite a previous
audit and independently requires the original fitting service to be terminal.

The verifier replays the metadata collector and requires its stable evidence to
match the committed review. It retains every case and every recorded attempt,
including failed convergence, interrupted children and missing archives. A
missing receipt or archive leaves verification incomplete; it cannot pass by
contributing an empty list of checks. A failure in an available attempt leaves
later attempts and cases in the report.

For each available, consistently recorded archive, the verifier checks its byte
identity and the complete retained-site inventory from the frozen storage
evidence. Every array must retain its original dtype, shape and grouped draw
axes, and every stored value must be finite. The existing bounded archive reader
uses temporary read-only memory maps and removes them on normal completion or
ordinary exceptions. No subset of fields, coordinates, chains or draws substitutes
for the saved archive. A process kill may leave temporary maps requiring cleanup;
the partial audit must be preserved and cannot establish a completed check.

The original bounded checker recomputes every coordinate report. The original
comparison function recomputes every current-driver qualifying and race contrast,
including its chain means and standard deviations. Reports are compared exactly,
with every unequal path retained. The original reports remain available alongside
the recomputed values. A reproduced convergence failure remains a failure.

The fitter did not archive its sampler extras. Divergence flags, tree-depth
counts and acceptance probabilities therefore cannot be independently recovered
from these arrays. Diagnostic reproduction reuses the recorded divergence count,
explicitly labels that dependency and retains the other unverified fields. It
does not claim to reproduce the complete sampler log or authenticate those counts.
Inputs and archives are checked again before a complete result is recorded.

The original resource registration and mid-run CPU-change records accompany the
audit. Its own capped runtime is measured separately. Report reproduction is an
engineering check; it establishes neither recovery, simulation-based calibration,
ranking precision, predictive improvement nor permission to publish.
