# Saved-array audit for the corrected computational profile

The [prepared verifier](../analysis/overall_stable_profile_diagnostics.py) checks
every recorded attempt in the [corrected profile](overall_stable_profile.md),
including failed retries. It has not been executed and its
[focused tests](../tests/test_overall_stable_profile_diagnostics.py) remain unrun
while the profile occupies the fitting allocation.

After the fitting service terminates, preserve and commit its original outputs,
test the metadata reviewer and this verifier, and produce and commit the
[complete metadata review](overall_stable_profile_review.md). Then invoke the
verifier in the separate diagnostic scope required by the existing resource
policy, with the committed review, a new output directory outside the original
profile output directory, and the containing scope name:

```sh
python -m analysis.overall_stable_profile_diagnostics \
  --review <committed-review-path> \
  --output <new-audit-directory> \
  --scope-name <containing-scope-name>
```

The verifier replays the terminal metadata review before and after the numerical
audit. The original frozen registration supplies the complete preflight site
inventory. Each posterior array must retain its dtype, dimensions and full
grouped chain/draw axes; no fields, coordinates or draws are selected according
to their outcome. The bounded archive reader checks all values and releases
temporary maps on ordinary exit. A killed process can leave temporary files and
an incomplete audit; neither supplies acceptance.

The current fit also retains its sampler extras. The verifier checks their
complete inventory, axes, types and valid ranges, then recounts divergences,
tree-depth-limit frequency and mean acceptance probability. It uses the saved
divergence flags when recomputing every coordinate report and current-driver
qualifying and race contrast. Numerical comparisons are exact; coordinate
report list order is normalized by parameter identity, without dropping fields.
Every unequal path remains in the audit. Historical runtime metadata is not
recreated and is identified separately from recomputed statistics.

Missing, corrupt or inconsistent evidence leaves that attempt unverified and
does not hide later attempts or cases. A faithfully reproduced convergence
failure stays failed. Reproducing all recorded diagnostics establishes agreement
with saved arrays, not recovery, calibration, ranking precision, scientific
acceptance or publication permission. The audit does not run a sampler or alter
the current experiment.
