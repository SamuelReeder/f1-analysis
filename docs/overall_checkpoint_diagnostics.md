# Diagnostics for a complete checkpoint artifact

The isolated [diagnostic adapter](../analysis/checkpoint_diagnostics/__init__.py)
connects finalized checkpoint archives to the existing bounded parameter and
current-driver-contrast checks. It is prepared for verification; no production
runner selects it, and no completed-model diagnostic result is claimed here.

The caller must supply the frozen attempt identity, complete collector inventory,
catalog and expected manifest digest. The adapter verifies them before reading,
rejects missing or additional members, checks all retained arrays, and validates
the numeric archive hashes again after computation. Source verification runs
before and after diagnostics. Only numeric arrays are read; serialized code or
pickled objects are not accepted.

Every declared sampler extra is retained. Divergences, tree-depth-limit frequency
and acceptance probability are recomputed from the complete arrays. Every
posterior site enters the existing bounded parameter checks, and all current
driver pairs enter the existing qualifying/race contrast reports. The original
thresholds and distinction between parameter and contrast reports are unchanged.
Temporary read-only memory maps avoid loading all posterior sites into RAM;
temporary disk requirements and full-scale memory still need measurement.

Failed diagnostics are returned intact. This adapter cannot select another
attempt, extend sampling, change the fixed retry ladder or authorize publication.
It records diagnostic software separately from the sampler's saved runtime so
CPU-based analysis cannot be mistaken for CPU-based fitting. Archive validation,
parameter checks and contrast-report timing are recorded separately.

The accompanying checks compare archive-derived reports with the complete
in-memory reference reports, include a deliberately failed fixture, and reject
changed archive bytes, identities, inventories and sources. They are software
verification of the adapter, not model recovery, calibration or an accepted fit.
