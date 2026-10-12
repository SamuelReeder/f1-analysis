"""Bounded legacy-format reads; byte/design identity is not fit acceptance.

The caller must separately verify fit source, numerical diagnostics, registration
and source availability before evaluating a baseline. Old metadata is not an
Overall fit receipt. This reader never converts it into one.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re

from analysis.overall_posterior_archive import _mapped_numeric_archive
from f1rank.fit import check, meta_path
from f1rank.overall.fitting import arrays_digest


def _sha(value):
    return hashlib.sha256(value).hexdigest()


@contextmanager
def mapped_legacy_posterior(path, design, *, names, expected_archive_sha256,
                             expected_metadata_sha256, expected_training_arrays_key,
                             temporary_parent=None):
    """Check every array and scalar; retain requested draws as read-only maps.

    Expected identities must come from the caller's independently checked input
    inventory. They prevent substitution, but do not establish that the original
    fit was valid. Successful exit rechecks the design, sidecar, opened archive
    and path identity. Do not persist derived outputs before successful exit.
    """
    for value in (expected_archive_sha256, expected_metadata_sha256, expected_training_arrays_key):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Expected legacy identities must be explicit SHA-256 values")
    path = Path(path).resolve()
    metadata_path = meta_path(path)
    original_metadata = metadata_path.read_bytes()
    if _sha(original_metadata) != expected_metadata_sha256:
        raise ValueError("Legacy metadata differs from the declared inventory")
    metadata = json.loads(original_metadata)
    if (not isinstance(metadata, dict) or metadata.get("model_kw") != {} or metadata.get("start_season") != 2010
            or metadata.get("sprint_quali_until") is not None):
        raise ValueError("Legacy baseline requires the standard qualifying-only model")
    settings = metadata.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Legacy metadata must record the fitted chain/draw settings")
    axes = (settings.get("chains"), settings.get("samples"))

    def verify_design_and_metadata():
        if metadata_path.read_bytes() != original_metadata:
            raise ValueError("Legacy metadata changed while reading")
        if check(path, design) != json.loads(original_metadata):
            raise ValueError("Legacy metadata changed during the design check")
        if arrays_digest(dict(qualifying=design.arrays())) != expected_training_arrays_key:
            raise ValueError("Legacy training arrays differ from the declared inventory")

    verify_design_and_metadata()
    with path.open("rb") as handle:
        def verify_archive():
            handle.seek(0)
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected_archive_sha256:
                raise ValueError("Legacy archive differs from the declared inventory")
            opened, current = os.fstat(handle.fileno()), path.stat()
            if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                raise ValueError("Legacy archive was replaced")
            handle.seek(0)

        verify_archive()
        with _mapped_numeric_archive(handle, names=names, axes=axes, scalar_prefix="info_",
                forbidden_prefix="overall_", temporary_parent=temporary_parent) as (posterior, info):
            yield posterior, info, dict(purpose="development_legacy_archive_read", metadata=metadata,
                archive_sha256=expected_archive_sha256, metadata_sha256=expected_metadata_sha256,
                training_arrays_key=expected_training_arrays_key, posterior_shape=list(axes),
                convergence_verified=False, fit_source_verified=False,
                source_availability_verified=False, publication_allowed=False)
        verify_design_and_metadata()
        verify_archive()
