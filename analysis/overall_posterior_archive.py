"""Bounded reading of a checked joint posterior's compressed NumPy archive.

All members are validated, even when the caller retains only selected sites.
Only requested sites become temporary read-only memory maps. The fitter owns
the shared receipt/source/checksum checks; this reader cannot accept a failed
fit or turn a synthetic/profile fit into scientific publication evidence.
"""
from contextlib import contextmanager, ExitStack
import math
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np

from f1rank.overall.fitting import checked_archive

READ_BYTES = 1024 * 1024


def _header(stream, member, expected_axes):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported posterior array format")
    if (len(shape) < 2 or shape[:2] != expected_axes
            or any(type(n) is not int or n < 0 for n in shape)):
        raise ValueError("Invalid posterior draw axes")
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufc":
        raise ValueError("Posterior arrays must be plain numeric values, never objects")
    size = math.prod(shape) * dtype.itemsize
    if stream.tell() + size != member.file_size:
        raise ValueError("Posterior array header and member length disagree")
    return shape, fortran, dtype, size


def _copy_and_validate(stream, size, dtype, destination):
    chunk_size = max(dtype.itemsize, READ_BYTES // dtype.itemsize * dtype.itemsize)
    while size:
        count = min(chunk_size, size)
        raw = stream.read(count)
        if len(raw) != count:
            raise ValueError("Truncated posterior array")
        if not np.isfinite(np.frombuffer(raw, dtype=dtype)).all():
            raise ValueError("Nonfinite posterior values")
        if destination is not None:
            destination.write(raw)
        size -= count
    if stream.read(1):
        raise ValueError("Unexpected trailing posterior bytes")


@contextmanager
def mapped_checked_posterior(run_dir, *, names, expected_key=None, expected_arrays_key=None,
                             temporary_parent=None):
    """Yield selected read-only maps only after checking the WHOLE archive.

    Every requested site must exist. Finite values and common chain/draw axes
    are checked for unselected sites too. Maps are valid only inside this context
    and are closed/deleted on every exit, including caller exceptions. Finish the
    context successfully before publishing anything derived from its values.
    """
    if (not isinstance(names, (list, tuple, set)) or not names
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Need unique posterior site names")
    wanted = set(names)
    with checked_archive(run_dir, expected_key) as (archive, receipt):
        if expected_arrays_key is not None and receipt["specification"]["arrays_key"] != expected_arrays_key:
            raise ValueError("Posterior training arrays do not match the forecast design")
        axes = (receipt["settings"]["chains"], receipt["attempts"][-1]["samples"])
        with zipfile.ZipFile(archive) as saved, tempfile.TemporaryDirectory(
                prefix="f1-overall-posterior-", dir=temporary_parent) as temporary:
            members = saved.infolist()
            keys = [m.filename[:-4] for m in members]
            if (not members or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.npy", m.filename) for m in members)
                    or len(set(keys)) != len(keys)):
                raise ValueError("Invalid or duplicate posterior archive members")
            if not wanted <= set(keys):
                raise ValueError("Posterior archive lacks required sites")
            retained_bytes = sum(m.file_size for m, key in zip(members, keys) if key in wanted)
            if shutil.disk_usage(temporary).free < retained_bytes + READ_BYTES:
                raise ValueError("Insufficient temporary disk for requested posterior sites")
            selected = {}
            try:
                for member, name in zip(members, keys):
                    with saved.open(member) as stream, ExitStack() as stack:
                        shape, fortran, dtype, size = _header(stream, member, axes)
                        path = Path(temporary) / member.filename
                        destination = stack.enter_context(path.open("wb")) if name in wanted else None
                        if destination is not None:
                            np.lib.format.write_array_header_2_0(destination,
                                dict(shape=shape, fortran_order=fortran, descr=np.lib.format.dtype_to_descr(dtype)))
                        _copy_and_validate(stream, size, dtype, destination)
                    if name in wanted:
                        selected[name] = np.load(path, mmap_mode="r", allow_pickle=False)
                yield selected, receipt
            finally:
                for value in selected.values():
                    value._mmap.close()
