"""Bounded finalization of a complete checkpoint attempt; no fit acceptance.

Reads at most one site's checkpoint chunk at a time. The resulting archives
retain every chain/draw/site and use ordinary numeric NPZ files. Only an atomic
directory rename seals an artifact; an interrupted partial directory is unusable.
The caller owns frozen input/attempt validation and the expected site inventory.
"""
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

from f1rank.overall.checkpoints import Checkpoints, canonical, digest, _sync_directory


def inventory(saved):
    """Read declared full axes without loading the complete posterior."""
    if saved.draws != saved.identity['settings']['samples']:
        raise ValueError('A complete retained draw inventory is required')
    if saved.warmup_completed != saved.identity['settings']['warmup']:
        raise ValueError('Warmup is incomplete')
    records = [r for r in saved.records if r['phase'] == 'sampling']
    if not records:
        raise ValueError('No retained checkpoint draws')
    result = {}
    for group, fields in records[0]['inventory'].items():
        result[group] = {name:dict(shape=[v['shape'][0], saved.draws, *v['shape'][2:]], dtype=v['dtype'])
                         for name,v in fields.items()}
    return result


def _write_archive(path, saved, group, fields):
    """Concatenate draw chunks inside each chain, without flattening chain axes."""
    records = [r for r in saved.records if r['phase'] == 'sampling']
    with path.open('xb') as file:
        with zipfile.ZipFile(file, mode='w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as out:
            for name, info in fields.items():
                shape, dtype = tuple(info['shape']), np.dtype(info['dtype'])
                with out.open(name+'.npy', mode='w', force_zip64=True) as member:
                    np.lib.format.write_array_header_2_0(member,
                        dict(shape=shape, fortran_order=False, descr=np.lib.format.dtype_to_descr(dtype)))
                    for chain in range(shape[0]):
                        for record in records:
                            source = saved.root/f'checkpoint-{record["generation"]:06d}'/(group+'.npz')
                            with np.load(source, allow_pickle=False) as archive:
                                value = archive[name]
                            expected = (shape[0], record['draws']-record['start_draw'], *shape[2:])
                            if value.shape != expected or value.dtype != dtype or not np.isfinite(value).all():
                                raise ValueError('Invalid retained checkpoint site')
                            # C-order serialization also handles Fortran-layout input and empty sites.
                            member.write(value[chain].tobytes(order='C'))
                            del value
        file.flush(); os.fsync(file.fileno())


def finalize(source, destination, *, identity, expected_inventory, verify_inputs):
    """Seal a complete numeric artifact without assigning convergence or publication."""
    verify_inputs()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError('Never replace a finalized attempt')
    timings = {}
    started = time.monotonic()
    with Checkpoints(source, identity) as saved:
        actual = inventory(saved)
        if actual != expected_inventory:
            raise ValueError('Require every expected posterior and sampler-extra site')
        verify_inputs()
        partial = Path(tempfile.mkdtemp(prefix='.partial-'+destination.name+'-', dir=destination.parent))
        hashes = {}
        for group in ('posterior', 'extras'):
            begun = time.monotonic()
            _write_archive(partial/(group+'.npz'), saved, group, actual[group])
            hashes[group+'.npz'] = digest(partial/(group+'.npz'))
            timings[group+'_archive_seconds'] = time.monotonic()-begun
        saved._require_lock(); saved._unchanged(); verify_inputs()
        record = dict(schema='checkpoint_numeric_archive_v1', state='draws_complete',
            identity_key=saved.identity_key, identity=identity, inventory=actual, files=hashes,
            checkpoint_generations=len(saved.records),
            checkpoint_tip_sha256=digest(saved.root/f'checkpoint-{len(saved.records)-1:06d}'/'manifest.json'),
            convergence_evaluated=False, publication_allowed=False, timings=timings,
            preseal_seconds=time.monotonic()-started)
        with (partial/'manifest.json').open('xb') as file:
            file.write(canonical(record)); file.flush(); os.fsync(file.fileno())
        _sync_directory(partial)
        saved._unchanged(); verify_inputs()
        partial.rename(destination)
        _sync_directory(destination.parent)
    return record
