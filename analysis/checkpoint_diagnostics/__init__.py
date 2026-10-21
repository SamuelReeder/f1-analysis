"""Complete bounded diagnostics for a sealed checkpoint artifact; no acceptance.

The caller supplies the registered identity, collector inventory and catalog.
Completion of this function cannot select an attempt, change the retry ladder,
or authorize publication. Failed diagnostics are returned intact.
"""
from importlib.metadata import version
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from analysis.overall_posterior_archive import _mapped_numeric_archive
from f1rank.overall.checkpoints import canonical, digest
from f1rank.overall.bounded_diagnostics import check
from f1rank.overall.inference import contrast_reports, validate_extras


def _read_manifest(directory, expected_sha256, expected_identity_key, expected_inventory):
    path=directory/'manifest.json'
    if digest(path)!=expected_sha256:
        raise ValueError('Finalized artifact manifest changed')
    record=json.loads(path.read_text())
    import hashlib
    if (record.get('schema')!='checkpoint_numeric_archive_v1' or record.get('state')!='draws_complete'
            or record.get('publication_allowed') is not False or record.get('convergence_evaluated') is not False
            or record.get('identity_key')!=expected_identity_key
            or hashlib.sha256(canonical(record['identity'])).hexdigest()!=expected_identity_key
            or record.get('inventory')!=expected_inventory or set(record['files'])!={'posterior.npz','extras.npz'}):
        raise ValueError('Require the complete registered finalized attempt')
    for name,checksum in record['files'].items():
        path=directory/name
        if path.resolve().parent!=directory or digest(path)!=checksum:
            raise ValueError('Finalized numeric archive changed')
    return record


def _require_members(path, fields):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(fields) or set(names)!={n+'.npy' for n in fields}:
            raise ValueError('Require exactly every registered archive member')


def diagnose(directory, *, expected_manifest_sha256, expected_identity_key, expected_inventory,
             catalog, verify_inputs, temporary_parent=None):
    directory=Path(directory).resolve();verify_inputs()
    record=_read_manifest(directory,expected_manifest_sha256,expected_identity_key,expected_inventory)
    settings=record['identity']['settings'];axes=(settings['chains'],settings['samples'])
    timings={};begun=time.monotonic()
    for group in ('posterior','extras'):
        _require_members(directory/(group+'.npz'),expected_inventory[group])
    started=time.monotonic()
    with np.load(directory/'extras.npz',allow_pickle=False) as archive:
        extras={n:archive[n] for n in archive.files}
    validate_extras(extras,settings)
    for n,a in extras.items():
        if list(a.shape)!=expected_inventory['extras'][n]['shape'] or a.dtype.str!=expected_inventory['extras'][n]['dtype']:
            raise ValueError('Changed sampler-extra shape or dtype')
    divergences=int(extras['diverging'].sum())
    tree_fraction=float(np.mean(extras['num_steps']>=2**settings['max_tree_depth']-1))
    acceptance=float(np.mean(extras['accept_prob']))
    timings['sampler_extras_seconds']=time.monotonic()-started
    started=time.monotonic()
    with _mapped_numeric_archive(directory/'posterior.npz',names=list(expected_inventory['posterior']),
            axes=axes,temporary_parent=temporary_parent) as (post,_):
        for name,a in post.items():
            info=expected_inventory['posterior'][name]
            if list(a.shape)!=info['shape'] or a.dtype.str!=info['dtype']:
                raise ValueError('Changed posterior collector shape or dtype')
        timings['posterior_mapping_and_validation_seconds']=time.monotonic()-started
        started=time.monotonic();diagnostic=check(post,divergences)
        timings['parameter_diagnostics_seconds']=time.monotonic()-started
        started=time.monotonic();comparisons=contrast_reports(post,catalog,divergences)
        timings['driver_contrasts_seconds']=time.monotonic()-started
    diagnostic.update(divergences=divergences,comparison_reports=comparisons,
                      tree_depth_limit_fraction=tree_fraction,mean_accept_prob=acceptance)
    verify_inputs()
    final=_read_manifest(directory,expected_manifest_sha256,expected_identity_key,expected_inventory)
    if final!=record:raise ValueError('Artifact changed during diagnostics')
    return dict(state='diagnostics_complete',identity_key=expected_identity_key,
        manifest_sha256=expected_manifest_sha256,diagnostics=diagnostic,
        sampling_runtime=record['identity'].get('runtime'),
        diagnostics_runtime={name:version(name) for name in ('numpy','arviz-stats')},
        timings=timings,seconds=time.monotonic()-begun,publication_allowed=False,
        scientific_acceptance_established=False,attempt_selection_performed=False)
