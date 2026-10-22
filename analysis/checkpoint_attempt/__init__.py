"""One complete checkpoint attempt, with separate sampling and diagnostic workers.

Callers own frozen registration and resource admission. These adapters preserve
all numeric artifacts and failed diagnostics; they cannot grant publication.
"""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from analysis.checkpoint_archive import finalize, inventory
from analysis.checkpoint_diagnostics import _read_manifest, diagnose
from analysis.checkpoint_execution import _write, execute
from f1rank.overall.checkpoints import Checkpoints, canonical, digest
from f1rank.overall.diagnostics import (
    DETERMINISTIC_SITES, MAX_DIVERGENCE_RATE, MAX_MCSE_SD_RATIO, MIN_CHAINS,
    MIN_ESS, RHAT_LIMIT,
)
from f1rank.overall.inference import EXTRA_FIELDS


def attempt_key(specification):
    return hashlib.sha256(canonical(specification)).hexdigest()


def _inventory(specification):
    fields = specification['inventory']
    axes = [specification['settings']['chains'], specification['settings']['samples']]
    if (set(fields) != {'posterior', 'extras'} or not fields['posterior']
            or set(fields['extras']) != set(EXTRA_FIELDS)):
        raise ValueError('Require the complete posterior and every sampler extra')
    for group in fields.values():
        for name, info in group.items():
            dtype = np.dtype(info['dtype'])
            if (not name.isidentifier() or info['shape'][:2] != axes
                    or any(type(n) is not int or n < 0 for n in info['shape'])
                    or dtype.hasobject or dtype.kind not in 'biuf'):
                raise ValueError('Require finite numeric collector axes and dtypes')
    return fields


def _artifact(directory, specification, *, manifest_sha256=None):
    """Validate the sealed inventory/bytes AND their original checkpoint identity."""
    directory = Path(directory).resolve()
    manifest = directory / 'archive/manifest.json'
    identity = json.loads((directory / 'checkpoints/identity.json').read_bytes())
    key = attempt_key(specification)
    if (identity['run_key'] != key or identity['settings'] != specification['settings']
            or identity['chunk_draws'] != specification['chunk_draws']
            or identity['chain_method'] != specification['chain_method']):
        raise ValueError('Checkpoint belongs to another declared attempt')
    identity_key = hashlib.sha256(canonical(identity)).hexdigest()
    saved_manifest_sha256 = manifest_sha256 or digest(manifest)
    record = _read_manifest(directory / 'archive', saved_manifest_sha256,
                            identity_key, _inventory(specification))
    if record['identity'] != identity:
        raise ValueError('Final archive belongs to a different checkpoint runtime')
    with Checkpoints(directory / 'checkpoints', identity) as saved:
        if (inventory(saved) != specification['inventory']
                or record['checkpoint_generations'] != len(saved.records)
                or record['checkpoint_tip_sha256'] != digest(
                    saved.root / f'checkpoint-{len(saved.records)-1:06d}' / 'manifest.json')):
            raise ValueError('Final archive checkpoint inventory changed')
    return record, saved_manifest_sha256


def _compare_checkpoint_archive(directory, specification):
    """Reconcile a sealed archive without a result by comparing EVERY stored byte.

    This is used only after interruption between atomic archive sealing and the
    worker result. The old archive is never replaced or trusted from a flag alone.
    """
    from analysis.overall_posterior_archive import _header
    directory = Path(directory)
    identity = json.loads((directory / 'checkpoints/identity.json').read_bytes())
    with Checkpoints(directory / 'checkpoints', identity) as saved:
        rows = [row for row in saved.records if row['phase'] == 'sampling']
        for group, fields in specification['inventory'].items():
            with zipfile.ZipFile(directory / 'archive' / (group + '.npz')) as archive:
                if (len(archive.namelist()) != len(fields)
                        or set(archive.namelist()) != {n + '.npy' for n in fields}):
                    raise ValueError('Incomplete numeric archive members')
                for name, info in fields.items():
                    member = archive.getinfo(name + '.npy')
                    with archive.open(member) as stream:
                        shape, fortran, dtype, _ = _header(
                            stream, member, tuple(info['shape'][:2]))
                        if list(shape) != info['shape'] or fortran or dtype.str != info['dtype']:
                            raise ValueError('Changed finalized collector layout')
                        for chain in range(shape[0]):
                            for row in rows:
                                source = saved.root / f'checkpoint-{row["generation"]:06d}' / (group + '.npz')
                                with np.load(source, allow_pickle=False) as values:
                                    expected = values[name][chain].tobytes(order='C')
                                if stream.read(len(expected)) != expected:
                                    raise ValueError('Sealed artifact differs from its checkpoints')
                        if stream.read(1):
                            raise ValueError('Unexpected archive trailing data')


def sample_stage(directory, specification, *, model, args, kwargs, verify_inputs,
                 observe=lambda: None):
    from analysis.checkpoint_sampling import run_chunks
    directory = Path(directory).resolve()
    result_path = directory / 'sample_and_archive/result.json'
    if result_path.exists():
        raise FileExistsError('A completed sampling stage is verified, never rerun')
    verify_inputs(); _inventory(specification)
    observations = [observe()]
    def checkpoint(row):
        observations.append(dict(progress=row, resource=observe()))
    sampling = run_chunks(model, args, kwargs=kwargs, settings=specification['settings'],
        run_key=attempt_key(specification), directory=directory / 'checkpoints',
        chunk_draws=specification['chunk_draws'], verify_inputs=verify_inputs,
        chain_method=specification['chain_method'], on_checkpoint=checkpoint)
    identity = json.loads((directory / 'checkpoints/identity.json').read_bytes())
    if (directory / 'archive').exists():
        _artifact(directory, specification)
        _compare_checkpoint_archive(directory, specification)
    else:
        finalize(directory / 'checkpoints', directory / 'archive', identity=identity,
                 expected_inventory=specification['inventory'], verify_inputs=verify_inputs)
    artifact, checksum = _artifact(directory, specification)
    verify_inputs(); observations.append(observe())
    result = dict(stage='sample_and_archive', state='complete',
        attempt_key=attempt_key(specification), specification=specification,
        sampling=sampling, identity_key=artifact['identity_key'],
        manifest_sha256=checksum, resource_observations=observations,
        publication_allowed=False)
    result_path.parent.mkdir(exist_ok=True)
    _write(result_path, result)
    return result


def diagnostic_stage(directory, specification, *, verify_inputs, observe=lambda: None):
    directory = Path(directory).resolve()
    path = directory / 'diagnostics/result.json'
    if path.exists():
        raise FileExistsError('A completed diagnostic stage is verified, never rerun')
    verify_inputs()
    sample = json.loads((directory / 'sample_and_archive/result.json').read_bytes())
    validate_stage(directory, specification, 'sample_and_archive', sample)
    before = observe()
    artifact, checksum = _artifact(directory, specification, manifest_sha256=sample['manifest_sha256'])
    report = diagnose(directory / 'archive', expected_manifest_sha256=checksum,
        expected_identity_key=artifact['identity_key'], expected_inventory=specification['inventory'],
        catalog=specification['catalog'], verify_inputs=verify_inputs, temporary_parent=directory)
    verify_inputs()
    result = dict(stage='diagnostics', state='complete', attempt_key=attempt_key(specification),
        specification=specification, sample_result_sha256=digest(directory / 'sample_and_archive/result.json'),
        report=report, resource_observations=[before, observe()], publication_allowed=False)
    validate_stage(directory, specification, 'diagnostics', result)
    path.parent.mkdir(exist_ok=True)
    _write(path, result)
    return result


def _validate_report(report, fields, settings, divergences):
    """Validate site coverage and reconstruct the EXISTING gate from reported metrics."""
    axes = [settings['chains'], settings['samples']]
    if axes[0] < MIN_CHAINS or axes[1] < MIN_ESS // MIN_CHAINS:
        if report != dict(converged=False, reason='Insufficient chains/draws', shape=axes):
            raise ValueError('Short-run diagnostics cannot establish convergence')
        return
    keys = {'converged', 'failures', 'parameters', 'constants', 'empty_sites',
            'n_draws', 'n_chains', 'divergences'}
    if (set(report) != keys or report['n_draws'] != np.prod(axes)
            or report['n_chains'] != axes[0] or report['divergences'] != divergences):
        raise ValueError('Complete diagnostic inventory and draw accounting required')
    def indexed(rows):
        result = {r['parameter']: r for r in rows}
        if len(result) != len(rows) or not set(result) <= set(fields):
            raise ValueError('Unknown or duplicate diagnostic site')
        return result
    params, constants, empty = (indexed(report[k]) for k in ('parameters', 'constants', 'empty_sites'))
    failures = []
    for name, info in fields.items():
        coords = int(np.prod(info['shape'][2:]))
        if coords == 0:
            if empty.get(name) != dict(parameter=name, shape=info['shape']) or name in params or name in constants:
                raise ValueError('Empty diagnostic site inventory disagrees with collector')
            continue
        if name in empty:
            raise ValueError('Nonempty diagnostic site omitted')
        nconstant = 0
        if name in constants:
            row = constants[name]; nconstant = row['coordinates']
            deterministic = name in DETERMINISTIC_SITES
            if (type(nconstant) is not int or not 0 < nconstant <= coords
                    or row != dict(parameter=name, coordinates=nconstant, deterministic=deterministic)):
                raise ValueError('Invalid constant-coordinate accounting')
            if not deterministic:
                failures.append(f'{name}: constant stochastic coordinate')
        if coords == nconstant:
            if name in params:
                raise ValueError('An all-constant site has no varying diagnostic row')
            continue
        if name not in params:
            raise ValueError('A varying diagnostic site is missing')
        row = params[name]
        metrics = [row[n] for n in ('rhat_max', 'bulk_ess_min', 'tail_ess_min', 'mcse_sd_max')]
        finite = all(type(v) in (int, float) and np.isfinite(v) for v in metrics)
        if not finite and any(v is not None for v in metrics):
            raise ValueError('Nonfinite diagnostics must preserve all missing metrics')
        passed = bool(finite and metrics[0] < RHAT_LIMIT and metrics[1] >= MIN_ESS
                      and metrics[2] >= MIN_ESS and metrics[3] <= MAX_MCSE_SD_RATIO)
        if row['passed'] is not passed or row['constant_coordinates'] != nconstant:
            raise ValueError('Diagnostic threshold decision changed')
        if not passed:
            failures.append(f'{name}: convergence/precision threshold failed')
    if divergences < 0 or divergences > np.prod(axes) * MAX_DIVERGENCE_RATE:
        failures.append('Divergence threshold failed')
    if not params:
        failures.append('No varying posterior coordinates')
    if set(report['failures']) != set(failures) or len(report['failures']) != len(failures):
        raise ValueError('Diagnostic failures do not match the unchanged numerical gate')
    if report['converged'] is not (not failures):
        raise ValueError('Diagnostic convergence decision changed')


def validate_stage(directory, specification, name, result):
    """Require bound numeric artifacts and every parameter/current-field report."""
    directory = Path(directory).resolve()
    if (result.get('stage') != name or result.get('state') != 'complete'
            or result.get('attempt_key') != attempt_key(specification)
            or result.get('specification') != specification or result.get('publication_allowed') is not False):
        raise ValueError('Stage belongs to another attempt or is incomplete')
    if name == 'sample_and_archive':
        artifact, checksum = _artifact(directory, specification, manifest_sha256=result['manifest_sha256'])
        sampled = result['sampling']; settings = specification['settings']
        if (result['identity_key'] != artifact['identity_key']
                or sampled['identity_key'] != artifact['identity_key']
                or sampled['state'] != 'draws_complete'
                or sampled['total_draws'] != settings['samples']
                or sampled['total_warmup'] != settings['warmup']
                or sampled['publication_allowed'] is not False or sampled['convergence_evaluated'] is not False):
            raise ValueError('Incomplete sampling stage')
        return dict(attempt_key=result['attempt_key'], manifest_sha256=checksum,
                    identity_key=artifact['identity_key'])
    if name != 'diagnostics':
        raise ValueError('Unknown attempt stage')
    sample_path = directory / 'sample_and_archive/result.json'
    sample = json.loads(sample_path.read_bytes())
    validate_stage(directory, specification, 'sample_and_archive', sample)
    report = result['report']; diagnostic = report['diagnostics']; settings = specification['settings']
    if (result['sample_result_sha256'] != digest(sample_path)
            or report['state'] != 'diagnostics_complete' or report['identity_key'] != sample['identity_key']
            or report['manifest_sha256'] != sample['manifest_sha256']
            or any(report[k] is not False for k in ('publication_allowed', 'scientific_acceptance_established',
                                                   'attempt_selection_performed'))):
        raise ValueError('Diagnostic report belongs to another artifact')
    additions = {'comparison_reports', 'tree_depth_limit_fraction', 'mean_accept_prob'}
    if settings['chains'] < MIN_CHAINS or settings['samples'] < MIN_ESS // MIN_CHAINS:
        additions.add('divergences')
    core = {k:v for k,v in diagnostic.items() if k not in additions}
    divergences = diagnostic['divergences']
    if type(divergences) is not int or not 0 <= divergences <= settings['chains'] * settings['samples']:
        raise ValueError('Invalid divergence accounting')
    _validate_report(core, specification['inventory']['posterior'], settings, divergences)
    comparisons = diagnostic['comparison_reports']; catalog = specification['catalog']
    expected = {(site, a, b) for site in ('skill', 'overall_race_skill')
                for i,a in enumerate(catalog['field']) for b in catalog['field'][:i]}
    rows = comparisons['current_driver_contrasts']
    if (len(rows) != len(expected)
            or {(r['site'], r['driver_i'], r['driver_j']) for r in rows} != expected
            or comparisons['changes_attempt_convergence_decision'] is not False
            or comparisons['publication_allowed'] is not False):
        raise ValueError('Require every declared current-driver contrast')
    for row in rows:
        _validate_report(row['diagnostic'], {'contrast':dict(shape=[settings['chains'],settings['samples']])},
                         settings, divergences)
        for key in ('chain_means','chain_standard_deviations'):
            values = np.asarray(row[key])
            if values.shape != (settings['chains'],) or not np.isfinite(values).all():
                raise ValueError('Incomplete contrast chain summary')
    if comparisons['all_reported_contrasts_passed'] is not all(r['diagnostic']['converged'] for r in rows):
        raise ValueError('Contrast summary decision changed')
    # Recount the small complete extras here rather than trusting a success flag.
    with np.load(directory / 'archive/extras.npz', allow_pickle=False) as archive:
        if (int(archive['diverging'].sum()) != divergences
                or float(archive['accept_prob'].mean()) != diagnostic['mean_accept_prob']
                or float(np.mean(archive['num_steps'] >= 2**settings['max_tree_depth']-1))
                != diagnostic['tree_depth_limit_fraction']):
            raise ValueError('Recorded diagnostics disagree with retained sampler extras')
    return dict(attempt_key=result['attempt_key'], manifest_sha256=report['manifest_sha256'],
                diagnostics_sha256=hashlib.sha256(canonical(diagnostic)).hexdigest(),
                converged=diagnostic['converged'])


def execute_attempt(directory, specification, request, *, verify_inputs):
    if request['attempt_key'] != attempt_key(specification):
        raise ValueError('Execution request belongs to a different attempt')
    return execute(directory, request, verify_inputs=verify_inputs,
        validate_result=lambda req,name,path,value: validate_stage(directory, specification, name, value))
