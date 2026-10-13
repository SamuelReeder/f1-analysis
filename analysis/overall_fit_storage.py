"""Inventory actual retained sample/deterministic arrays without a new fit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


LOADED_SHA = sha(__file__)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inventory(values):
    import numpy as np
    if not values:
        raise ValueError('Retain the complete nonempty postprocessing inventory')
    result = {}
    for name, value in values.items():
        a = np.asarray(value)
        if a.dtype.kind not in 'biufc' or a.dtype.fields is not None or not np.isfinite(a).all():
            raise ValueError('Only complete finite plain numeric postprocessed sites are supported')
        result[name] = dict(shape=list(a.shape), dtype=a.dtype.str, coordinates=int(a.size),
                            itemsize=a.dtype.itemsize, bytes_per_draw=int(a.nbytes))
    return result


def accounting(sites, settings, retry):
    import numpy as np
    if not sites or type(settings['chains']) is not int or settings['chains'] < 1:
        raise ValueError('Complete sites and positive chain count required')
    for site in sites.values():
        shape = site['shape']
        if (any(type(d) is not int or d < 0 for d in shape)
                or math.prod(shape) != site['coordinates']
                or np.dtype(site['dtype']).itemsize != site['itemsize']
                or site['bytes_per_draw'] != site['coordinates'] * site['itemsize']):
            raise ValueError('Inconsistent site shape, dtype or byte inventory')
    per_draw = sum(s['bytes_per_draw'] for s in sites.values())
    largest = max(s['coordinates'] for s in sites.values())
    attempts = []
    for i, (warmup, draws, target) in enumerate(retry, 1):
        samples = settings['samples'] * draws
        if type(samples) is not int or samples < 1:
            raise ValueError('Positive fixed draw count required')
        total = settings['chains'] * samples
        attempts.append(dict(attempt=i, chains=settings['chains'], samples_per_chain=samples,
            warmup_per_chain=settings['warmup'] * warmup,
            target_accept=settings['target_accept'] if target is None else target,
            posterior_payload_bytes=per_draw * total,
            largest_single_site_float64_bytes=largest * total * np.dtype(float).itemsize))
    return dict(site_count=len(sites), coordinates_per_draw=sum(s['coordinates'] for s in sites.values()),
        posterior_bytes_per_draw=per_draw, attempts=attempts,
        maximum_attempt_payload_bytes=max(a['posterior_payload_bytes'] for a in attempts),
        whole_retry_ladder_payload_bytes=sum(a['posterior_payload_bytes'] for a in attempts),
        largest_sites=[name for name, s in sites.items() if s['coordinates'] == largest],
        physical_peak_memory_measured=False, compressed_storage_estimated=False)


def compare_archive(path, sites, axes):
    """Headers of a separately fingerprint-bound archive, never fit acceptance."""
    import numpy as np
    from analysis.overall_posterior_archive import _header
    payload = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename[:-4] for m in members]
        if (any(not m.filename.endswith('.npy') for m in members)
                or len(set(names)) != len(names) or set(names) != set(sites)):
            raise ValueError('Saved archive does not contain the exact complete site inventory')
        for name, member in zip(names, members):
            with archive.open(member) as handle:
                shape, _, dtype, size = _header(handle, member, axes)
            expected = sites[name]
            if shape[2:] != tuple(expected['shape']) or dtype != np.dtype(expected['dtype']):
                raise ValueError('Saved archive per-draw shape or dtype differs')
            payload += size
        npy_bytes = sum(m.file_size for m in members)
    expected_bytes = sum(s['bytes_per_draw'] for s in sites.values()) * math.prod(axes)
    if payload != expected_bytes:
        raise ValueError('Stored payload disagrees with complete byte arithmetic')
    return dict(axes=list(axes), site_count=len(sites), payload_bytes=payload,
                npy_members_bytes=npy_bytes, archive_file_bytes=Path(path).stat().st_size,
                complete_headers_match=True)


def checked_protocol(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    if subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(ROOT)}'], cwd=ROOT) != content:
        raise ValueError('Commit the storage protocol before execution')
    p = json.loads(content)
    if (p['purpose'] != 'complete_retained_fit_storage_inventory' or p['status'] != 'frozen'
            or p['mcmc_allowed'] is not False or p['publication_allowed'] is not False
            or not re.fullmatch('[a-z][a-z0-9-]*', p['id'])
            or p['resources'] != dict(slice='f1.slice', MemoryMax='3G', CPUQuota='200%', Nice=19)
            or p['source_sha256'].get('analysis/overall_fit_storage.py') != LOADED_SHA):
        raise ValueError('Wrong or changed computation-only storage protocol')
    for name, expected in p['source_sha256'].items():
        target = ROOT / name
        if (sha(target) != expected
                or subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != target.read_bytes()):
            raise ValueError(f'Inventory source changed: {name}')
    for group in ('input_sha256', 'runtime_source_sha256'):
        for name, expected in p[group].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f'Inventory input/runtime changed: {name}')
    if p['preflight_protocol'] not in p['input_sha256'] or p['fit_registration'] not in p['input_sha256']:
        raise ValueError('Bind both original frozen registrations')
    if p['studies'] != json.loads((ROOT / p['preflight_protocol']).read_text())['studies']:
        raise ValueError('Preserve the complete preflight inventory before starting')
    if p['archive_review'] not in p['input_sha256']:
        raise ValueError('Bind the complete previous archive review')
    review = json.loads((ROOT / p['archive_review']).read_text())
    expected = [dict(case=s['study'], attempt=a['attempt'], path=a['archive'], sha256=a['archive_sha256'],
                     axes=[a['settings']['chains'], a['settings']['samples']], role=a['archive_role'])
                for s in review['studies'] for a in s['attempts']]
    if p['archive_comparisons'] != expected:
        raise ValueError('Keep every saved attempt in the comparison inventory')
    return p


def child(protocol_path, case, output, scope):
    p = checked_protocol(protocol_path)
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    record = dict(case=case, state='running', stage='scope', started_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        protocol_sha256=sha(protocol_path), mcmc_run=False, publication_allowed=False)
    def phase(name):
        record['stage'] = name
        save(output / 'progress.json', record)
    try:
        properties = subprocess.check_output(['systemctl', '--user', 'show', scope,
            '-p', 'Slice', '-p', 'MemoryMax', '-p', 'CPUQuotaPerSecUSec'], text=True)
        actual = dict(line.split('=', 1) for line in properties.splitlines())
        if (actual != dict(Slice='f1.slice', MemoryMax=str(3 * 1024**3), CPUQuotaPerSecUSec='2s')
                or os.getpriority(os.PRIO_PROCESS, 0) != 19):
            raise ValueError('Use the fixed capped child scope')
        record['actual_scope'] = dict(actual, Nice=19)
        phase('imports_and_input_checks')
        import jax
        import jax.numpy as jnp
        import numpy as np
        import numpyro
        from functools import partial
        from types import SimpleNamespace
        from numpyro.infer.mcmc import _collect_and_postprocess
        from numpyro.infer.util import constrain_fn, unconstrain_fn
        from analysis.overall_recovery_preflight import checked_protocol as checked_preflight
        from analysis.overall_recovery_checks import generate_with_parameters
        from analysis.overall_control_rotation import ROTATED_SITE, build_rotation
        from analysis.overall_weekend_coordinates import HYPER_SITES, TOTAL, COMPLEMENT, transform, weekend_model
        from f1rank.overall.fitting import committed_registration
        preflight = checked_preflight(ROOT / p['preflight_protocol'])
        profile = committed_registration(ROOT / p['fit_registration'])
        if p['studies'] != preflight['studies']:
            raise ValueError('Preserve every original preflight case and seed')
        for name in p['runtime_source_sha256']:
            if not Path(name).resolve().is_relative_to(Path(numpyro.__file__).resolve().parent):
                raise ValueError('Loaded NumPyro package differs from the recorded source')
        spec = next(row for row in p['studies'] if row['case'] == case)
        previous = json.loads((ROOT / p['preflight_output'] / case / 'result.json').read_text())
        record.update(generation_seed=spec['seed'], runtime=dict(python=sys.version, numpy=np.__version__,
            jax=jax.__version__, numpyro=numpyro.__version__, backend=jax.default_backend(), x64=jax.config.x64_enabled))
        phase('regenerate_matched_inputs')
        generated = generate_with_parameters(case, spec['seed'])
        for key in ('arrays_key', 'parameters_key', 'truth_key'):
            if generated[key] != previous[key]:
                raise ValueError('Previously inspected inputs changed')
            record[key] = generated[key]
        blocks, params = generated['blocks'], generated['parameters']
        rotation = build_rotation(blocks['laps'])
        if rotation.fingerprint() != previous['control_rotation_sha256']:
            raise ValueError('Original coordinate map changed')
        updated = dict(params)
        for original, new in HYPER_SITES.items(): updated[new] = updated.pop(original)
        updated[TOTAL], updated[COMPLEMENT] = transform(params['sd_driver_form'], params['overall_sd_day'],
            params['overall_shared_weekend'], updated.pop('driver_form_z'), updated.pop('overall_day_z'))
        updated[ROTATED_SITE] = rotation.rotated({name: updated.pop(name) for name in rotation.shapes})
        updated = {k: jnp.asarray(v) for k, v in updated.items()}
        args = tuple({k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in blocks[name].items()}
                     for name in ('qualifying', 'laps', 'outcomes'))
        wrapped = weekend_model(rotation)
        phase('actual_collector_postprocessing')
        z = unconstrain_fn(wrapped, args, {}, updated)
        postprocess = partial(constrain_fn, wrapped, args, {}, return_deterministic=True)
        direct = postprocess(z)
        collected = _collect_and_postprocess(postprocess, ('z',), ())((SimpleNamespace(z=z),))
        if set(direct) != set(collected):
            raise ValueError('Collector and complete postprocessor inventories disagree')
        for name in direct: np.testing.assert_array_equal(direct[name], collected[name])
        sites = inventory(collected)
        record.update(sites=sites, accounting=accounting(sites, profile['fit_settings'], profile['retry']),
                      complete_collector_matches_postprocessing=True)
        phase('saved_archive_inventory_comparisons')
        comparisons = []
        for reference in p['archive_comparisons']:
            if reference['case'] != case: continue
            archive = ROOT / reference['path']
            if sha(archive) != reference['sha256']: raise ValueError('Original saved attempt changed')
            comparison = compare_archive(archive, sites, tuple(reference['axes']))
            if sha(archive) != reference['sha256']: raise ValueError('Saved attempt changed during inspection')
            comparisons.append(dict(reference, inventory=comparison))
        record['archive_comparisons'] = comparisons
        phase('source_recheck')
        if checked_protocol(protocol_path) != p: raise ValueError('Storage protocol changed')
        checked_preflight(ROOT / p['preflight_protocol'])
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
                      process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(output / 'result.json', record)


def execute(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    p = checked_protocol(protocol_path)
    output = ROOT / 'outputs/championship/development' / p['id']
    output.mkdir(exist_ok=False)
    rows = []
    for spec in p['studies']:
        checked_protocol(protocol_path)
        case = spec['case']
        scope = f"f1-{p['id']}-{case.replace('_', '-')}.scope"
        command = ['systemd-run', '--user', '--scope', '--unit', scope, '--slice=f1.slice', '-q',
            '-p', 'MemoryMax=3G', '-p', 'CPUQuota=200%', '--', '/usr/bin/time', '-v', '-o', str(output / f'{case}.resources.txt'),
            '/usr/bin/nice', '-n', '19', sys.executable, str(Path(__file__).resolve()), '--protocol', str(protocol_path),
            '--case', case, '--output', str(output / case), '--scope-name', scope]
        with (output / f'{case}.output.txt').open('x') as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        target = output / case / 'result.json'
        result = json.loads(target.read_text()) if target.exists() else None
        row = dict(case=case, process_exit_code=process.returncode,
            state='complete' if process.returncode == 0 and result and result['state'] == 'complete' else 'failed',
            result_sha256=sha(target) if target.exists() else None, command=command)
        save(output / f'{case}.completion.json', row)
        rows.append(row)
        print(json.dumps(dict(case=case, state=row['state'], exit_code=process.returncode)), flush=True)
    checked_protocol(protocol_path)
    result = dict(protocol=str(protocol_path.relative_to(ROOT)), protocol_sha256=sha(protocol_path), studies=rows,
        all_processed=True, all_complete=all(r['state'] == 'complete' for r in rows),
        mcmc_run=False, full_fit_peak_memory_verified=False, publication_allowed=False)
    save(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scope-name')
    args = parser.parse_args()
    if args.case:
        if args.output is None or args.scope_name is None: parser.error('A child requires its output and scope')
        child(args.protocol, args.case, args.output, args.scope_name)
    else:
        raise SystemExit(0 if execute(args.protocol)['all_complete'] else 1)
