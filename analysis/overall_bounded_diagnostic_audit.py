"""All saved attempts plus the complete largest-retry diagnostic resource fixture."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.overall_bounded_diagnostics import check, COORDINATE_BLOCK
from analysis.overall_posterior_archive import _mapped_numeric_archive
from f1rank.overall.provenance import snapshot

SEED = 20261026
FILL_COORDINATES = 128
PROFILE = 'outputs/championship/development/overall-weekend-coordinate-profile-v1'
STORAGE = 'outputs/championship/development/overall-fit-storage-v1'
REPORT_KEYS = ('converged', 'failures', 'parameters', 'constants', 'empty_sites', 'n_draws', 'n_chains', 'divergences')
SOURCES = ('analysis/overall_bounded_diagnostic_audit.py', 'analysis/overall_bounded_diagnostics.py',
           'analysis/overall_posterior_archive.py', 'docs/overall_bounded_diagnostics.md')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


LOADED = {name: sha(ROOT / name) for name in SOURCES}


def require_sources():
    for name, fingerprint in LOADED.items():
        if (sha(ROOT / name) != fingerprint
                or subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != (ROOT / name).read_bytes()):
            raise ValueError('Commit and preserve the exact loaded diagnostic audit sources')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(output, scope):
    output = Path(output)
    if output.exists(): raise FileExistsError('Preserve the existing diagnostic audit')
    require_sources()
    properties = subprocess.check_output(['systemctl', '--user', 'show', scope,
        '-p', 'Slice', '-p', 'MemoryMax', '-p', 'CPUQuotaPerSecUSec'], text=True)
    actual = dict(line.split('=', 1) for line in properties.splitlines())
    if (actual != dict(Slice='f1.slice', MemoryMax=str(3 * 1024**3), CPUQuotaPerSecUSec='2s')
            or os.getpriority(os.PRIO_PROCESS, 0) != 19):
        raise ValueError('Use the fixed diagnostic audit scope')
    source = snapshot()
    summary = json.loads((ROOT / PROFILE / 'summary.json').read_text())
    if source != summary['source']: raise ValueError('Original model source changed')
    review = json.loads((ROOT / PROFILE / 'review.json').read_text())
    storage_summary = json.loads((ROOT / STORAGE / 'summary.json').read_text())
    if not storage_summary['all_processed'] or not storage_summary['all_complete']:
        raise ValueError('The full storage inventory must be complete')
    inputs = {name: sha(ROOT / name) for name in [f'{PROFILE}/summary.json', f'{PROFILE}/review.json',
        f'{STORAGE}/summary.json', *[f'{PROFILE}/{s["study"]}.json' for s in review['studies']],
        *[f'{STORAGE}/{s["case"]}/result.json' for s in storage_summary['studies']]]}
    for name in inputs:
        if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Commit complete source evidence before this audit')
    output.mkdir(parents=True)
    started = time.monotonic()
    record = dict(state='running', stage='saved_attempts', started_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources=LOADED, inputs=inputs, model_inputs_key=source['key'], actual_scope=dict(actual, Nice=19),
        coordinate_block=COORDINATE_BLOCK, attempts=[], mcmc_run=False, scientific_evidence_verified=False,
        full_fit_peak_memory_verified=False, publication_allowed=False)
    cache = ROOT / 'outputs/championship/cache'
    try:
        for study in review['studies']:
            original = json.loads((ROOT / PROFILE / f'{study["study"]}.json').read_text())['receipt']
            sites = json.loads((ROOT / STORAGE / study['study'] / 'result.json').read_text())['sites']
            for attempt in study['attempts']:
                archive = ROOT / attempt['archive']
                if sha(archive) != attempt['archive_sha256']: raise ValueError('Saved attempt bytes changed')
                axes = (attempt['settings']['chains'], attempt['settings']['samples'])
                before = time.monotonic()
                with _mapped_numeric_archive(archive, names=sorted(sites), axes=axes, temporary_parent=cache) as (post, _):
                    observed = check(post, attempt['divergences'])
                expected = original['attempts'][attempt['attempt']-1]['diagnostics']
                expected = {k: expected[k] for k in REPORT_KEYS}
                if sha(archive) != attempt['archive_sha256']: raise ValueError('Archive changed during audit')
                row = dict(study=study['study'], attempt=attempt['attempt'], archive_sha256=attempt['archive_sha256'],
                    original_role=attempt['archive_role'], complete_original_report_equal=observed == expected,
                    original_converged=expected['converged'], diagnostic=observed, seconds=time.monotonic()-before)
                record['attempts'].append(row)
                save(output / 'audit.json', record)
        record['all_saved_reports_identical'] = all(a['complete_original_report_equal'] for a in record['attempts'])
        if not record['all_saved_reports_identical']:
            raise ValueError('Bounded implementation changed an original diagnostic report')
        print('Every saved attempt diagnostic reproduced exactly.', flush=True)
        record['stage'] = 'full_inventory_resource_fixture'
        full = json.loads((ROOT / STORAGE / 'real_network/result.json').read_text())
        sites = full['sites']
        largest = max(full['accounting']['attempts'], key=lambda a: a['posterior_payload_bytes'])
        axes = (largest['chains'], largest['samples_per_chain'])
        required_bytes = largest['posterior_payload_bytes']
        free = shutil.disk_usage(cache).free
        record['resource'] = dict(case='real_network', axes=list(axes), seed=SEED, fill_coordinates=FILL_COORDINATES,
            complete_site_inventory=sites, logical_payload_bytes=required_bytes, disk_free_before_bytes=free,
            compatible_retry_attempts=[a['attempt'] for a in full['accounting']['attempts']
                                      if (a['chains'], a['samples_per_chain']) == axes],
            fixture='Independent standard-normal values in every recorded floating array; not a model posterior.')
        save(output / 'audit.json', record)
        if free < required_bytes + 1024**2: raise ValueError('Insufficient disk for the complete fixture')
        if any(np.dtype(s['dtype']).kind != 'f' for s in sites.values()):
            raise ValueError('The frozen synthetic formula requires the recorded floating-site inventory')
        before = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='overall-diagnostic-resource-', dir=cache) as directory:
            temporary = Path(directory)
            record['resource']['temporary_directory'] = str(temporary)
            save(output / 'audit.json', record)
            files = {}
            for i, (name, spec) in enumerate(sorted(sites.items())):
                target = temporary / f'{name}.npy'
                a = np.lib.format.open_memmap(target, mode='w+', dtype=spec['dtype'], shape=axes+tuple(spec['shape']))
                try:
                    flat = a.reshape(*axes, -1)
                    random = np.random.default_rng(np.random.SeedSequence([SEED, i]))
                    for start in range(0, spec['coordinates'], FILL_COORDINATES):
                        stop = min(start+FILL_COORDINATES, spec['coordinates'])
                        flat[..., start:stop] = random.standard_normal(axes+(stop-start,)).astype(a.dtype)
                    a.flush()
                finally:
                    a._mmap.close()
                files[name] = dict(sha256=sha(target), file_bytes=target.stat().st_size)
            record['resource'].update(construction_seconds=time.monotonic()-before, files=files)
            save(output / 'audit.json', record)
            post = {name: np.load(temporary / f'{name}.npy', mmap_mode='r', allow_pickle=False) for name in sites}
            try:
                if sum(v.nbytes for v in post.values()) != required_bytes:
                    raise ValueError('Fixture dropped a recorded field or dimension')
                before = time.monotonic()
                diagnostic = check(post, 0)
                record['resource'].update(diagnostic_seconds=time.monotonic()-before, diagnostic=diagnostic)
            finally:
                for value in post.values(): value._mmap.close()
        record['resource']['temporary_arrays_removed'] = not temporary.exists()
        require_sources()
        if snapshot() != source or any(sha(ROOT/name) != fingerprint for name, fingerprint in inputs.items()):
            raise ValueError('Original model, inputs or audit source changed')
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
            process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(output / 'audit.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope-name', required=True)
    args = parser.parse_args()
    result = run(args.output, args.scope_name)
    print(json.dumps(dict(state=result['state'], all_saved_reports_identical=result['all_saved_reports_identical'],
                          process_max_rss_kib=result['process_max_rss_kib'])))
