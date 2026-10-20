"""Match the integrated production model to every frozen research coordinate."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('stable_audit', ROOT/'analysis/overall_studentt_stable_audit.py')
stable = importlib.util.module_from_spec(spec); spec.loader.exec_module(stable)
previous, sha, save = stable.previous, stable.sha, stable.save
BASE_PROTOCOL = 'docs/overall_studentt_stable_protocol.json'
RESULTS = {precision:f'outputs/championship/development/overall-studentt-stable-v1/network{suffix}/result.json'
           for precision, suffix in (('float32', '32'), ('float64', '64'))}
SOURCES = ('analysis/overall_studentt_production_audit.py', 'tests/test_overall_studentt_production.py')


def freeze(path):
    from f1rank.overall.fitting import require_clean_inputs
    from f1rank.overall.inference import METHOD
    from f1rank.overall.provenance import snapshot
    stable.checked(ROOT/BASE_PROTOCOL)
    inputs = snapshot(); require_clean_inputs(inputs)
    result = dict(id='overall-studentt-production-audit-v1', status='frozen',
        purpose='complete_fixed_coordinate_production_migration_check',
        registered_at=datetime.now(timezone.utc).isoformat(), inference_method=METHOD,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        model_snapshot=inputs, precisions=list(previous.PRECISIONS),
        source_sha256={name:previous.committed(ROOT, name) for name in SOURCES},
        input_sha256={name:previous.committed(ROOT, name) for name in (BASE_PROTOCOL, *RESULTS.values())},
        criterion='identical_potential_and_every_gradient_coordinate_at_all_original_points_in_each_precision',
        sampling_allowed=False, publication_allowed=False)
    with Path(path).open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')


def checked(path):
    from f1rank.overall.inference import METHOD
    from f1rank.overall.provenance import require_snapshot
    path = Path(path).resolve(); previous.committed(ROOT, path.relative_to(ROOT).as_posix())
    result = json.loads(path.read_bytes())
    if (result['inference_method'] != METHOD or result['sampling_allowed'] is not False
            or result['publication_allowed'] is not False or result['precisions'] != list(previous.PRECISIONS)):
        raise ValueError('Changed registered production audit')
    require_snapshot(result['model_snapshot'])
    if set(result['source_sha256']) != set(SOURCES) or set(result['input_sha256']) != {BASE_PROTOCOL, *RESULTS.values()}:
        raise ValueError('Incomplete audit source/input binding')
    for name, value in {**result['source_sha256'], **result['input_sha256']}.items():
        if previous.committed(ROOT, name) != value:
            raise ValueError('Changed committed audit source/input')
    stable.checked(ROOT/BASE_PROTOCOL)
    return result


def evaluate(args):
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import potential_energy
    from f1rank.overall.batched_rotation import prepare
    from f1rank.overall.control_rotation import build_rotation
    from f1rank.overall.fitting import arrays_digest
    from f1rank.overall.weekend_coordinates import weekend_model

    protocol = checked(args.protocol)
    actual = previous.scope_properties(args.scope, args.precision)
    if jax.config.x64_enabled != (args.precision == 'float64'):
        raise ValueError('Wrong arithmetic precision')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    cache = Path(args.cache).resolve(); cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raise FileExistsError('Preserve existing cache')
    begun = time.monotonic()
    record = dict(state='running', precision=args.precision, actual_scope=actual,
        started_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(args.protocol),
        model_inputs_key=protocol['model_snapshot']['key'], inference_method=protocol['inference_method'],
        mcmc_executed=False, publication_allowed=False, points=[])
    try:
        baseline = json.loads((ROOT/RESULTS[args.precision]).read_bytes())
        prepared = json.loads((ROOT/stable.audit.PREPARED).read_bytes())
        if baseline['state'] != 'complete' or sha(baseline['cache']) != baseline['cache_sha256']:
            raise ValueError('Incomplete or changed research baseline')
        if sha(prepared['cache']) != prepared['cache_sha256']:
            raise ValueError('Changed original coordinates')
        with np.load(prepared['cache'], allow_pickle=False) as saved:
            arrays = {name:saved[name] for name in saved.files}
        blocks = previous.unpack_blocks(arrays, prepared['block_scalars'])
        coordinates = {name:arrays[f'z/{name}'] for name in prepared['parameter_shapes']}
        if (arrays_digest(blocks) != prepared['arrays_key']
                or arrays_digest({'coordinates':coordinates}) != prepared['point_coordinates_sha256']):
            raise ValueError('Changed model inputs or coordinates')
        rotation = prepare(build_rotation(blocks['laps']))
        if rotation.fingerprint() != prepared['control_rotation_sha256']:
            raise ValueError('Production control map differs')
        model = weekend_model(rotation)
        arguments = tuple({k:jnp.asarray(previous.lift(v, args.precision)) if isinstance(v, np.ndarray) else v
                           for k, v in blocks[name].items()} for name in ('qualifying', 'laps', 'outcomes'))
        value_gradient = jax.jit(jax.value_and_grad(lambda z:potential_energy(model, arguments, {}, z)))
        with np.load(baseline['cache'], allow_pickle=False) as saved:
            expected = {name:saved[name] for name in saved.files}
        keys = {'potential'} | {f'gradient/{name}' for name in coordinates}
        if set(expected) != keys or baseline['point_labels'] != prepared['point_labels']:
            raise ValueError('Incomplete research comparison inventory')
        payload = {name:[] for name in keys}
        for index, label in enumerate(prepared['point_labels']):
            z = {name:jnp.asarray(previous.lift(values[index], args.precision)) for name, values in coordinates.items()}
            value, gradient = value_gradient(z)
            actual_values = dict(potential=np.asarray(value), **{f'gradient/{k}':np.asarray(v) for k, v in gradient.items()})
            if set(actual_values) != keys or any(v.dtype != np.dtype(args.precision) for v in actual_values.values()):
                raise ValueError('Incomplete or mixed-precision production calculation')
            fields = {name:previous.difference(values, expected[name][index]) for name, values in actual_values.items()}
            identical = all(np.array_equal(values, expected[name][index]) for name, values in actual_values.items())
            record['points'].append(dict(label=label, identical=identical, fields=fields))
            for name, value in actual_values.items():
                payload[name].append(value)
            save(output/'progress.json', record)
        with cache.open('xb') as f:
            np.savez_compressed(f, **{name:np.stack(values) for name, values in payload.items()})
        record.update(cache=str(cache), cache_sha256=sha(cache), all_points_identical=all(r['identical'] for r in record['points']),
                      baseline_result_sha256=sha(ROOT/RESULTS[args.precision]))
        if sha(baseline['cache']) != baseline['cache_sha256'] or sha(prepared['cache']) != prepared['cache_sha256']:
            raise ValueError('Changed cache during comparison')
        checked(args.protocol)
        if not record['all_points_identical']:
            raise ValueError('Production migration differs from the verified arithmetic')
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed_exception', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-begun, process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_at=datetime.now(timezone.utc).isoformat())
        save(output/'result.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'evaluate'])
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--precision', choices=previous.PRECISIONS, default='float32')
    for name in ('scope', 'output', 'cache'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.protocol)
    else:
        result = evaluate(args)
        print(json.dumps({k:result[k] for k in ('state', 'seconds', 'process_max_rss_kib', 'all_points_identical')}))
