"""Frozen independent reference and matched-model audit of equivalent arithmetic."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'analysis'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load('overall_studentt_numerics')
candidate = load('overall_studentt_stable')
previous = audit.previous
sha, save = audit.sha, audit.save
BASE_PROTOCOL = 'docs/overall_studentt_numerics_v2_protocol.json'
STUDY_ID = 'overall-studentt-stable-v1'
SOURCES = ('analysis/overall_studentt_stable.py', 'analysis/overall_studentt_stable_audit.py',
           'tests/test_overall_studentt_stable.py', 'docs/overall_studentt_stable.md')
BASELINE_INPUTS = tuple(f'outputs/championship/development/overall-studentt-numerics-v2/primitive{p}/result.json'
                        for p in ('32', '64'))
METHODS = ('original', 'log_beta', 'corrected_normalizer', 'stable')


def definition():
    return dict(id=STUDY_ID, methods=list(METHODS), precisions=list(previous.PRECISIONS),
        error_eps_multipliers=audit.ERROR_EPS_MULTIPLIERS, reference_digits=list(audit.REFERENCE_DIGITS),
        gamma_ratio_min_b=candidate.GAMMA_RATIO_MIN_B,
        stirling_coefficients=[list(pair) for pair in candidate.STIRLING_COEFFICIENTS],
        tail_series_max_q=candidate.TAIL_SERIES_MAX_Q,
        tail_series_last_power=candidate.TAIL_SERIES_LAST_POWER,
        boundary_policy='append_signed_float32_residual_and_adjacent_values_at_tail_series_boundary_for_each_original_parameter_point',
        sampling_allowed=False, scientific_acceptance_established=False, publication_allowed=False)


def freeze(path):
    audit.checked(ROOT/BASE_PROTOCOL)
    result = dict(definition(), registered_at=datetime.now(timezone.utc).isoformat(), status='frozen',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={name:previous.committed(ROOT, name) for name in SOURCES},
        input_sha256={name:previous.committed(ROOT, name) for name in (BASE_PROTOCOL, *BASELINE_INPUTS)})
    with Path(path).open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')


def checked(path):
    path = Path(path).resolve()
    previous.committed(ROOT, path.relative_to(ROOT).as_posix())
    result = json.loads(path.read_bytes())
    if any(result.get(k) != value for k, value in definition().items()):
        raise ValueError('Changed arithmetic definition, workload or error budget')
    if set(result['source_sha256']) != set(SOURCES) or set(result['input_sha256']) != {BASE_PROTOCOL, *BASELINE_INPUTS}:
        raise ValueError('Incomplete source or evidence inventory')
    for name, value in {**result['source_sha256'], **result['input_sha256']}.items():
        if previous.committed(ROOT, name) != value:
            raise ValueError('Changed committed source or evidence')
    audit.checked(ROOT/BASE_PROTOCOL)
    return result


def input_rows(original):
    rows = list(original)
    q = candidate.TAIL_SERIES_MAX_Q
    for source in original:
        if source['residual'] != 0:
            continue
        boundary = np.float32(np.sqrt(np.exp(source['log_df'])*q/(1-q)))
        for sign in (-1, 1):
            for direction in (-np.inf, 0., np.inf):
                value = boundary if direction == 0 else np.nextafter(boundary, np.float32(direction))
                rows.append(dict(kind='tail_boundary', source=source, direction=str(direction),
                                 log_df=source['log_df'], residual=float(sign*value)))
    return rows


def primitive(precision, record, checkpoint):
    import jax
    import jax.numpy as jnp
    from numpyro.distributions import StudentT
    original_path = ROOT/BASELINE_INPUTS[0 if precision == 'float32' else 1]
    original = json.loads(original_path.read_bytes())
    rows = input_rows(original['rows'])
    mp = audit.load_reference()
    references, unstable = [], []
    for index, row in enumerate(rows):
        low, high = [audit.reference_row(mp, row['log_df'], row['residual'], digits)
                     for digits in audit.REFERENCE_DIGITS]
        references.append(high)
        if low != high:
            unstable.append(index)
    if references[:len(original['rows'])] != original['reference']:
        raise ValueError('Independent reference differs on original inventory')
    record.update(rows=rows, reference=references, reference_unstable_rows=unstable,
        original_row_count=len(original['rows']), columns=original['columns'], methods={})
    coordinates = jnp.asarray([[r['log_df'], 0., 0.] for r in rows], dtype=precision)
    observations = jnp.asarray([r['residual'] for r in rows], dtype=precision)
    methods = dict(original=lambda df, loc, scale, value:StudentT(df, loc, scale).log_prob(value),
        log_beta=audit.beta_log_prob, corrected_normalizer=candidate.normalizer_only_log_prob,
        stable=candidate.stable_log_prob)
    for method, calculation in methods.items():
        def f(p, x):
            return calculation(jnp.exp(p[0]), p[1], jnp.exp(p[2]), x)
        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(f), in_axes=(0, 0)))(coordinates, observations)
        actual = np.column_stack([np.asarray(values), np.asarray(gradients)])
        if actual.dtype != np.dtype(precision):
            raise ValueError('Wrong actual output precision')
        encoded = [[float(v) if np.isfinite(v) else str(v) for v in row] for row in actual]
        result = dict(values=encoded, **audit.assessment(actual, references, precision))
        if method in original['methods']:
            result['prior_values_identical'] = encoded[:len(original['rows'])] == original['methods'][method]['values']
        record['methods'][method] = result
        checkpoint()
    if unstable:
        raise ValueError('Independent reference did not stabilize')
    if any(not r.get('prior_values_identical', True) for r in record['methods'].values()):
        raise ValueError('Original comparison calculation changed')
    record['candidate_passes_primitive_criteria'] = record['methods']['stable']['all_within_fixed_error_budget']


def execute(args):
    checked(args.protocol)
    actual = previous.scope_properties(args.scope, args.precision)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    record = dict(state='running', action=args.action, precision=args.precision, actual_scope=actual,
        started_at=datetime.now(timezone.utc).isoformat(), stable_protocol_sha256=sha(args.protocol),
        protocol_sha256=sha(ROOT/audit.ORIGINAL_PROTOCOL),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        mcmc_executed=False, scientific_acceptance_established=False, publication_allowed=False)
    def checkpoint():
        save(output/'progress.json', record)
    checkpoint()
    try:
        if args.action == 'primitive':
            primitive(args.precision, record, checkpoint)
        elif args.action == 'network':
            cache = Path(args.cache).resolve(); cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists():
                raise FileExistsError('Preserve earlier cache')
            record['calculation'] = STUDY_ID
            with candidate.stable_studentt_calculation():
                previous.evaluate(previous.checked(ROOT/audit.ORIGINAL_PROTOCOL), ROOT/audit.PREPARED,
                                  args.precision, cache, record, checkpoint)
        else:
            for path in (args.left, args.right):
                if json.loads(Path(path).read_bytes()).get('stable_protocol_sha256') != sha(args.protocol):
                    raise ValueError('Network result belongs to a different arithmetic study')
            previous.compare(args.left, args.right, record)
        checked(args.protocol)
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed_exception', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-start, finished_at=datetime.now(timezone.utc).isoformat(),
                      process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(output/'result.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'primitive', 'network', 'compare'])
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--precision', choices=previous.PRECISIONS, default='float32')
    for name in ('scope', 'output', 'cache', 'left', 'right'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.protocol)
    else:
        r = execute(args)
        print(json.dumps({key:r[key] for key in ('state', 'action', 'seconds', 'process_max_rss_kib')}))
