"""Prospective deterministic-GPU numerical candidate; no sampling or acceptance."""
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import time
import traceback

from analysis import backend_precision as precision

base = precision.base
ROOT = base.ROOT
PARENT = 'docs/overall_backend_precision_protocol.json'
RESULTS = 'outputs/championship/development/overall-backend-precision-v1'
SOURCES = ('analysis/backend_determinism/__init__.py', 'analysis/backend_determinism/__main__.py',
           'docs/overall_backend_determinism.md')
ARMS = ('gpu64_a', 'gpu64_b')


def definition(parent):
    environments = {k: dict(v) for k, v in parent['environments'].items()}
    for arm in ARMS:
        environments[arm]['XLA_FLAGS'] += ' --xla_gpu_exclude_nondeterministic_ops=true'
    return dict(id='overall-backend-determinism-v1',
        purpose='prospective_deterministic_gpu_fixed_input_check', studies=parent['studies'],
        arms=list(ARMS), environments=environments, unset_environment=parent['unset_environment'],
        matmul_precision=None, resources=parent['resources'], gpu_admission=parent['gpu_admission'],
        repetition_requirement='complete_bitwise_equality',
        comparison_requirement='original_unchanged_budgets_against_committed_cpu64_a',
        sampling_allowed=False, publication_allowed=False, scientific_acceptance_allowed=False,
        changes_original_screen=False)


def evidence(parent):
    return [PARENT, *[f'{RESULTS}/{operation}/cpu32/{case}/result.json'
                     for case in parent['studies'] for operation in ('compare',)],
            *[f'{RESULTS}/evaluate/cpu64_a/{case}/result.json' for case in parent['studies']]]


def freeze(destination):
    parent, _ = precision.checked(ROOT/PARENT)
    for case in parent['studies']:
        r = base.read(ROOT/f'{RESULTS}/compare/cpu32/{case}/result.json')
        if (r['state'] != 'complete' or not r['higher_precision_within_original_budgets']
                or r['protocol_sha256'] != base.sha(ROOT/PARENT)):
            raise ValueError('Require complete preceding precision evidence')
    p = dict(definition(parent), registered_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={n:base.committed(ROOT,n) for n in SOURCES},
        evidence_sha256={n:base.committed(ROOT,n) for n in evidence(parent)})
    with Path(destination).open('x') as f:
        json.dump(p, f, indent=2); f.write('\n')


def checked(path):
    path = Path(path).resolve(); base.committed(ROOT, str(path.relative_to(ROOT)))
    p = base.read(path); parent, original = precision.checked(ROOT/PARENT)
    if (any(p.get(k) != v for k,v in definition(parent).items())
            or set(p['source_sha256']) != set(SOURCES)
            or set(p['evidence_sha256']) != set(evidence(parent))):
        raise ValueError('Changed deterministic candidate definition')
    for name, expected in {**p['source_sha256'], **p['evidence_sha256']}.items():
        if base.sha(ROOT/name) != expected:
            raise ValueError(f'Changed deterministic candidate input: {name}')
    return p, original


def compare(p, original, study, evaluations, record):
    labels = base.point_labels(study)
    path = ROOT/f'{RESULTS}/evaluate/cpu64_a/{study["case"]}/result.json'
    receipt, cpu = precision.load_evaluation(path, study['case'], labels, base.sha(ROOT/PARENT), 'cpu64_a')
    identities = {'cpu64_a_reference':dict(path=str(path.relative_to(ROOT)), sha256=base.sha(path))}
    arrays = {}
    for arm in ARMS:
        path = Path(evaluations)/arm/study['case']/'result.json'
        r, arrays[arm] = precision.load_evaluation(path, study['case'], labels, record['protocol_sha256'], arm)
        if r['prepared_sha256'] != receipt['prepared_sha256']:
            raise ValueError('Different fixed inputs')
        identities[arm] = dict(path=str(path.relative_to(ROOT)), sha256=base.sha(path))
    repeated = precision.differences(arrays['gpu64_a'], arrays['gpu64_b'])
    budgets = {arm:base.compare_fields(cpu, arrays[arm], dict(state=original['state_tolerance'],
        value_and_gradient=original['value_and_gradient_tolerance'])) for arm in ARMS}
    exact = all(v['bitwise_equal'] for v in repeated.values())
    agreement = all(v['passed'] for rows in budgets.values() for v in rows.values())
    record.update(point_labels=labels, evaluations=identities, repetitions=repeated, cpu_comparisons=budgets,
        complete_bitwise_repetition=exact, complete_cpu_agreement=agreement,
        numerical_candidate_passed=exact and agreement, full_model_checkpoint_replay_established=False)


def run(args):
    p, original = checked(args.protocol)
    study = next(s for s in original['studies'] if s['case'] == args.case)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    record = dict(operation=args.operation, case=args.case, arm=args.arm, state='running',
        protocol_sha256=base.sha(args.protocol), started_at=datetime.now(timezone.utc).isoformat(),
        sampling_executed=False, publication_allowed=False, scientific_acceptance_established=False)
    started = time.monotonic(); base.save(output/'result.json', record)
    try:
        record['actual_scope'] = precision.scope(p, original,
            args.arm if args.operation == 'evaluate' else 'cpu32', args.scope)
        if args.operation == 'evaluate':
            precision.evaluate64(original, study,
                ROOT/f'{precision.RESULTS}/prepare/cpu/{args.case}/result.json',
                base.cache_path(args.cache), record)
        else:
            compare(p, original, study, args.evaluations, record)
        checked(args.protocol); record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-started,
            process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_at=datetime.now(timezone.utc).isoformat())
        base.save(output/'result.json', record)
