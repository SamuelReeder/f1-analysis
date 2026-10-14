"""Terminal-record accounting, including failed and interrupted children."""
import hashlib
import json
from pathlib import Path

import pytest

from analysis import overall_complete_profile_review as review


def object_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fixture(monkeypatch, root, outcomes=('converged', 'converged', 'converged')):
    monkeypatch.setattr(review, 'ROOT', root)
    committed_bytes = {}

    def save(path, value, tracked=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        if tracked:
            committed_bytes[path] = path.read_bytes()

    def committed(path):
        path = Path(path)
        if committed_bytes.get(path) != path.read_bytes():
            raise ValueError('Uncommitted fixture evidence')

    monkeypatch.setattr(review, 'committed', committed)
    reviewer = root / 'analysis/overall_complete_profile_review.py'
    save(reviewer, 'fixture reviewer identity')
    monkeypatch.setattr(review, '__file__', str(reviewer))
    monkeypatch.setattr(review, 'terminal_service', lambda identifier: dict(unit=f'f1-{identifier}.service',
        properties=dict(LoadState='loaded', MainPID='0', ActiveState='active', SubState='exited')))
    cases = ['first', 'middle', 'last']
    identity = 'complete-fixture'
    destination = root / 'outputs/championship/development' / identity
    cache_root = root / 'outputs/championship/cache' / identity
    model = root / 'model.txt'
    save(model, 'fixed model')
    source = dict(inputs={'model.txt': review.sha(model)})
    source['key'] = object_key(source['inputs'])
    preflight_path = root / 'preflight.json'
    save(preflight_path, dict(model_source=source))
    inputs = {'preflight.json': review.sha(preflight_path)}
    preflights = {}
    for case in cases:
        path = root / f'{case}-preflight.json'
        previous = dict(arrays_key=case + '-arrays', parameters_key=case + '-parameters', truth_key=case + '-truth')
        save(path, previous)
        preflights[case] = path.name
        inputs[path.name] = review.sha(path)
    base = dict(chains=2, warmup=2, samples=2, seed=0, target_accept=.9)
    reg = dict(id=identity, status='frozen', profile='complete_design_inference', purpose='numerical_profile',
        publication_allowed=False, studies=cases, source_sha256={}, evidence_sha256=inputs,
        preflight_registration='preflight.json', registered_model_inputs_key=source['key'],
        preflight_results=preflights, synthetic={case: dict(seed=i) for i, case in enumerate(cases)},
        fit_settings=base, retry=[[1, 1, None], [2, 2, None]], resources={'fixture': True})
    registration = root / 'registration.json'
    save(registration, reg)
    completions = []
    for case, outcome in zip(cases, outcomes):
        cache = cache_root / case
        cache.mkdir(parents=True)
        previous = review.read(root / preflights[case])
        catalog = dict(field=['A', 'B'])
        specification = dict(registration_sha256=review.sha(registration), source_key=source['key'],
            study=case, synthetic=True, settings=base, catalog=catalog, arrays_key=previous['arrays_key'])
        receipt = dict(state=outcome, source=source, publication_allowed=False, specification=specification,
            run_key=object_key(specification), catalog=catalog, attempts=[])
        if outcome in ('converged', 'failed_convergence'):
            for index in range(1 if outcome == 'converged' else len(reg['retry'])):
                settings = dict(base, warmup=base['warmup'] * (index + 1), samples=base['samples'] * (index + 1),
                    seed=base['seed'] + index)
                diagnostic = dict(converged=outcome == 'converged', n_chains=settings['chains'],
                    n_draws=settings['chains'] * settings['samples'], comparison_reports=dict(
                        current_driver_contrasts=[dict(site=site, driver_i='B', driver_j='A')
                            for site in ('skill', 'overall_race_skill')],
                        changes_attempt_convergence_decision=False, publication_allowed=False))
                attempt = dict(settings, diagnostics=diagnostic, seconds=1.)
                receipt['attempts'].append(attempt)
                receipt['diagnostics'] = diagnostic
                save(cache / f'attempt_{index + 1}.json', attempt, tracked=False)
                archive = cache / ('posterior.npz' if diagnostic['converged'] else f'failed_{index + 1}.npz')
                # Byte-identity fixture only: this reviewer never verifies arrays.
                archive.write_bytes(b'fixture archive bytes')
                if diagnostic['converged']:
                    receipt.update(posterior_file=archive.name, posterior_sha256=review.sha(archive))
        result = dict(case=case, state='converged' if outcome == 'converged' else 'failed',
            registration_sha256=review.sha(registration), receipt=receipt)
        result_path = destination / case / 'result.json'
        if outcome != 'killed':
            save(cache / 'receipt.json', receipt, tracked=False)
            save(result_path, result)
            save(destination / case / 'truth.json', dict(previous, seed=reg['synthetic'][case]['seed']))
        output, resources = destination / f'{case}.output.txt', destination / f'{case}.resources.txt'
        save(output, dict(state=outcome))
        save(resources, 'external resource log')
        completion = dict(case=case, exit_code=0 if outcome == 'converged' else -9 if outcome == 'killed' else 1,
            state='converged' if outcome == 'converged' else 'failed', result_present=outcome != 'killed',
            result_sha256=review.sha(result_path) if outcome != 'killed' else None,
            log_sha256=review.sha(output), resources_sha256=review.sha(resources))
        save(destination / f'{case}.completion.json', completion)
        completions.append(completion)
    summary_path = destination / 'summary.json'
    save(summary_path, dict(all_processed=True, registration=registration.name,
        registration_sha256=review.sha(registration), publication_allowed=False, studies=completions,
        all_converged=all(row['state'] == 'converged' for row in completions)))
    return registration, destination, cache_root, save, committed_bytes


def test_complete_consistent_records_are_not_array_or_scientific_acceptance(monkeypatch, tmp_path):
    registration, output, cache, save, _ = fixture(monkeypatch, tmp_path)
    result = review.collect(registration)
    assert result['records_consistent'] and result['recorded_all_converged']
    assert [r['case'] for r in result['studies']] == ['first', 'middle', 'last']
    assert all(r['attempts'][0]['archive_sha256'] for r in result['studies'])
    for name in ('archive_arrays_verified', 'diagnostics_recomputed', 'full_recovery_complete',
                 'sbc_complete', 'publication_allowed', 'full_goal_complete'):
        assert result[name] is False


@pytest.mark.parametrize('middle', ['failed_convergence', 'failed_exception', 'killed'])
def test_failed_and_killed_cases_remain_in_the_complete_review(monkeypatch, tmp_path, middle):
    registration, *_ = fixture(monkeypatch, tmp_path, ('converged', middle, 'converged'))
    result = review.collect(registration)
    assert result['records_consistent'] and not result['recorded_all_converged']
    assert [r['case'] for r in result['studies']] == ['first', 'middle', 'last']
    assert result['studies'][1]['completion']['state'] == 'failed'
    assert result['studies'][2]['attempts'][0]['archive_present']
    if middle == 'failed_convergence':
        assert len(result['studies'][1]['attempts']) == 2
        assert all(a['archive_role'] == 'unaccepted_diagnostic_only' for a in result['studies'][1]['attempts'])


@pytest.mark.parametrize('change', ['settings', 'comparisons', 'archive', 'missing_archive', 'attempt_record'])
def test_bad_attempt_is_flagged_without_hiding_other_cases(monkeypatch, tmp_path, change):
    registration, output, cache, save, _ = fixture(monkeypatch, tmp_path)
    if change == 'archive':
        (cache / 'middle/posterior.npz').write_bytes(b'changed archive')
    elif change == 'missing_archive':
        (cache / 'middle/posterior.npz').unlink()
    elif change == 'attempt_record':
        save(cache / 'middle/attempt_1.json', {}, tracked=False)
    else:
        path = cache / 'middle/receipt.json'
        receipt = review.read(path)
        if change == 'settings':
            receipt['attempts'][0]['seed'] += 1
        else:
            receipt['attempts'][0]['diagnostics']['comparison_reports']['current_driver_contrasts'].pop()
            receipt['diagnostics'] = receipt['attempts'][0]['diagnostics']
        save(path, receipt, tracked=False)
        save(cache / 'middle/attempt_1.json', receipt['attempts'][0], tracked=False)
        result_path = output / 'middle/result.json'
        raw = review.read(result_path)
        raw['receipt'] = receipt
        save(result_path, raw)
        summary = review.read(output / 'summary.json')
        summary['studies'][1]['result_sha256'] = review.sha(result_path)
        save(output / 'middle.completion.json', summary['studies'][1])
        save(output / 'summary.json', summary)
    result = review.collect(registration)
    assert not result['records_consistent']
    assert result['studies'][1]['issues']
    assert not result['studies'][0]['issues'] and not result['studies'][2]['issues']
    assert result['studies'][2]['attempts'][0]['archive_present']


@pytest.mark.parametrize('change', ['missing_case', 'reordered_case', 'running_summary', 'uncommitted', 'source'])
def test_incomplete_or_changed_original_evidence_is_rejected(monkeypatch, tmp_path, change):
    registration, output, cache, save, _ = fixture(monkeypatch, tmp_path)
    summary_path = output / 'summary.json'
    summary = review.read(summary_path)
    if change == 'missing_case': summary['studies'].pop()
    elif change == 'reordered_case': summary['studies'].reverse()
    elif change == 'running_summary': summary['all_processed'] = False
    elif change == 'source': save(tmp_path / 'model.txt', 'changed model')
    else: summary['uncommitted_change'] = True
    save(summary_path, summary, tracked=change != 'uncommitted')
    with pytest.raises(ValueError):
        review.collect(registration)


def test_mid_run_resource_changes_are_retained_without_rewriting_registration(monkeypatch, tmp_path):
    registration, output, cache, save, _ = fixture(monkeypatch, tmp_path)
    before = registration.read_bytes()
    record = dict(state='applied', user_request='lower the cpu cores',
        previous_quota_percent=800, new_quota_percent=400)
    save(output.with_name(output.name + '-resource-changes') / 'change.json', record)
    result = review.collect(registration)
    assert result['runtime_spans_changed_resources']
    assert result['resource_changes'][0]['record'] == record
    assert result['registered_resources'] == {'fixture': True}
    assert registration.read_bytes() == before


def test_an_earlier_success_cannot_be_bypassed(monkeypatch, tmp_path):
    registration, output, cache, save, _ = fixture(monkeypatch, tmp_path,
        ('converged', 'failed_convergence', 'converged'))
    receipt = review.read(cache / 'middle/receipt.json')
    receipt['attempts'][0]['diagnostics']['converged'] = True
    save(cache / 'middle/attempt_1.json', receipt['attempts'][0], tracked=False)
    attempts = review.attempt_review(receipt, review.read(registration), cache / 'middle')
    assert any('bypassed' in issue for issue in attempts[0]['issues'])
    assert len(attempts) == len(receipt['attempts'])


def test_a_shortened_failure_ladder_is_rejected(monkeypatch, tmp_path):
    registration, output, cache, save, _ = fixture(monkeypatch, tmp_path,
        ('converged', 'failed_convergence', 'converged'))
    receipt = review.read(cache / 'middle/receipt.json')
    receipt['attempts'].pop()
    receipt['diagnostics'] = receipt['attempts'][0]['diagnostics']
    with pytest.raises(ValueError, match='exhaust'):
        review.attempt_review(receipt, review.read(registration), cache / 'middle')


@pytest.mark.parametrize('state,substate,pid,loaded,allowed', [
    ('active', 'running', '9', 'loaded', False),
    ('active', 'exited', '0', 'loaded', True),
    ('failed', 'failed', '0', 'loaded', True),
    ('inactive', 'dead', '0', 'not-found', False),
    ('inactive', 'dead', '9', 'loaded', False),
])
def test_actual_service_must_be_terminal(monkeypatch, state, substate, pid, loaded, allowed):
    properties = dict(LoadState=loaded, ActiveState=state, SubState=substate, MainPID=pid, Result='success')
    commands = []
    def output(command, **kwargs):
        commands.append(command)
        return '\n'.join(f'{k}={v}' for k, v in properties.items())
    monkeypatch.setattr(review.subprocess, 'check_output', output)
    if allowed:
        assert review.terminal_service('complete-fixture')['properties'] == properties
    else:
        with pytest.raises(ValueError): review.terminal_service('complete-fixture')
    assert commands[0][3] == 'f1-complete-fixture.service'


def interrupted_fixture(monkeypatch, root):
    registration, output, cache, save, tracked = fixture(monkeypatch, root,
        ('converged', 'converged', 'failed_convergence'))
    for path in (output / 'summary.json', output / 'last.completion.json', output / 'last/result.json',
                 cache / 'last/attempt_2.json', cache / 'last/failed_2.npz'):
        path.unlink()
        tracked.pop(path, None)
    receipt = review.read(cache / 'last/receipt.json')
    receipt.update(state='running', attempts=receipt['attempts'][:1])
    receipt['diagnostics'] = receipt['attempts'][-1]['diagnostics']
    save(cache / 'last/receipt.json', receipt, tracked=False)
    reg = review.read(registration)
    settings = dict(reg['fit_settings'], warmup=4, samples=4, seed=1)
    event = dict(study='last', attempt=2, state='sampling', settings=settings)
    save(output / 'last.output.txt', event)
    saved = output.with_name(output.name + '-interruption')
    observed = dict(LoadState='not-found', ActiveState='inactive', SubState='dead', MainPID='0')
    unit = 'f1-complete-fixture.service'
    snapshot = dict(state='interrupted_without_final_record', registration=registration.name,
        registration_sha256=review.sha(registration), publication_allowed=False, recovery_started=False,
        boot_id='fixture-boot', terminal_unit_observation=dict(exit_code=0,
            command=['systemctl', '--user', 'show', unit],
            stdout='\n'.join(f'{k}={v}' for k, v in observed.items())),
        raw_output_inventory={}, copied_cache_metadata={}, cached_archive_inventory={}, cases=[],
        missing_terminal_records=[str((output / 'summary.json').relative_to(root)),
            f"outputs/logs/{reg['id']}/exit.status", str((output / 'last.completion.json').relative_to(root)),
            str((output / 'last/result.json').relative_to(root))], last_recorded_sampling_start=event)
    for path in output.rglob('*'):
        if path.is_file():
            snapshot['raw_output_inventory'][str(path.relative_to(root))] = dict(sha256=review.sha(path), bytes=path.stat().st_size)
    for case in reg['studies']:
        completion = output / f'{case}.completion.json'
        receipt = review.read(cache / case / 'receipt.json')
        row = dict(case=case, completion_present=completion.exists(),
            result_present=(output / case / 'result.json').exists(), receipt_present=True,
            receipt_state=receipt['state'], recorded_attempt_count=len(receipt['attempts']))
        if completion.exists(): row['original_completion'] = review.read(completion)
        snapshot['cases'].append(row)
        for path in (cache / case).glob('*'):
            name = str(path.relative_to(root))
            if path.suffix == '.json':
                destination = saved / 'cache-metadata' / case / path.name
                save(destination, review.read(path))
                snapshot['copied_cache_metadata'][name] = dict(snapshot=str(destination.relative_to(root)), sha256=review.sha(path))
            else:
                snapshot['cached_archive_inventory'][name] = dict(sha256=review.sha(path), bytes=path.stat().st_size)
    path = saved / 'interruption.json'
    save(path, snapshot)
    # Keep the actual interruption guard: only its OS observations are replaced.
    monkeypatch.setattr(review, 'service_state', lambda _: dict(unit=unit, properties=observed))
    real_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw:
        'fixture-boot\n' if str(path) == '/proc/sys/kernel/random/boot_id' else real_read(path, *a, **kw))
    return registration, path, output, cache, save, observed


def test_interrupted_run_keeps_all_cases_without_inventing_a_completion(monkeypatch, tmp_path):
    registration, interruption, output, cache, _, _ = interrupted_fixture(monkeypatch, tmp_path)
    before = {str(p): p.read_bytes() for base in (output, cache) for p in base.rglob('*') if p.is_file()}
    result = review.collect(registration, interruption=interruption)
    assert result['records_consistent'] and result['all_required_cases_accounted']
    assert not result['original_run_complete'] and not result['recorded_all_converged']
    assert [r['case'] for r in result['studies']] == ['first', 'middle', 'last']
    interrupted = result['studies'][-1]
    assert interrupted['completion'] is None and not interrupted['completion_record_present']
    assert interrupted['receipt']['state'] == 'running'
    assert interrupted['attempts'][0]['archive_role'] == 'unaccepted_diagnostic_only'
    assert result['last_recorded_sampling_start']['attempt'] == 2
    assert result['interruption_record'] == str(interruption.relative_to(tmp_path))
    assert not result['archive_arrays_verified'] and not result['publication_allowed']
    assert before == {str(p): p.read_bytes() for base in (output, cache) for p in base.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['raw', 'raw_extra', 'metadata', 'snapshot', 'archive', 'cache_extra',
    'missing_inventory_case', 'missing_terminal', 'event_settings', 'event_log', 'uncommitted', 'summary'])
def test_interruption_requires_all_original_bytes_and_observations(monkeypatch, tmp_path, change):
    registration, interruption, output, cache, save, _ = interrupted_fixture(monkeypatch, tmp_path)
    record = review.read(interruption)
    if change == 'raw': save(output / 'last/truth.json', {})
    elif change == 'raw_extra': save(output / 'extra.json', {})
    elif change == 'metadata': save(cache / 'last/receipt.json', {}, tracked=False)
    elif change == 'snapshot': save(interruption.parent / 'cache-metadata/last/receipt.json', {})
    elif change == 'archive': (cache / 'last/failed_1.npz').write_bytes(b'changed bytes')
    elif change == 'cache_extra': (cache / 'last/extra.json').write_text('{}')
    elif change == 'missing_inventory_case': record['cases'].pop()
    elif change == 'missing_terminal': record['missing_terminal_records'].pop()
    elif change == 'event_settings': record['last_recorded_sampling_start']['settings']['seed'] += 1
    elif change == 'event_log': record['last_recorded_sampling_start']['attempt'] = 1
    elif change == 'summary': save(output / 'summary.json', {'invented': True})
    else: record['changed'] = True
    save(interruption, record, tracked=change != 'uncommitted')
    with pytest.raises(ValueError):
        review.collect(registration, interruption=interruption)


@pytest.mark.parametrize('change', ['live_service', 'different_boot', 'different_original_unit', 'original_service_live'])
def test_interruption_cannot_bypass_live_service_or_changed_boot(monkeypatch, tmp_path, change):
    registration, interruption, _, _, save, observed = interrupted_fixture(monkeypatch, tmp_path)
    record = review.read(interruption)
    if change == 'live_service': observed.update(LoadState='loaded', MainPID='100', ActiveState='active', SubState='running')
    elif change == 'different_boot': record['boot_id'] = 'another-boot'
    elif change == 'different_original_unit': record['terminal_unit_observation']['command'][3] = 'another.service'
    else: record['terminal_unit_observation']['stdout'] = 'LoadState=loaded\nMainPID=100\nActiveState=active\nSubState=running\n'
    save(interruption, record)
    with pytest.raises(ValueError):
        review.collect(registration, interruption=interruption)
