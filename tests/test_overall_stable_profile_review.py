"""Metadata fixtures only: no fitting, posterior loading or numerical libraries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import overall_stable_profile as profile
from analysis import overall_stable_profile_review as review


class CorrectedProfileReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tracked = {}
        for module in (review, review.original, profile):
            change = patch.object(module, 'ROOT', self.root)
            change.start(); self.addCleanup(change.stop)
        for module in (review, review.original):
            path = self.root / 'analysis' / Path(module.__file__).name
            self.save(path, 'reviewer fixture')
            change = patch.object(module, '__file__', str(path))
            change.start(); self.addCleanup(change.stop)
        for name in (review.DEFINITION, 'tests/test_overall_stable_profile_review.py'):
            self.save(self.root / name, 'fixture source')

        def committed(path):
            if self.tracked.get(Path(path)) != Path(path).read_bytes():
                raise ValueError('Uncommitted fixture evidence')
        change = patch.object(review.original, 'committed', committed)
        change.start(); self.addCleanup(change.stop)
        change = patch.object(review.original, 'terminal_service', return_value={'terminal': True})
        self.terminal = change.start(); self.addCleanup(change.stop)

    def save(self, path, value, *, tracked=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        if tracked:
            self.tracked[path] = path.read_bytes()

    def fixture(self, outcomes=('converged', 'converged', 'converged')):
        cases = ['first', 'middle', 'last']
        self.output = self.root / 'outputs/championship/development/fixture'
        self.cache = self.root / 'outputs/championship/cache/fixture'
        source = dict(inputs={}, key=review.object_key({}))
        base = dict(chains=2, warmup=2, samples=3, seed=0, target_accept=.9, max_tree_depth=10)
        self.reg = dict(id='fixture', schema_version=2, registered_model_inputs_key=source['key'],
            inference_method='fixture_method', studies=cases, fit_settings=base,
            retry=[[1, 1, None], [2, 2, .98]],
            synthetic={case:dict(case=case, seed=index) for index, case in enumerate(cases)})
        self.registration = self.root / 'registration.json'
        self.save(self.registration, self.reg)
        self.save(self.root / profile.PREFLIGHT_PROTOCOL, dict(model_snapshot=source))
        previous, completions = {}, []
        for case, outcome in zip(cases, outcomes):
            old = dict(arrays_key=case+'-arrays', parameters_key=case+'-parameters', truth_key=case+'-truth',
                role='prior_predictive', control_rotation_sha256='fixture_rotation',
                sites={name:dict(shape=[2], dtype='<f4') for name in ('skill', 'overall_race_skill')})
            previous[case] = old
            preflight_path = self.root / profile.PREFLIGHT_OUTPUT / case / 'result.json'
            self.save(preflight_path, old)
            catalog = dict(field=['A', 'B'], inference_method=self.reg['inference_method'],
                control_rotation_sha256=old['control_rotation_sha256'], preflight_sha256=review.sha(preflight_path))
            spec = dict(source_key=source['key'], arrays_key=old['arrays_key'], catalog=catalog,
                registration_sha256=review.sha(self.registration), study=case, synthetic=True, settings=base)
            receipt = dict(schema_version=2, source=source, catalog=catalog, settings=base,
                specification=spec, run_key=review.object_key(spec), publication_allowed=False,
                state='running' if outcome == 'killed' else outcome, attempts=[])
            run = self.cache / case
            if outcome in ('converged', 'failed_convergence'):
                count = 1 if outcome == 'converged' else len(self.reg['retry'])
                for index in range(count):
                    warm, draws, accept = self.reg['retry'][index]
                    settings = dict(base, warmup=base['warmup']*warm, samples=base['samples']*draws,
                        seed=base['seed']+index, target_accept=base['target_accept'] if accept is None else accept)
                    diagnostic = dict(converged=outcome == 'converged', n_chains=settings['chains'],
                        n_draws=settings['chains']*settings['samples'], inference_method=self.reg['inference_method'],
                        control_rotation_sha256=old['control_rotation_sha256'], comparison_reports=dict(
                            current_driver_contrasts=[dict(site=site, driver_i='B', driver_j='A',
                                diagnostic=dict(converged=True)) for site in ('skill', 'overall_race_skill')],
                            all_reported_contrasts_passed=True, changes_attempt_convergence_decision=False,
                            publication_allowed=False))
                    sampler = run / f'sampler_{index+1}.npz'
                    self.save(sampler, 'invented sampler bytes; no arrays', tracked=False)
                    attempt = dict(settings, diagnostics=diagnostic, seconds=1., sampler_file=sampler.name,
                        sampler_sha256=review.sha(sampler), posterior_sites={name:dict(
                            shape=[settings['chains'], settings['samples'], *site['shape']], dtype=site['dtype'])
                            for name, site in old['sites'].items()})
                    receipt['attempts'].append(attempt); receipt['diagnostics'] = diagnostic
                    self.save(run / f'attempt_{index+1}.json', attempt, tracked=False)
                    archive = run / ('posterior.npz' if diagnostic['converged'] else f'failed_{index+1}.npz')
                    self.save(archive, 'invented posterior bytes; no arrays', tracked=False)
                    if diagnostic['converged']:
                        receipt.update(posterior_file=archive.name, posterior_sha256=review.sha(archive))
            self.save(run / 'receipt.json', receipt, tracked=False)
            identity = review.generating_identity(self.reg, case, old)
            self.save(self.output / case / 'truth.json', dict(identity, truth={}))
            result_path = self.output / case / 'result.json'
            if outcome != 'killed':
                self.save(result_path, dict(case=case, state='converged' if outcome == 'converged' else 'failed',
                    model_inputs_key=source['key'], registration_sha256=review.sha(self.registration),
                    receipt=receipt, original_inputs=identity, posterior_inventory_verified=outcome == 'converged',
                    **{k:False for k in review.FALSE_FLAGS}))
            log, resources = self.output / f'{case}.output.txt', self.output / f'{case}.resources.txt'
            self.save(log, 'fixture log'); self.save(resources, 'separate external resource fixture')
            completion = dict(case=case, exit_code=0 if outcome == 'converged' else -9 if outcome == 'killed' else 1,
                state='converged' if outcome == 'converged' else 'failed', result_present=outcome != 'killed',
                result_sha256=review.sha(result_path) if outcome != 'killed' else None,
                log_sha256=review.sha(log), resources_sha256=review.sha(resources))
            self.save(self.output / f'{case}.completion.json', completion); completions.append(completion)
        self.save(self.output / 'summary.json', dict(registration=self.registration.name,
            registration_sha256=review.sha(self.registration), studies=completions, all_processed=True,
            all_converged=all(row['state'] == 'converged' for row in completions),
            **{k:False for k in review.FALSE_FLAGS}))
        change = patch.object(profile, 'checked', return_value=self.reg)
        self.checked = change.start(); self.addCleanup(change.stop)
        change = patch.object(profile, 'load_preflight', return_value=previous)
        change.start(); self.addCleanup(change.stop)
        return previous

    def rewrite_receipt(self, case, receipt):
        self.save(self.cache / case / 'receipt.json', receipt, tracked=False)
        for index, attempt in enumerate(receipt['attempts'], 1):
            self.save(self.cache / case / f'attempt_{index}.json', attempt, tracked=False)
        path = self.output / case / 'result.json'
        result = review.read(path); result['receipt'] = receipt; self.save(path, result)
        summary = review.read(self.output / 'summary.json')
        row = next(row for row in summary['studies'] if row['case'] == case)
        row['result_sha256'] = review.sha(path)
        self.save(self.output / f'{case}.completion.json', row)
        self.save(self.output / 'summary.json', summary)

    def test_complete_metadata_review_is_not_scientific_or_array_acceptance(self):
        self.fixture()
        report = review.collect(self.registration)
        self.assertTrue(report['records_consistent'])
        self.assertTrue(report['recorded_all_converged'])
        for key in (*review.FALSE_FLAGS, 'archive_arrays_verified', 'diagnostics_recomputed',
                    'sampler_extras_recounted', 'scientific_acceptance_established', 'full_goal_complete'):
            self.assertIs(report[key], False)

    def test_failed_and_killed_cases_do_not_hide_later_cases(self):
        self.fixture(('failed_convergence', 'killed', 'converged'))
        report = review.collect(self.registration)
        self.assertTrue(report['records_consistent']); self.assertFalse(report['recorded_all_converged'])
        self.assertEqual([r['case'] for r in report['studies']], self.reg['studies'])
        self.assertEqual(len(report['studies'][0]['attempts']), len(self.reg['retry']))
        self.assertEqual(report['studies'][1]['attempts'], [])
        self.assertTrue(report['studies'][2]['attempts'][0]['archive_present'])

    def test_bad_sampler_bytes_are_retained_as_an_issue(self):
        self.fixture()
        (self.cache / 'middle/sampler_1.npz').write_bytes(b'changed')
        report = review.collect(self.registration)
        self.assertFalse(report['records_consistent'])
        self.assertTrue(report['studies'][1]['issues']); self.assertFalse(report['studies'][2]['issues'])

    def test_missing_sampler_archive_is_an_issue(self):
        self.fixture()
        (self.cache / 'middle/sampler_1.npz').unlink()
        report = review.collect(self.registration)
        self.assertFalse(report['records_consistent'])

    def test_earlier_failed_attempt_cannot_omit_a_field(self):
        self.fixture(('failed_convergence', 'converged', 'converged'))
        receipt = review.read(self.cache / 'first/receipt.json')
        del receipt['attempts'][0]['posterior_sites']['skill']
        self.rewrite_receipt('first', receipt)
        report = review.collect(self.registration)
        self.assertFalse(report['records_consistent']); self.assertTrue(report['studies'][0]['attempts'][0]['issues'])

    def test_changed_source_and_retry_are_rejected(self):
        self.fixture()
        receipt = review.read(self.cache / 'first/receipt.json')
        receipt['source'] = copy.deepcopy(receipt['source']); receipt['source']['key'] = 'changed'
        self.rewrite_receipt('first', receipt)
        receipt = review.read(self.cache / 'middle/receipt.json'); receipt['attempts'][0]['seed'] += 1
        self.rewrite_receipt('middle', receipt)
        report = review.collect(self.registration)
        self.assertFalse(report['records_consistent'])
        self.assertTrue(report['studies'][0]['issues']); self.assertTrue(report['studies'][1]['issues'])

    def test_unreferenced_cache_prevents_a_complete_success(self):
        self.fixture()
        self.save(self.cache / 'middle/failed_99.npz', 'unexpected', tracked=False)
        report = review.collect(self.registration)
        self.assertFalse(report['records_consistent'])
        self.assertEqual(report['studies'][1]['unreferenced_cache_files'], ['failed_99.npz'])

    def test_changed_raw_record_requires_preserved_evidence(self):
        self.fixture()
        (self.output / 'middle.output.txt').write_text('uncommitted change')
        with self.assertRaisesRegex(ValueError, 'Uncommitted'):
            review.collect(self.registration)

    def test_live_service_rejected_before_numerical_registration_imports(self):
        self.fixture()
        self.terminal.side_effect = ValueError('still running')
        with self.assertRaisesRegex(ValueError, 'still running'):
            review.collect(self.registration)
        self.checked.assert_not_called()

    def test_terminal_state_rechecked_on_successful_exit(self):
        self.fixture()
        self.terminal.side_effect = [{'terminal': True}, {'terminal': False}]
        with self.assertRaisesRegex(ValueError, 'Terminal service changed'):
            review.collect(self.registration)

    def test_contradictory_child_inventory_flag_does_not_count_as_success(self):
        self.assertFalse(review.completion_agrees(dict(exit_code=0, state='failed'),
            dict(state='converged', posterior_inventory_verified=False)))
        with self.assertRaisesRegex(ValueError, 'completion decision'):
            review.completion_agrees(dict(exit_code=0, state='converged'),
                dict(state='converged', posterior_inventory_verified=False))


if __name__ == '__main__':
    unittest.main()
