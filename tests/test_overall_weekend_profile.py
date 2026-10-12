"""The separate weekend experiment cannot run without its fixed registration."""
import json

import pytest


def test_pending_weekend_proposal_cannot_run():
    from analysis import overall_weekend_profile as runner
    with pytest.raises(ValueError, match='frozen'):
        runner.checked_registration(runner.ROOT / 'docs/overall_weekend_profile_proposal.json')


@pytest.mark.parametrize('change', ['approval','seed','settings','gate','inputs','preceding','truth'])
def test_changed_weekend_experiment_is_rejected(monkeypatch,change):
    from analysis import overall_weekend_profile as runner
    path=runner.ROOT / 'docs/overall_weekend_profile_proposal.json'
    r=json.loads(path.read_text());r.update(status='frozen',user_approved=True)
    if change=='approval':r['user_approved']=False
    elif change=='seed':r['synthetic']['confounded']['seed']+=1
    elif change=='settings':r['fit_settings']['max_tree_depth']+=1
    elif change=='gate':r['diagnostics']['rank_rhat_strict_upper']+=.1
    elif change=='inputs':r['registered_model_inputs_key']='changed'
    elif change=='preceding':r['preceding_experiment_registration_sha256']='changed'
    else:r['original_truth_files']['confounded']['sha256']='changed'
    monkeypatch.setattr(runner,'committed_registration',lambda _:r)
    with pytest.raises(ValueError):runner.checked_registration(path)


def test_prepared_proposal_would_pass_only_the_approval_boundary(monkeypatch):
    from analysis import overall_weekend_profile as runner
    path=runner.ROOT / 'docs/overall_weekend_profile_proposal.json'
    r=json.loads(path.read_text());r.update(status='frozen',user_approved=True)
    # This is a read-only preflight, not execute/sample; actual launch also
    # requires the real user-approved registration to be committed.
    monkeypatch.setattr(runner,'committed_registration',lambda _:r)
    assert runner.checked_registration(path)==r


def test_an_existing_weekend_experiment_is_not_overwritten(tmp_path,monkeypatch):
    from analysis import overall_weekend_profile as runner
    destination=tmp_path / 'outputs/championship/development/example'
    destination.mkdir(parents=True);marker=destination / 'summary.json'
    marker.write_text('original result')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'checked_registration',lambda _:dict(id='example'))
    monkeypatch.setattr(runner,'snapshot',lambda: {})
    with pytest.raises(FileExistsError,match='Preserve'):runner.execute(tmp_path / 'registration.json')
    assert marker.read_text()=='original result'


def test_failed_attempt_reports_all_pairs_without_changing_the_decision(monkeypatch):
    import numpy as np
    from analysis import overall_weekend_profile as runner
    rng=np.random.default_rng(21)
    post={key:rng.normal(size=(4,500,6)) for key in ('skill','overall_race_skill')}
    catalog=dict(field=[f'synthetic_{i}' for i in range(6)],field_entries=list(range(6)))
    diagnostic=dict(converged=False,divergences=0,reason='preserved original failure')
    monkeypatch.setattr(runner,'sample_rotated',lambda *args:(post,diagnostic))
    monkeypatch.setattr(runner,'require_prototype_sources',lambda _:None)
    returned,reported=runner.sample_with_reports({}, {}, {}, None,catalog)
    assert returned is post and not reported['converged'] and reported['reason']==diagnostic['reason']
    assert len(reported['comparison_reports']['current_driver_contrasts'])==30
    assert reported['comparison_reports']['changes_attempt_convergence_decision'] is False
