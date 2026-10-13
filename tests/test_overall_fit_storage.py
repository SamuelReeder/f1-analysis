"""Exact array-payload arithmetic and independently written archive inventories."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from analysis import overall_fit_storage as storage


def values():
    return dict(scalar=np.asarray(.5, dtype='float32'),
                matrix=np.arange(12, dtype='float64').reshape(3, 4),
                integer=np.arange(2, dtype='int16'), empty=np.empty((0, 5), dtype='float32'))


def test_all_dtype_and_empty_site_bytes_match_independent_allocations():
    original=values();sites=storage.inventory(original)
    settings=dict(chains=4, samples=3, warmup=2, target_accept=.9)
    retry=[[1,1,None],[2,2,None],[3,4,None],[3,4,.98]]
    actual=storage.accounting(sites, settings, retry)
    assert set(sites)==set(original)
    total=0
    for row,(_,multiplier,_) in zip(actual['attempts'],retry,strict=True):
        arrays={k:np.empty((settings['chains'],settings['samples']*multiplier)+v.shape,dtype=v.dtype)
                for k,v in original.items()}
        assert row['posterior_payload_bytes']==sum(v.nbytes for v in arrays.values())
        assert row['largest_single_site_float64_bytes']==max(np.asarray(v,dtype=float).nbytes for v in arrays.values())
        total+=sum(v.nbytes for v in arrays.values())
    assert actual['whole_retry_ladder_payload_bytes']==total
    assert actual['largest_sites']==['matrix']
    assert not actual['physical_peak_memory_measured'] and not actual['compressed_storage_estimated']


@pytest.mark.parametrize('bad', ['coordinates','bytes_per_draw','itemsize','shape'])
def test_arithmetic_rejects_inconsistent_metadata(bad):
    sites=storage.inventory(values())
    if bad=='shape': sites['matrix']['shape']=[-3,4]
    else: sites['matrix'][bad]+=1
    with pytest.raises(ValueError):
        storage.accounting(sites,dict(chains=4,samples=3,warmup=2,target_accept=.9),[[1,1,None]])


@pytest.mark.parametrize('change',['unchanged','reordered','missing','extra','shape','dtype','axes','duplicate'])
def test_complete_archive_header_comparison(tmp_path,change):
    original=values();sites=storage.inventory(original);axes=(4,3)
    arrays={k:np.broadcast_to(v,axes+v.shape).copy() for k,v in original.items()}
    if change=='reordered': arrays=dict(reversed(list(arrays.items())))
    elif change=='missing': del arrays['scalar']
    elif change=='extra': arrays['unregistered']=np.ones(axes)
    elif change=='shape': arrays['matrix']=arrays['matrix'].reshape(*axes,12)
    elif change=='dtype': arrays['scalar']=arrays['scalar'].astype('float64')
    elif change=='axes': arrays['scalar']=arrays['scalar'][:2]
    path=tmp_path/'independent.npz';np.savez_compressed(path,**arrays)
    if change=='duplicate':
        with zipfile.ZipFile(path,'a') as archive:
            raw=archive.read('scalar.npy')
            with pytest.warns(UserWarning,match='Duplicate'):
                archive.writestr('scalar.npy',raw)
    if change in ('unchanged','reordered'):
        result=storage.compare_archive(path,sites,axes)
        assert result['complete_headers_match'] and result['payload_bytes']==sum(a.nbytes for a in arrays.values())
        assert result['npy_members_bytes']>result['payload_bytes']
        assert result['archive_file_bytes']==path.stat().st_size
    else:
        with pytest.raises(ValueError):storage.compare_archive(path,sites,axes)


@pytest.mark.parametrize('bad',[{},dict(x=np.array([np.inf])),dict(x=np.array([object()])),
                               dict(x=np.array([(1.,)],dtype=[('field','f8')]))])
def test_non_numeric_or_nonfinite_sites_are_not_silently_counted(bad):
    with pytest.raises(ValueError):storage.inventory(bad)


def test_parent_preserves_failures_and_completes_fixed_inventory(monkeypatch,tmp_path):
    protocol=dict(id='storage-fixture',studies=[dict(case='first',seed=0),dict(case='second',seed=1)])
    path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol))
    (tmp_path/'outputs/championship/development').mkdir(parents=True)
    monkeypatch.setattr(storage,'ROOT',tmp_path)
    monkeypatch.setattr(storage,'checked_protocol',lambda p:protocol)
    commands=[]
    def run(command,**kwargs):
        commands.append(command)
        folder=Path(command[command.index('--output')+1]);folder.mkdir()
        if folder.name=='first':return SimpleNamespace(returncode=137)
        (folder/'result.json').write_text(json.dumps(dict(state='complete')))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(storage.subprocess,'run',run)
    result=storage.execute(path)
    assert result['all_processed'] and not result['all_complete']
    assert [r['state'] for r in result['studies']]==['failed','complete']
    assert not result['mcmc_run'] and not result['full_fit_peak_memory_verified'] and not result['publication_allowed']
    assert all('MemoryMax=3G' in c and 'CPUQuota=200%' in c and '--slice=f1.slice' in c for c in commands)
    monkeypatch.setattr(storage.subprocess,'run',lambda *a,**k:pytest.fail('Existing inventory relaunched'))
    with pytest.raises(FileExistsError):storage.execute(path)


@pytest.mark.parametrize('change',['none','missing','axes','role','extra'])
def test_bound_review_requires_every_original_attempt(monkeypatch,tmp_path,change):
    (tmp_path/'analysis').mkdir()
    source=tmp_path/'analysis/overall_fit_storage.py'
    source.write_bytes(Path(storage.__file__).read_bytes())
    studies=[dict(case='connected',seed=42)]
    preflight=tmp_path/'preflight.json';preflight.write_text(json.dumps(dict(studies=studies)))
    fit=tmp_path/'fit.json';fit.write_text('{}')
    original=[dict(attempt=i,archive=f'failed_{i}.npz',archive_sha256=str(i)*64,
                   settings=dict(chains=4,samples=100*i),archive_role='failed_diagnostic_only') for i in (1,2)]
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(studies=[dict(study='connected',attempts=original)])))
    comparisons=[dict(case='connected',attempt=a['attempt'],path=a['archive'],sha256=a['archive_sha256'],
                      axes=[4,a['settings']['samples']],role=a['archive_role']) for a in original]
    if change=='missing':comparisons.pop()
    elif change=='axes':comparisons[-1]['axes']=[4,100]
    elif change=='role':comparisons[0]['role']='accepted'
    elif change=='extra':comparisons.append(copy.deepcopy(comparisons[-1]))
    p=dict(id='fixture',purpose='complete_retained_fit_storage_inventory',status='frozen',mcmc_allowed=False,
        publication_allowed=False,resources=dict(slice='f1.slice',MemoryMax='3G',CPUQuota='200%',Nice=19),
        source_sha256={'analysis/overall_fit_storage.py':storage.LOADED_SHA},
        input_sha256={f.name:storage.sha(f) for f in (preflight,fit,review)},runtime_source_sha256={},
        preflight_protocol=preflight.name,fit_registration=fit.name,archive_review=review.name,
        studies=studies,archive_comparisons=comparisons)
    path=tmp_path/'protocol.json';path.write_text(json.dumps(p))
    monkeypatch.setattr(storage,'ROOT',tmp_path)
    monkeypatch.setattr(storage.subprocess,'check_output',lambda cmd,**kw:(tmp_path/cmd[-1].removeprefix('HEAD:')).read_bytes())
    if change=='none':assert storage.checked_protocol(path)==p
    else:
        with pytest.raises(ValueError,match='every saved attempt'):storage.checked_protocol(path)
