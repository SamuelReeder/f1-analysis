"""Known-probability and original-model checks; no real-event score or fitting."""
import copy
import itertools

import numpy as np
import pytest

from analysis import overall_legacy_prequalifying as pre
from analysis.overall_forecast_catalog import build_qualifying_catalog
from analysis.overall_observation_kernels import qualifying_segments
from f1rank.overall.fitting import arrays_digest
from test_overall_forecast_states import future_path
from test_overall_legacy_forecast import setup, full_fixture


def qualifying_case():
    source,design,snap,p = setup()
    for name,value in dict(sigma0=.4,sigma_tau=.3,nu=5.,sd_car_segment=.2).items():
        p[name] = np.full(p["sd_level"].shape,value)
    return source,design,snap,p


def generate(design,path,p,batch=3,advancing=(4,2)):
    return pre.forecast_legacy_qualifying(design,path,p,cutoff=str(design.events.event_id.max()),
        fit_metadata=dict(start_season=2010,model_kw={}),seed=41,
        training_arrays_key=build_qualifying_catalog(design,path[0],cutoff=str(design.events.event_id.max()))["training_arrays_key"],
        advancing={s["event"]["event_id"]:advancing for s in path},grid_policy=pre.GRID_POLICY,draw_batch_size=batch)


def bind_grids(forecast):
    forecast["grid_draws_key"] = arrays_digest({e["event_id"]:dict(positions=e["grid_draws"]) for e in forecast["events"]})
    forecast["future_features_key"] = arrays_digest({e["event_id"]:e["race_features"] for e in forecast["events"]})
    return forecast


def mixture_fixture(n=2):
    grids = np.array([[*range(1,n+1)]]*4+[[*range(n,0,-1)]]*2).reshape(2,3,n)
    field = [f"d{i}" for i in range(n)]
    event = dict(event_id="2026-01",drivers=field,grid_draws=grids,
        race_features=dict(driver=np.linspace(-.3,.4,n),car=np.linspace(.7,-.2,n)),
        catalog=dict(information_set="prequalifying",origin_at="2026-03-01T00:00:00Z",snapshot_sha256="a"*64))
    forecast = bind_grids(dict(purpose="development_legacy_prequalifying",prequalifying_policy=pre.POLICY,
        grid_policy=pre.GRID_POLICY,feature_policy="median_of_paired_persistent_states_without_future_transients",
        posterior_shape=[2,3],events=[event],consumed_posterior_key="synthetic-qualifying",training_arrays_key="b"*64,
        qualifying_observation_key="synthetic-observations"))
    beta = dict(c_grid=np.array([[1.,1.],[0.,0.]]),a_car=np.array([[.2,-.3],[.7,1.1]]),
                b_driver=np.array([[1.2,-.1],[.3,.8]]))
    panels = {event["event_id"]:{d:i+1 for i,d in enumerate(field)}}
    return forecast,beta,panels


def integrate(forecast,beta,panels,**kwargs):
    return pre.integrate_legacy_prequalifying(forecast,beta,panels,**kwargs)


def direct_probability(forecast,beta,panels,arm="legacy_grid_qualifying"):
    event = forecast["events"][0]; n=len(event["drivers"])
    rank = np.array([panels[event["event_id"]][d] for d in event["drivers"]])
    order = np.argsort(np.where(rank>0,rank,np.inf))[:(rank>0).sum()]
    results=[]
    for grid in event["grid_draws"].reshape(-1,n):
        for index in range(beta["c_grid"].size):
            strength = -beta["c_grid"].flat[index]*np.log(grid)
            if arm=="legacy_grid_qualifying":
                strength = strength+beta["a_car"].flat[index]*event["race_features"]["car"]+beta["b_driver"].flat[index]*event["race_features"]["driver"]
            weights=np.exp(strength[order]); probability=1.
            for i in range(len(weights)): probability*=weights[i]/weights[i:].sum()
            results.append(probability)
    return np.mean(results)


def test_hand_computable_grid_mixture_uses_all_unequal_draw_counts():
    forecast,beta,panels=mixture_fixture()
    result=integrate(forecast,beta,panels,arm="grid_only")
    np.testing.assert_allclose(np.exp(result["log_probabilities"]),[19/36],rtol=1e-14)
    assert result["qualifying_posterior_shape"]==[2,3]
    assert result["coefficient_posterior_shape"]==[2,2]
    assert result["unique_grid_counts"]==[2]
    assert result["integration"]=="full_empirical_product"
    assert not result["publication_allowed"] and not result["forecast_precision_verified"]


@pytest.mark.parametrize("arm",["legacy_grid_qualifying","grid_only"])
def test_product_matches_independent_nested_sum_and_normalizes_orders(arm):
    forecast,beta,panels=mixture_fixture(3)
    probabilities=[]
    for permutation in itertools.permutations(range(1,4)):
        panel={"2026-01":dict(zip(forecast["events"][0]["drivers"],permutation))}
        actual=np.exp(integrate(forecast,beta,panel,arm=arm)["log_probabilities"][0])
        np.testing.assert_allclose(actual,direct_probability(forecast,beta,panel,arm),rtol=1e-14)
        probabilities.append(actual)
    np.testing.assert_allclose(sum(probabilities),1.,rtol=1e-14)


def test_independent_inventory_permutations_and_product_blocks_preserve_score():
    forecast,beta,panels=mixture_fixture(3)
    expected=integrate(forecast,beta,panels)["log_probabilities"]
    changed=copy.deepcopy(forecast)
    e=changed["events"][0]
    e["grid_draws"]=e["grid_draws"].reshape(-1,3)[[5,1,4,3,2,0]].reshape(2,3,3)
    bind_grids(changed)
    other={k:v.reshape(-1)[[2,0,3,1]].reshape(2,2) for k,v in beta.items()}
    for grid_batch,coefficient_batch in ((1,1),(1,3),(2,2),(11,7)):
        actual=integrate(changed,other,panels,grid_batch_size=grid_batch,coefficient_batch_size=coefficient_batch)
        np.testing.assert_allclose(actual["log_probabilities"],expected,rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize("panel",[{"d0":0,"d1":0},{"d0":0,"d1":1}])
def test_empty_or_singleton_panel_has_exact_unit_conditional_mass(panel):
    forecast,beta,_=mixture_fixture()
    assert integrate(forecast,beta,{"2026-01":panel})["log_probabilities"].item()==0.


def test_qualifying_batch_field_and_path_prefix_invariance():
    _,design,snap,p=qualifying_case()
    path=future_path(snap)
    short=generate(design,path[:1],p,batch=1)
    longer=generate(design,path,p,batch=5)
    shuffled=copy.deepcopy(path[0]); shuffled["entrants"].reverse()
    reverse=generate(design,[shuffled],p,batch=19)
    for result in (longer,reverse):
        np.testing.assert_array_equal(result["events"][0]["grid_draws"],short["events"][0]["grid_draws"])
        for name,value in short["events"][0]["race_features"].items():
            np.testing.assert_array_equal(result["events"][0]["race_features"][name],value)
    event=short["events"][0]; shape=p["nu"].shape
    direct=qualifying_segments(event,p,pre.LegacyObservationNoise(41,shape,(0,shape[1])),advancing=(4,2),full_df=p["nu"])
    np.testing.assert_array_equal(event["grid_draws"],direct["positions"])


def test_generated_grid_forecast_connects_to_each_baseline_for_complete_path():
    _,design,snap,p=qualifying_case()
    forecast=generate(design,future_path(snap),p)
    _,beta,_=mixture_fixture()
    panels={e["event_id"]:{d:i+1 for i,d in enumerate(e["drivers"])} for e in forecast["events"]}
    for arm in ("legacy_grid_qualifying","grid_only"):
        result=integrate(forecast,beta,panels,arm=arm)
        assert result["event_ids"]==[e["event_id"] for e in forecast["events"]]
        assert result["qualifying_training_arrays_key"]==forecast["training_arrays_key"]
        for i,event in enumerate(forecast["events"]):
            expected=direct_probability(dict(forecast,events=[event]),beta,panels,arm)
            np.testing.assert_allclose(np.exp(result["log_probabilities"][i]),expected,rtol=1e-14)
        assert not result["official_grid_policy_complete"]
        assert not result["publication_allowed"]


def test_qualifying_locations_scales_and_density_match_original_legacy_model():
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from scipy.stats import t
    from f1rank.model import model
    source,design,snap,p=qualifying_case()
    event=generate(design,[snap],p,advancing=())["events"][0]
    shape=p["nu"].shape; noise=pre.LegacyObservationNoise(41,shape,(0,shape[1]))
    generated=qualifying_segments(event,p,noise,advancing=(),full_df=p["nu"])
    full,expanded=full_fixture(source,design,snap,event,p,keep_future_times=True)
    future=full.obs[(full.obs.event_id==snap["event"]["event_id"])&(full.obs.segment=="Q1")].sort_values("driver_id")
    assert list(future.driver_id)==event["drivers"]
    identity=[event["event_id"],0]; session=int(future.session_idx.iloc[0])
    expanded["sigma_u"][...,session]=noise.draw("qualifying_scale",identity)
    for row in future.itertuples():
        expanded["car_segment_z"][...,row.car_segment_idx]=noise.draw("qualifying_car_segment",[*identity,row.team])
    run=handlers.seed(handlers.condition(model,data={k:jnp.asarray(v[0,0]) for k,v in expanded.items()}),jax.random.PRNGKey(19))
    trace=handlers.trace(run).get_trace({k:jnp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in full.arrays().items()})
    likelihood=trace["y"]["fn"].base_dist; segment=generated["segments"][0]
    np.testing.assert_allclose(segment["location"][0,0],np.asarray(likelihood.loc)[future.index],rtol=3e-6,atol=3e-6)
    np.testing.assert_allclose(segment["scale"][0,0],np.asarray(likelihood.scale)[future.index],rtol=3e-6)
    values=np.asarray(trace["y"]["value"]).copy(); values[future.index]=segment["pace"][0,0]
    expected=np.asarray(likelihood.log_prob(jnp.asarray(values)))[future.index]
    actual=t.logpdf(segment["pace"][0,0],p["nu"][0,0],loc=segment["location"][0,0],scale=segment["scale"][0,0])
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize("fault",["last_nan","bad_df","negative_scale","axes","missing","actual_grid","advance"])
def test_invalid_qualifying_inputs_cannot_leave_a_partial_forecast(fault):
    _,design,snap,p=qualifying_case()
    if fault=="last_nan": p["sigma0"][-1,-1]=np.nan
    elif fault=="bad_df": p["nu"][-1,-1]=0
    elif fault=="negative_scale": p["sigma_tau"][-1,-1]=-.1
    elif fault=="axes": p["nu"]=p["nu"][:,:-1]
    elif fault=="missing": del p["nu"]
    elif fault=="actual_grid": snap["information_set"]="actual_grid"
    with pytest.raises(ValueError): generate(design,[snap],p,advancing=(6,) if fault=="advance" else (4,2))


@pytest.mark.parametrize("fault",["grid_tail","grid_shape","grid_mutation","feature_mutation","beta_axes","beta_tail",
                                   "missing_event","missing_driver","bad_order","actual_grid","duplicate_event","batch_bool"])
def test_incomplete_or_changed_product_inputs_fail(fault):
    forecast,beta,panels=mixture_fixture()
    event=forecast["events"][0]
    if fault=="grid_tail": event["grid_draws"][-1,-1,-1]=event["grid_draws"][-1,-1,0]
    elif fault=="grid_shape": event["grid_draws"]=event["grid_draws"][:,:-1]
    elif fault=="grid_mutation": event["grid_draws"][-1,-1]=event["grid_draws"][-1,-1,::-1]
    elif fault=="feature_mutation": event["race_features"]["driver"][0]+=.1
    elif fault=="beta_axes": beta["a_car"]=beta["a_car"][:,:1]
    elif fault=="beta_tail": beta["b_driver"][-1,-1]=np.nan
    elif fault=="missing_event": panels.clear()
    elif fault=="missing_driver": del panels["2026-01"]["d0"]
    elif fault=="bad_order": panels["2026-01"]["d1"]=3
    elif fault=="actual_grid": event["catalog"]["information_set"]="actual_grid"
    elif fault=="duplicate_event": forecast["events"].append(copy.deepcopy(event))
    with pytest.raises(ValueError): integrate(forecast,beta,panels,grid_batch_size=True if fault=="batch_bool" else 2)
