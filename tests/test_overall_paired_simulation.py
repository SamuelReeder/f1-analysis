"""Exact paired-kernel equivalence; no posterior or scientific acceptance."""
from dataclasses import replace

import numpy as np
import pytest

from analysis.overall_paired_simulation import PROBE_ORDER, paired_seasons
from f1rank.overall.simulation import JointDraws, QUALIFYING_FIELDS, Scenario, simulate


def fixture(n=6, nd=3, dtype=np.float64, seed=17):
    rng = np.random.default_rng(seed)
    return JointDraws(drivers=tuple(f"driver_{i:02}" for i in rng.permutation(n)),
        qualifying=rng.normal(size=(nd, n)).astype(dtype), race=rng.normal(size=(nd, n)).astype(dtype),
        q_weekend_sd=rng.uniform(.1, .6, nd).astype(dtype), r_weekend_sd=rng.uniform(.1, .6, nd).astype(dtype),
        shared_weekend=rng.normal(size=nd).astype(dtype), q_scale=rng.uniform(.1, .8, nd).astype(dtype),
        q_log_scale_sd=rng.uniform(.1, .4, nd).astype(dtype), q_df=rng.uniform(2, 10, nd).astype(dtype),
        grid_coefficient=rng.uniform(.1, 2., (nd, 2)).astype(dtype),
        pace_coefficient=rng.uniform(.1, 2., (nd, 2)).astype(dtype),
        retirement_hazard=np.broadcast_to(np.array([.0, .008], dtype=dtype), (nd, 2)).copy())


def reference(joint, scenarios, **settings):
    baseline = simulate(joint, scenarios, **settings)
    probes = np.empty(baseline["points"].shape + (len(PROBE_ORDER),))
    order = np.argsort(joint.drivers)
    for driver in range(len(joint.drivers)):
        for k, components in enumerate((("qualifying",), ("race",), ("qualifying", "race"))):
            changed = {}
            for name in components:
                original = getattr(joint, name)
                changed[name] = original.copy()
                changed[name][:, driver] = original[:, order].mean(axis=-1)
            result = simulate(replace(joint, **changed), scenarios, **settings)
            probes[..., driver, k] = result["points"][..., driver]
    return baseline, probes


def assert_same(actual, expected):
    a, ap = actual
    b, bp = expected
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    np.testing.assert_array_equal(ap, bp)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [0, 71, [43, 1, 128, 256]])
def test_every_intervention_matches_original_seasons_exactly(dtype, seed):
    joint = fixture(dtype=dtype)
    before = {key:getattr(joint, key).copy() for key in joint.__dataclass_fields__ if key != "drivers"}
    scenarios = [Scenario("green", 0, 45, 45), Scenario("retirements", 1, 60, 60),
                 Scenario("no-points", 0, 50, 0)]
    settings = dict(replicates=3, seed=seed, advancing=(4, 2))
    assert_same(paired_seasons(joint, scenarios, **settings), reference(joint, scenarios, **settings))
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(joint, name), value)


@pytest.mark.parametrize("n", sorted(QUALIFYING_FIELDS))
@pytest.mark.parametrize("explicit", [True, False])
def test_supported_qualifying_fields_have_exact_probe_points(n, explicit):
    joint = fixture(n=n, nd=2)
    scenarios = [Scenario("reference", 1, 45, 45)]
    settings = dict(replicates=2, seed=[71, 3, 256, 128], advancing=QUALIFYING_FIELDS[n] if explicit else None)
    assert_same(paired_seasons(joint, scenarios, **settings), reference(joint, scenarios, **settings))


def test_permuted_field_and_noop_interventions_keep_original_identity_pairing():
    joint = fixture()
    scenarios = [Scenario("reference", 1, 60, 60)]
    settings = dict(replicates=3, seed=[71, 0, 0, 0], advancing=(4, 2))
    baseline, probes = paired_seasons(joint, scenarios, **settings)
    order = [3, 0, 5, 2, 1, 4]
    permuted = replace(joint, drivers=tuple(joint.drivers[i] for i in order),
                       qualifying=joint.qualifying[:, order], race=joint.race[:, order])
    other, op = paired_seasons(permuted, scenarios, **settings)
    for key in ("points", "title_share", "rank_low", "rank_high"):
        np.testing.assert_array_equal(other[key], baseline[key][..., order])
    np.testing.assert_array_equal(op, probes[..., order, :])
    zero = replace(joint, qualifying=np.zeros_like(joint.qualifying), race=np.zeros_like(joint.race))
    baseline, probes = paired_seasons(zero, scenarios, **settings)
    np.testing.assert_array_equal(probes, np.repeat(baseline["points"][..., None], len(PROBE_ORDER), axis=-1))


def test_exact_sporting_ties_and_compensated_large_strengths_are_unchanged():
    joint = fixture()
    # A no-points season can retain championship ties. Extreme common ability
    # also exercises the original compensated Gumbel ordering without overflow.
    joint = replace(joint, race=np.full_like(joint.race, 1e20))
    scenarios = [Scenario("reference", 0, 60, 0)]
    settings = dict(replicates=3, seed=[71, 0, 0, 0], advancing=(4, 2))
    assert_same(paired_seasons(joint, scenarios, **settings), reference(joint, scenarios, **settings))


@pytest.mark.parametrize("case", ["all-retired", "duplicate", "bad-context", "bad-control"])
def test_original_failures_remain_failures(case):
    joint = fixture()
    scenarios = [Scenario("reference", 0, 60, 60)]
    if case == "all-retired":
        joint = replace(joint, retirement_hazard=np.full_like(joint.retirement_hazard, .999))
    elif case == "duplicate":
        scenarios *= 2
    elif case == "bad-context":
        scenarios = [Scenario("reference", 2, 60, 60)]
    else:
        scenarios = [Scenario("reference", 0, 60, 61)]
    settings = dict(replicates=3, seed=71, advancing=(4, 2))
    with pytest.raises(ValueError) as original:
        reference(joint, scenarios, **settings)
    with pytest.raises(ValueError) as optimized:
        paired_seasons(joint, scenarios, **settings)
    assert str(original.value) == str(optimized.value)
