import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from f1rank.design import SIGN_ANCHOR, build_design, load_tables
from f1rank.lineage import LINEAGE
from f1rank.model import centre, linear_recurrence


def test_linear_recurrence_matches_loop():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 1.0, 50)
    a[[0, 17, 33]] = 0.0  # sequence starts
    b = rng.standard_normal(50)
    x = np.zeros(50)
    for j in range(50):
        x[j] = (a[j] * x[j - 1] if j else 0.0) + b[j]
    np.testing.assert_allclose(np.asarray(linear_recurrence(jnp.asarray(a), jnp.asarray(b))), x,
                               rtol=1e-5, atol=1e-5)


def test_centre_zero_mean_per_group():
    x = jnp.array([1.0, 2.0, 3.0, 10.0, 20.0])
    g = jnp.array([0, 0, 0, 1, 1])
    out = np.asarray(centre(x, g, 2))
    np.testing.assert_allclose([out[:3].mean(), out[3:].mean()], [0, 0], atol=1e-6)


@pytest.fixture(scope="module")
def design():
    return build_design(2010)


def test_every_constructor_has_a_lineage():
    assert set(load_tables()["entries"].constructor_id) <= set(LINEAGE)


def test_entries_sorted_and_first_flags(design):
    e = design.entries
    assert (e.groupby("driver_idx").event_idx.diff().dropna() > 0).all()
    assert e.groupby("driver_idx")["first"].sum().eq(1).all()
    c = design.cars
    assert (c.groupby("team").event_idx.diff().dropna() > 0).all()
    assert c.groupby("team")["first"].sum().eq(1).all()


def test_observations_link_consistently(design):
    o, e, c = design.obs, design.entries, design.cars
    assert (e.event_idx.to_numpy()[o.entry_idx] == o.event_idx.to_numpy()).all()
    assert (c.event_idx.to_numpy()[o.car_idx] == o.event_idx.to_numpy()).all()
    assert (e.team.to_numpy()[o.entry_idx] == c.team.to_numpy()[o.car_idx]).all()
    assert (o.y >= -5).all()  # slow outliers dropped; fast ones (wet, drying track) kept and down-weighted


def test_circuit_factor_orientation(design):
    idx = design.events.groupby("circuit_id").circuit_idx.first()
    hi, lo = (idx[c] for c in SIGN_ANCHOR)
    assert design.circuit_factor[hi] > design.circuit_factor[lo]


def test_cutoff_masks_future_and_recomputes_circuit_factor(design):
    cut = int(design.events.query("season == 2020").event_idx.max())
    d2 = design.with_cutoff(cut)
    assert not d2.train[design.obs.event_idx.to_numpy() > cut].any()
    assert d2.train[design.obs.event_idx.to_numpy() <= cut].all()
    assert not np.allclose(d2.circuit_factor, design.circuit_factor)
