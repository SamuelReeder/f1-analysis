"""Checks that saved fits, snapshots and job runs cannot silently go wrong."""

import json
import subprocess
import sys

import numpy as np
import pytest

from f1rank import jobs
from f1rank.design import build_design, load_tables
from f1rank.diagnostics import residuals
from f1rank.export import write_once
from f1rank.fit import IncompatibleFit, design_for, fingerprint, load, save


@pytest.fixture(scope="module")
def tables():
    return load_tables()


@pytest.fixture(scope="module")
def designs(tables):
    latest = build_design(2010, tables=tables)
    previous_event = latest.events.event_id.iloc[-2]
    return latest, build_design(2010, end_event=previous_event, tables=tables)


def fake_post(design, n_draws=2, **values):
    """Posterior-shaped arrays (1 chain) for every model term, zero unless given."""
    a = design.arrays()
    size = {"mu": a["n_sessions"], "sigma_u": a["n_sessions"], "car": a["n_cars"],
            "car_event": a["n_cars"], "car_track": a["n_cars"], "skill": a["n_entries"],
            "compat": a["n_entries"], "driver_form": a["n_entries"], "car_segment": a["n_car_segments"]}
    post = {k: np.full((1, n_draws, n), values.get(k, 0.0), np.float32) for k, n in size.items()}
    post.update(sigma0=np.ones((1, n_draws), np.float32), sigma_tau=np.zeros((1, n_draws), np.float32))
    return post


def test_fingerprint_changes_when_a_race_is_added(designs, tables):
    latest, previous = designs
    assert fingerprint(latest) != fingerprint(previous)
    again = build_design(2010, end_event=previous.events.event_id.iloc[-1], tables=tables)
    assert fingerprint(again) == fingerprint(previous)


def test_load_refuses_a_fit_made_on_other_data(designs, tmp_path):
    latest, previous = designs
    path = tmp_path / "old.npz"
    save(path, {"x": np.zeros((1, 2, 3))}, {"divergences": 0}, previous)
    load(path, previous)  # same data: fine
    with pytest.raises(IncompatibleFit):
        load(path, latest)  # a new race shifts the indices
    (tmp_path / "bare.npz").write_bytes(path.read_bytes())
    with pytest.raises(IncompatibleFit):  # no metadata: refused
        load(tmp_path / "bare.npz", previous)


def test_design_for_rebuilds_the_fitted_design(designs, tmp_path):
    _, previous = designs
    cut = previous.with_cutoff(int(previous.events.event_idx.iloc[-30]))
    path = tmp_path / "lfo.npz"
    save(path, {"x": np.zeros((1, 2, 3))}, {}, cut)
    assert fingerprint(design_for(path)) == fingerprint(cut)


def test_circuit_factors_finite_for_short_history(tables):
    for start in (2025, 2026):
        f = build_design(start, tables=tables).circuit_factor
        assert np.isfinite(f).all()


def test_residuals_use_every_fitted_term(designs):
    latest, _ = designs
    r = residuals(latest, fake_post(latest, compat=0.5, driver_form=0.25))
    np.testing.assert_allclose(r.resid, latest.obs.y - 0.75, atol=1e-5)
    post = fake_post(latest)
    del post["driver_form"]
    with pytest.raises(KeyError):
        residuals(latest, post)
    residuals(latest, post, driver_form=False)  # fine when the model had no form term


def test_snapshots_are_never_overwritten(tmp_path):
    path = tmp_path / "2026-15_quali-v1_20260101T000000Z.json"
    assert write_once(path, {"v": 1})
    assert not write_once(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 1}


def test_failed_job_fails_the_command():
    with pytest.raises(SystemExit) as e:
        jobs.main(["run", "no_such_job"])
    assert e.value.code == 1
    out = subprocess.run([sys.executable, "-m", "f1rank.jobs", "all", "no_such_job", "--parallel", "1"],
                         capture_output=True, text=True)
    assert out.returncode == 1


def test_synthetic_jobs_need_an_explicit_source(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "SYNTH_SOURCE", tmp_path / "synth_source.npz")
    with pytest.raises(FileNotFoundError, match="synth-source"):
        jobs.run("synth_clean")


@pytest.mark.parametrize("unit", ["spell", "era"])
def test_team_effect_variants_fit(tables, unit):
    from functools import partial

    from f1rank.fit import fit
    from f1rank.model import model
    d = build_design(2026, tables=tables)
    post, _ = fit(d, warmup=3, samples=2, chains=1, progress=False, model_fn=partial(model, compat_unit=unit))
    assert post["compat"].shape[-1] == len(d.entries)
    assert np.isfinite(post["compat"]).all()
