"""Mathematical checks for the proposed coordinates; no MCMC or driver rankings."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.overall_control_rotation import ROTATED_SITE, build_rotation, rotated_model
from analysis.overall_profile_geometry import control_design, local_geometry, saved_controls


def test_curvature_has_known_student_t_and_ar_values():
    simple = local_geometry(np.diag([1., 2.]), np.ones(2), np.zeros(2, bool), 0., 1., 1.)
    np.testing.assert_allclose(simple["conditional_zero_residual_curvature_eigenvalues"], [3, 9])
    assert simple["conditional_zero_residual_curvature_ratio"] == pytest.approx(3)
    # The first row uses the initial residual scale; the second uses the AR term.
    actual = local_geometry(np.ones((2, 1)), np.ones(1), np.array([False, True]), .5, 1., 1.)
    np.testing.assert_allclose(actual["conditional_zero_residual_curvature_eigenvalues"], [3.])


def test_archive_coordinate_extraction_preserves_chains_and_draws():
    original = {"a": np.arange(24).reshape(2, 3, 4), "b": np.arange(48).reshape(2, 3, 2, 4)}
    actual = saved_controls(original, ["a[1]", "b[1,2]"])
    np.testing.assert_array_equal(actual[..., 0], original["a"][..., 1])
    np.testing.assert_array_equal(actual[..., 1], original["b"][..., 1, 2])


@pytest.fixture(scope="module")
def design():
    from f1rank.overall.recovery import synthetic_design
    return synthetic_design("connected")


def test_rotation_uses_no_responses_and_preserves_all_prior_directions(design):
    rotation = build_rotation(design.laps)
    changed = copy.deepcopy(design.laps)
    changed["y"] = np.full(len(changed["y"]), np.nan)
    assert rotation.fingerprint() == build_rotation(changed).fingerprint()
    z = np.random.default_rng(0).normal(size=rotation.size)
    controls = rotation.original(z)
    np.testing.assert_allclose(rotation.rotated(controls), z, atol=1e-6, rtol=1e-6)
    for block in rotation.blocks:
        transform = np.diag(block["scales"]) @ block["basis"]
        # Analytical covariance: the original independent Normal priors survive.
        np.testing.assert_allclose(transform @ transform.T, np.diag(block["scales"] ** 2), atol=1e-12)


def test_deficient_design_keeps_unused_and_null_prior_directions():
    laps = dict(n_race=1, n_rc=1, race=np.zeros(2, int), lap=np.zeros(2),
        rc=np.zeros(2, int), age=np.ones(2), nonref=np.zeros(2, bool), traffic=np.zeros((2, 3)))
    rotation = build_rotation(laps)
    assert rotation.size == 7  # All original controls, even with fewer observations.
    z = np.arange(rotation.size, dtype=float)
    np.testing.assert_allclose(rotation.rotated(rotation.original(z)), z, atol=1e-6)


def test_design_matrix_reproduces_original_control_mean(design):
    r = design.laps
    rotation = build_rotation(r)
    controls = rotation.original(np.random.default_rng(1).normal(size=rotation.size))
    controls = {key: np.asarray(value) for key, value in controls.items()}
    direct = (controls["overall_intercept"][r["race"]]
        + controls["overall_trend"][r["race"]] * r["lap"]
        + np.where(r["nonref"], controls["overall_compound"][r["rc"]], 0)
        + controls["overall_tyre_slope"][r["rc"]] * r["age"]
        + (controls["overall_traffic"][r["race"]] * r["traffic"]).sum(-1))
    for race in range(r["n_race"]):
        matrix, names, _, mask = control_design(r, race)
        coefficients = saved_controls({key:value[None, None] for key, value in controls.items()}, names)[0, 0]
        np.testing.assert_allclose(matrix @ coefficients, np.asarray(direct)[mask], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("case,seed", [("connected", 8), ("confounded", 9), ("sparse_rookie", 10)])
def test_full_joint_density_is_preserved_with_change_of_variables(case, seed):
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from numpyro.infer.util import log_density
    from f1rank.overall.model import model
    from f1rank.overall.recovery import synthetic_design

    d = synthetic_design(case)
    args = (d.qualifying.arrays(), d.laps, d.outcomes)
    rotation = build_rotation(d.laps)
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*args)
    params = {name:site["value"] for name, site in trace.items()
              if site["type"] == "sample" and not site["is_observed"]}
    original_controls = {name:params[name] for name in rotation.shapes}
    updated = {name:value for name, value in params.items() if name not in rotation.shapes}
    updated[ROTATED_SITE] = jnp.asarray(rotation.rotated(original_controls))
    before, original_trace = log_density(model, args, {}, params)
    after, new_trace = log_density(rotated_model(rotation), args, {}, updated)
    np.testing.assert_allclose(after, before + rotation.log_abs_det, atol=5e-3, rtol=2e-6)
    for name in rotation.shapes:
        assert new_trace[name]["type"] == "deterministic"
        np.testing.assert_allclose(new_trace[name]["value"], original_controls[name], atol=2e-6, rtol=2e-6)
    for name in ("skill", "overall_race_skill", "overall_car", "overall_team", "overall_day",
                 "overall_finish_order", "overall_completion"):
        np.testing.assert_array_equal(new_trace[name]["value"], original_trace[name]["value"])
    # Check the actual lap likelihood term too, rather than only the large total.
    np.testing.assert_allclose(new_trace["overall_laps"]["fn"].log_factor,
                               original_trace["overall_laps"]["fn"].log_factor, atol=5e-3, rtol=2e-6)


def test_rotated_joint_density_has_finite_gradient(design):
    import jax
    from numpyro.infer.util import initialize_model
    rotation = build_rotation(design.laps)
    initialized = initialize_model(jax.random.PRNGKey(11), rotated_model(rotation),
        model_args=(design.qualifying.arrays(), design.laps, design.outcomes))
    assert all(np.isfinite(a).all() for a in jax.tree_util.tree_leaves(initialized.param_info.z_grad))


def test_misaligned_or_missing_coordinates_are_rejected(design):
    rotation = build_rotation(design.laps)
    with pytest.raises(ValueError, match="dimensions"):
        rotation.original(np.zeros(rotation.size - 1))
    with pytest.raises(ValueError, match="sites"):
        rotation.rotated({})
    bad = copy.deepcopy(design.laps)
    bad["n_rc"] += 1
    with pytest.raises(ValueError, match="every original control"):
        build_rotation(bad)


def test_pending_proposal_cannot_launch_a_fit(tmp_path, monkeypatch):
    from f1rank.overall import fitting
    proposal = Path(__file__).resolve().parents[1] / "docs/overall_rotation_profile_proposal.json"
    # Exercise a pending registration even after the real proposal is approved.
    pending = json.loads(proposal.read_text())
    pending.update(status="proposal_pending_user_approval", user_approved=False)
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(pending))
    monkeypatch.setattr(fitting.subprocess, "check_output", lambda *a, **k:path.read_bytes())
    with pytest.raises(ValueError, match="frozen"):
        fitting.committed_registration(path, root=tmp_path)


@pytest.mark.parametrize("change", ["approval", "seed", "settings", "gate", "inputs"])
def test_matched_experiment_rejects_unauthorized_or_changed_registration(monkeypatch, change):
    from analysis import overall_rotation_profile as runner
    proposal = runner.ROOT / "docs/overall_rotation_profile_proposal.json"
    registration = json.loads(proposal.read_text())
    registration.update(status="frozen", user_approved=True)
    if change == "approval":
        registration["user_approved"] = False
    elif change == "seed":
        registration["synthetic"]["confounded"]["seed"] += 1
    elif change == "settings":
        registration["fit_settings"]["max_tree_depth"] += 1
    elif change == "gate":
        registration["diagnostics"]["bulk_ess_min"] -= 1
    else:
        registration["registered_model_inputs_key"] = "changed"
    monkeypatch.setattr(runner, "committed_registration", lambda _:registration)
    with pytest.raises(ValueError):
        runner.checked_registration(proposal)


def test_recorded_experiment_is_never_overwritten_or_restarted(tmp_path, monkeypatch):
    from analysis import overall_rotation_profile as runner
    destination = tmp_path / "outputs/championship/development/example"
    destination.mkdir(parents=True)
    marker = destination / "summary.json"
    marker.write_text('original failed experiment')
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "checked_registration", lambda _:dict(id="example"))
    monkeypatch.setattr(runner, "snapshot", lambda: {})
    with pytest.raises(FileExistsError, match="Preserve"):
        runner.execute(tmp_path / "registration.json")
    assert marker.read_text() == 'original failed experiment'
