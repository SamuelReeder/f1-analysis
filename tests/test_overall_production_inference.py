"""Production wiring and numerical identity; no MCMC or acceptance experiment."""
import ast
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from f1rank.overall import fitting, inference
from test_overall_fitting import runner, sampled

ROOT = Path(__file__).resolve().parents[1]


def definitions(path):
    class Normalize(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            # Relocation is the only permitted executable change in these copies.
            node.module = node.module.removeprefix("f1rank.overall.")
            node.level = 0
            return node

        def visit_FunctionDef(self, node):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
            return self.generic_visit(node)

    tree = Normalize().visit(ast.parse(path.read_text()))
    return {node.name: ast.dump(node) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


@pytest.mark.parametrize("name", ["control_rotation", "batched_rotation", "weekend_coordinates", "bounded_diagnostics"])
def test_every_migrated_definition_preserves_the_verified_implementation(name):
    old = definitions(ROOT / "analysis" / f"overall_{name}.py")
    new = definitions(ROOT / "f1rank/overall" / f"{name}.py")
    for key in old:
        assert new[key] == old[key], key
    if name == "control_rotation":
        assert new["control_design"] == definitions(ROOT / "analysis/overall_profile_geometry.py")["control_design"]
    else:
        assert set(new) == set(old)


def test_new_modules_are_in_source_snapshot_and_do_not_import_research_code():
    from f1rank.overall.provenance import snapshot
    sources = snapshot()["inputs"]
    for name in ("control_rotation", "batched_rotation", "weekend_coordinates", "bounded_diagnostics", "inference"):
        relative = f"f1rank/overall/{name}.py"
        assert relative in sources
        tree = ast.parse((ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("analysis")
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("analysis") for alias in node.names)


@pytest.mark.parametrize("case", ["connected", "confounded", "sparse_rookie", "dominant_teammate", "era_boundary"])
def test_production_trace_density_and_gradient_match_checked_batched_model(case):
    import jax
    from numpyro import handlers
    from numpyro.infer.util import log_density, potential_energy, unconstrain_fn
    from analysis.overall_control_rotation import build_rotation as old_rotation
    from analysis.overall_batched_rotation import prepare as old_prepare
    from analysis.overall_weekend_coordinates import weekend_model as old_model
    from f1rank.overall.control_rotation import build_rotation
    from f1rank.overall.batched_rotation import prepare
    from f1rank.overall.weekend_coordinates import weekend_model
    from f1rank.overall.recovery import synthetic_design

    d = synthetic_design(case)
    args = (d.qualifying.arrays(), d.laps, d.outcomes)
    old_map, new_map = old_prepare(old_rotation(d.laps)), prepare(build_rotation(d.laps))
    assert old_map.fingerprint() == new_map.fingerprint()
    old, new = old_model(old_map), weekend_model(new_map)
    trace = handlers.trace(handlers.seed(old, jax.random.PRNGKey(53))).get_trace(*args)
    params = {k: site["value"] for k, site in trace.items()
              if site["type"] == "sample" and not site["is_observed"]}
    before, a = log_density(old, args, {}, params)
    after, b = log_density(new, args, {}, params)
    np.testing.assert_array_equal(after, before)
    assert a.keys() == b.keys()
    for name in a:
        if a[name]["type"] in ("sample", "deterministic"):
            np.testing.assert_array_equal(a[name]["value"], b[name]["value"], err_msg=name)
        if a[name]["type"] == "sample":
            np.testing.assert_array_equal(a[name]["fn"].log_prob(a[name]["value"]),
                                          b[name]["fn"].log_prob(b[name]["value"]), err_msg=name)
    z = unconstrain_fn(old, args, {}, params)
    derivatives = []
    for model in (old, new):
        value, gradient = jax.value_and_grad(lambda values: potential_energy(model, args, {}, values))(z)
        derivatives.append((value, gradient))
    for expected, actual in zip(jax.tree_util.tree_leaves(derivatives[0]), jax.tree_util.tree_leaves(derivatives[1])):
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, expected)
    jax.clear_caches()


def test_bounded_diagnostics_and_every_driver_comparison_match_research_reports():
    from analysis.overall_bounded_diagnostics import check
    from analysis.overall_weekend_profile import contrast_reports
    rng = np.random.default_rng(72)
    post = {name: rng.normal(size=(4, 256, 6)) for name in ("skill", "overall_race_skill")}
    post["skill"][-1, :, 3] += 3
    catalog = dict(field=["a", "b", "c"], field_entries=[0, 3, 5])
    assert inference.check(post, 1) == check(post, 1)
    assert inference.contrast_reports(post, catalog, 1) == contrast_reports(post, catalog, 1)


def test_sampler_wires_composed_model_and_returns_every_extra_without_running_mcmc(monkeypatch):
    import jax
    from numpyro import handlers
    import numpyro.infer
    from f1rank.overall.recovery import synthetic_design
    from f1rank.overall.control_rotation import ROTATED_SITE
    from f1rank.overall.weekend_coordinates import TOTAL, COMPLEMENT

    d = synthetic_design("connected")
    blocks = dict(qualifying=d.qualifying.arrays(), laps=d.laps, outcomes=d.outcomes)
    settings = dict(fitting.FIT_SETTINGS, samples=128)
    rng = np.random.default_rng(94)
    post = {name: rng.normal(size=(4, 128, blocks["qualifying"]["n_entries"]))
            for name in ("skill", "overall_race_skill")}
    extras = result(settings, True).extras
    extras["diverging"][0, 0] = True
    extras["num_steps"][0, 0] = 2**settings["max_tree_depth"]-1
    seen = {}

    class Kernel:
        def __init__(self, model, **kwargs):
            self.model = model
            seen["kernel"] = kwargs

    class Sampler:
        def __init__(self, kernel, **kwargs):
            self.kernel = kernel
            seen["sampler"] = kwargs

        def run(self, key, *args, extra_fields):
            seen["extra_fields"] = extra_fields
            trace = handlers.trace(handlers.seed(self.kernel.model, key)).get_trace(*args)
            for name in (ROTATED_SITE, TOTAL, COMPLEMENT):
                assert trace[name]["type"] == "sample"
            for name in ("overall_intercept", "driver_form_z", "overall_day_z"):
                assert trace[name]["type"] == "deterministic"

        def get_samples(self, *, group_by_chain):
            assert group_by_chain is True
            return post

        def get_extra_fields(self, *, group_by_chain):
            assert group_by_chain is True
            return extras

    monkeypatch.setattr(jax, "local_device_count", lambda: settings["chains"])
    monkeypatch.setattr(numpyro.infer, "NUTS", Kernel)
    monkeypatch.setattr(numpyro.infer, "MCMC", Sampler)
    sampled = inference.sample(blocks, settings, d.catalog)
    assert seen["extra_fields"] == inference.EXTRA_FIELDS
    assert seen["kernel"]["target_accept_prob"] == settings["target_accept"]
    assert seen["kernel"]["max_tree_depth"] == settings["max_tree_depth"]
    assert seen["sampler"] == dict(num_warmup=settings["warmup"], num_samples=settings["samples"],
        num_chains=settings["chains"], chain_method="parallel", progress_bar=False)
    assert sampled.posterior.keys() == post.keys() and sampled.extras.keys() == extras.keys()
    for name in post:
        np.testing.assert_array_equal(sampled.posterior[name], post[name])
    for name in extras:
        np.testing.assert_array_equal(sampled.extras[name], extras[name])
    assert sampled.diagnostics["divergences"] == 1
    assert sampled.diagnostics["tree_depth_limit_fraction"] == 1 / extras["num_steps"].size
    assert sampled.diagnostics["inference_method"] == inference.METHOD


def coordinate_registration(path):
    reg = json.loads(path.read_text())
    reg.update(schema_version=fitting.COORDINATE_RUN_SCHEMA,
               inference_method=inference.METHOD, registered_model_inputs_key="snapshot")
    path.write_text(json.dumps(reg))
    return reg


def result(settings, success):
    post, diagnostic = sampled(settings, success)
    axes = (settings["chains"], settings["samples"])
    extras = dict(diverging=np.zeros(axes, bool), num_steps=np.full(axes, 3, dtype=np.int32),
                  accept_prob=np.full(axes, .75, dtype=np.float32))
    diagnostic.update(mean_accept_prob=.75, tree_depth_limit_fraction=0., inference_method=inference.METHOD)
    return inference.SampleResult(post, diagnostic, extras)


def test_registered_path_calls_production_sampler_and_preserves_each_attempt(runner, monkeypatch):
    blocks, path, directory = runner
    coordinate_registration(path)
    calls = []
    catalog = {"identity": "original catalog"}
    def sample(b, settings, c):
        assert b is blocks and c is catalog
        calls.append(settings)
        return result(settings, len(calls) == len(fitting.RETRY))
    monkeypatch.setattr(inference, "sample", sample)
    post, receipt = fitting.run(blocks, catalog, directory, path, study="toy", synthetic=True)
    assert receipt["schema_version"] == fitting.COORDINATE_RUN_SCHEMA
    assert len(calls) == len(fitting.RETRY)
    assert receipt["state"] == "converged" and receipt["publication_allowed"] is False
    for i, attempt in enumerate(receipt["attempts"], 1):
        assert (directory / f"sampler_{i}.npz").exists()
        assert attempt["sampler_sha256"] == fitting.digest(directory / attempt["sampler_file"])
        assert attempt["posterior_sites"]["latent"]["shape"] == [attempt["chains"], attempt["samples"], 1]
    cached, repeated = fitting.run(blocks, catalog, directory, path, study="toy", synthetic=True)
    assert len(calls) == len(fitting.RETRY) and repeated == receipt
    np.testing.assert_array_equal(cached["latent"], post["latent"])


@pytest.mark.parametrize("change", ["source", "method", "override"])
def test_new_registration_rejects_mismatched_source_method_or_sampler(runner, change):
    blocks, path, directory = runner
    reg = coordinate_registration(path)
    if change == "source": reg["registered_model_inputs_key"] = "stale"
    elif change == "method": reg["inference_method"] = "foreign"
    path.write_text(json.dumps(reg))
    options = {"sample": lambda *args: pytest.fail("Must not run")} if change == "override" else {}
    with pytest.raises(ValueError):
        fitting.run(blocks, {}, directory, path, study="toy", synthetic=True, **options)
    assert not directory.exists()


@pytest.mark.parametrize("change", ["bytes", "missing", "diagnostic", "replacement_during_read"])
def test_posterior_cannot_be_used_with_corrupt_sampler_evidence(runner, monkeypatch, change):
    blocks, path, directory = runner
    coordinate_registration(path)
    monkeypatch.setattr(inference, "sample", lambda b, s, c: result(s, True))
    fitting.run(blocks, {}, directory, path, study="toy", synthetic=True)
    archive = directory / "sampler_1.npz"
    if change == "bytes": archive.write_bytes(b"corrupt")
    elif change == "missing": archive.unlink()
    elif change == "diagnostic":
        receipt = json.loads((directory / "receipt.json").read_text())
        receipt["attempts"][0]["diagnostics"]["divergences"] = 1
        (directory / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises((ValueError, FileNotFoundError)):
        with fitting.checked_archive(directory):
            assert change == "replacement_during_read"
            archive.write_bytes(b"changed")


def test_failed_new_fit_retains_sampler_evidence_and_cannot_restart(runner, monkeypatch):
    blocks, path, directory = runner
    coordinate_registration(path)
    monkeypatch.setattr(inference, "sample", lambda b, s, c: result(s, False))
    with pytest.raises(RuntimeError, match="All registered"):
        fitting.run(blocks, {}, directory, path, study="toy", synthetic=True)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["state"] == "failed_convergence"
    fitting._check_sampler_extras(directory.resolve(), receipt)
    with pytest.raises(ValueError, match="Failed or incomplete"):
        fitting.run(blocks, {}, directory, path, study="toy", synthetic=True)


def test_schema_versions_do_not_silently_select_new_implementation(tmp_path):
    original = json.loads((ROOT / "docs/overall_profile_protocol.json").read_text())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "registration.json"
    def committed(reg):
        path.write_text(json.dumps(reg))
        subprocess.run(["git", "add", "registration.json"], cwd=tmp_path, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "commit", "-qm", "Synthetic registration"], cwd=tmp_path, check=True)
        return fitting.committed_registration(path, root=tmp_path)
    assert committed(original) == original
    invalid = dict(original, inference_method=inference.METHOD)
    with pytest.raises(ValueError, match="Historical"):
        committed(invalid)
    new = dict(invalid, schema_version=fitting.COORDINATE_RUN_SCHEMA, registered_model_inputs_key="a" * 64)
    assert committed(new) == new
    for key, value in (("inference_method", "foreign"), ("registered_model_inputs_key", ""), ("schema_version", True)):
        with pytest.raises(ValueError):
            committed(dict(new, **{key: value}))


@pytest.mark.parametrize("change", ["missing", "shape", "nan", "boolean_steps", "float_divergences", "negative_steps", "probability"])
def test_invalid_sampler_extras_are_rejected(change):
    settings = dict(chains=4, samples=128)
    extras = copy.deepcopy(result(settings, True).extras)
    if change == "missing": extras.pop("diverging")
    elif change == "shape": extras["num_steps"] = extras["num_steps"][:, :-1]
    elif change == "nan": extras["accept_prob"][0, 0] = np.nan
    elif change == "boolean_steps": extras["num_steps"] = extras["num_steps"].astype(bool)
    elif change == "float_divergences": extras["diverging"] = extras["diverging"].astype(float)
    elif change == "negative_steps": extras["num_steps"][0, 0] = -1
    elif change == "probability": extras["accept_prob"][0, 0] = 1.1
    with pytest.raises(ValueError):
        inference.validate_extras(extras, settings)
