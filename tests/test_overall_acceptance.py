"""Independent numeric evidence from invented arrays; no real fit or acceptance."""
import copy
import functools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from f1rank.overall import acceptance, fitting, inference
from f1rank.overall.design import build
from test_overall import tables
from test_overall_production_simulation import commit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inventory_fixture():
    source, clean = tables()
    design = build("2022-02", tables=source, clean=clean)
    blocks = dict(qualifying=design.qualifying.arrays(), laps=design.laps, outcomes=design.outcomes)
    inventory = acceptance.retained_inventory(blocks)
    axes = (fitting.FIT_SETTINGS["chains"], fitting.FIT_SETTINGS["samples"])
    rng = np.random.default_rng(482006)
    # Deliberately artificial, independent traces. Shape/numerical tests do not
    # certify model support, a posterior distribution or calibration.
    post = {name: rng.normal(size=axes+tuple(spec["shape"])).astype(spec["dtype"])
            for name, spec in inventory.items()}
    return design, blocks, inventory, post


def test_prior_trace_inventory_matches_the_actual_numpyro_collector(inventory_fixture):
    import jax
    from numpyro import handlers
    from numpyro.infer.mcmc import _collect_and_postprocess
    from numpyro.infer.util import constrain_fn, unconstrain_fn
    from f1rank.overall.control_rotation import build_rotation
    from f1rank.overall.batched_rotation import prepare
    from f1rank.overall.weekend_coordinates import weekend_model
    _, blocks, expected, _ = inventory_fixture
    model = weekend_model(prepare(build_rotation(blocks["laps"])))
    args = tuple(blocks[name] for name in ("qualifying", "laps", "outcomes"))
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(0))).get_trace(*args)
    params = {name: site["value"] for name, site in trace.items()
              if site["type"] == "sample" and not site["is_observed"]}
    z = unconstrain_fn(model, args, {}, params)
    postprocess = functools.partial(constrain_fn, model, args, {}, return_deterministic=True)
    actual = _collect_and_postprocess(postprocess, ("z",), ())((SimpleNamespace(z=z),))
    assert expected == {name: dict(shape=list(value.shape), dtype=np.asarray(value).dtype.str)
                        for name, value in actual.items()}
    jax.clear_caches()


@pytest.fixture
def evidence(tmp_path, monkeypatch, inventory_fixture, request):
    import subprocess
    design, blocks, inventory, original = inventory_fixture
    mode = getattr(request, "param", "matching")
    post = {name: value.copy() for name, value in original.items()}
    if mode == "separated": post["skill"][-1] += 10
    if mode == "contrast":
        rng = np.random.default_rng(630)
        a = post["skill"]
        common = rng.normal(size=a.shape[:2]+(1,))
        offsets = np.arange(a.shape[0])[:, None, None] * np.arange(a.shape[-1])[None, None, :] * .0001
        post["skill"] = (common + offsets + rng.normal(size=a.shape) * .000001).astype(a.dtype)
    diagnostic = inference.check(post, 0)
    comparisons = inference.contrast_reports(post, design.catalog, 0)
    if mode == "matching":
        assert diagnostic["converged"] and comparisons["all_reported_contrasts_passed"]
    elif mode == "separated": assert diagnostic["converged"] is False
    elif mode == "contrast":
        assert diagnostic["converged"] is True and comparisons["all_reported_contrasts_passed"] is False
    diagnostic.update(comparison_reports=comparisons, inference_method=inference.METHOD,
                      mean_accept_prob=.75, tree_depth_limit_fraction=0.)
    # Emulate a recorded success claim. The evaluator must compute the actual
    # decision, including when this claim contradicts the retained arrays.
    diagnostic["converged"] = True
    if mode == "mismatch": diagnostic["parameters"][0]["bulk_ess_min"] += 1
    root = tmp_path/"registration"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    source = dict(key="a"*64, inputs={})
    reg = json.loads((ROOT/"docs/overall_profile_protocol.json").read_text())
    reg.update(schema_version=fitting.COORDINATE_RUN_SCHEMA, studies=["numeric_fixture"],
        inference_method=inference.METHOD, registered_model_inputs_key=source["key"])
    path = root/"fit.json"
    path.write_text(json.dumps(reg)); commit(root, path)
    monkeypatch.setattr(fitting, "committed_registration", functools.partial(fitting.committed_registration, root=root))
    monkeypatch.setattr(fitting, "snapshot", lambda: copy.deepcopy(source))
    monkeypatch.setattr(fitting, "require_clean_inputs", lambda _: None)
    def require(expected):
        if expected != source: raise ValueError("Synthetic source changed")
    monkeypatch.setattr(fitting, "require_snapshot", require)
    axes = (fitting.FIT_SETTINGS["chains"], fitting.FIT_SETTINGS["samples"])
    extras = dict(diverging=np.zeros(axes, bool), num_steps=np.full(axes,3,dtype=np.int32),
                  accept_prob=np.full(axes,.75,dtype=np.float32))
    monkeypatch.setattr(inference, "sample", lambda *_: inference.SampleResult(post, diagnostic, extras))
    run = tmp_path/"fit"
    _, receipt = fitting.run(blocks, design.catalog, run, path, study="numeric_fixture", synthetic=True)
    return SimpleNamespace(run=run, design=design, path=path, root=root, output=tmp_path/"evidence",
        post=post, inventory=inventory, receipt=receipt, maps=tmp_path/"maps")


def execute(case):
    case.maps.mkdir(exist_ok=True)
    return acceptance.evaluate_convergence(case.run, case.design, case.path, case.output,
        registration_root=case.root, temporary_parent=case.maps)


def test_matching_complete_numeric_evidence_passes_only_its_own_check(evidence):
    result, receipt = execute(evidence)
    assert result["passed"] is True and receipt["state"] == "complete"
    assert result["recorded_parameter_diagnostics_match"] and result["recorded_comparison_diagnostics_match"]
    assert result["sampler_extras_recounted"] and result["recorded_sampler_diagnostics_match"]
    assert result["model_inventory"] == evidence.inventory
    assert result["functional_simulation_precision_established"] is False
    assert result["scientific_acceptance_established"] is False and result["publication_allowed"] is False
    assert result["prior_attempt_parameter_diagnostics_recomputed"] is False
    assert not list(evidence.maps.iterdir())
    with pytest.raises(FileExistsError): execute(evidence)


@pytest.mark.parametrize("evidence", ["separated", "mismatch", "contrast"], indirect=True)
def test_recorded_success_does_not_override_recomputed_failures(evidence):
    result, receipt = execute(evidence)
    assert result["passed"] is False and receipt["state"] == "complete" and receipt["passed"] is False
    assert evidence.receipt["diagnostics"]["converged"] is True
    assert (evidence.output/"convergence.json").exists()


@pytest.mark.parametrize("change", ["omitted", "extra", "shape", "dtype"])
def test_self_consistent_but_wrong_receipt_inventory_cannot_hide_model_fields(evidence, change):
    post = dict(evidence.post)
    name = "overall_grid"
    if change == "omitted": post.pop(name)
    elif change == "extra": post["invented_extra"] = np.ones(post[name].shape, dtype=post[name].dtype)
    elif change == "shape": post[name] = post[name][..., :-1]
    else: post[name] = post[name].astype("float64")
    archive = evidence.run/"posterior.npz"
    np.savez_compressed(archive, **post)
    receipt = copy.deepcopy(evidence.receipt)
    receipt["posterior_sha256"] = acceptance.digest(archive)
    receipt["attempts"][-1]["posterior_sites"] = {n:dict(shape=list(v.shape),dtype=v.dtype.str) for n,v in post.items()}
    (evidence.run/"receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="inventory"):
        execute(evidence)
    saved = json.loads((evidence.output/"receipt.json").read_text())
    assert saved["state"] == "failed_exception" and saved["passed"] is False
    assert not (evidence.output/"convergence.json").exists()


@pytest.mark.parametrize("change", ["registration", "catalog", "archive"])
def test_input_change_during_diagnostics_cannot_complete(evidence, monkeypatch, change):
    checker = acceptance.check
    def mutate(post, divergences):
        result = checker(post, divergences)
        if change == "catalog": evidence.design.catalog["mutated"] = True
        else:
            path = evidence.path if change == "registration" else evidence.run/"posterior.npz"
            with path.open("ab") as handle: handle.write(b"\n")
        return result
    monkeypatch.setattr(acceptance, "check", mutate)
    with pytest.raises(ValueError): execute(evidence)
    saved = json.loads((evidence.output/"receipt.json").read_text())
    assert saved["state"] == "failed_exception" and saved["passed"] is False
    assert not (evidence.output/"convergence.json").exists()
