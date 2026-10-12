"""Verify reuse of the published qualifying fit without refitting or changing gates.

This audit supplements the legacy rounded input fingerprint with exact original
data, probability-model source, sampling-function and design-array comparisons.
It does not accept an Overall posterior or refresh any publication manifest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from f1rank.artifacts import atomic_json, digest
from f1rank.design import build_design
from f1rank.fit import check, design_ids, fingerprint


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(path: Path) -> dict:
    path = path.resolve()
    sidecar = path.with_suffix(".meta.json")
    initial = {"archive": digest(path), "metadata": digest(sidecar)}
    metadata = json.loads(sidecar.read_text())
    require(metadata == json.loads((ROOT / "outputs/ratings/fit_metadata.json").read_text()),
            "Fit identity differs from the published qualifying fit")
    require(metadata.get("git_dirty") is False, "Original fitting source was not clean")
    require(metadata.get("last_train_event") is None, "Expected a full-history fit")
    require(not metadata.get("sprint_quali_until"), "Expected the published qualifying model")
    require(metadata.get("model_kw") == {}, "Expected the published model defaults")
    revision = metadata["git_commit"]

    def historical(name):
        return subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)

    protected = ["f1rank/model.py", "f1rank/lineage.py"]
    protected += [f"data/processed/{name}.parquet"
                  for name in ("events", "entries", "drivers", "quali_times")]
    hashes = {}
    for name in protected:
        original, current = historical(name), (ROOT / name).read_bytes()
        require(original == current, f"Original fit input/source changed: {name}")
        hashes[name] = hashlib.sha256(current).hexdigest()

    old_fit = ast.parse(historical("f1rank/fit.py"))
    current_fit = ast.parse((ROOT / "f1rank/fit.py").read_text())
    functions = ("to_jax", "initial_values", "fit", "fingerprint", "design_ids")
    for name in functions:
        def definition(tree):
            return next(ast.dump(node, include_attributes=False) for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        require(definition(old_fit) == definition(current_fit),
                f"Original sampling/input function changed: {name}")

    module = types.ModuleType("f1rank._published_fit_design_audit")
    module.__file__ = str(ROOT / "f1rank/design.py")
    sys.modules[module.__name__] = module
    try:
        exec(compile(historical("f1rank/design.py"), module.__file__, "exec"), module.__dict__)
        original = module.build_design(metadata["start_season"])
        current = build_design(metadata["start_season"])
    finally:
        del sys.modules[module.__name__]
    before, after = original.arrays(), current.arrays()
    require(before.keys() == after.keys(), "Design array inventory changed")
    arrays = {}
    for name in before:
        left, right = np.asarray(before[name]), np.asarray(after[name])
        require(left.dtype == right.dtype and left.shape == right.shape
                and left.tobytes() == right.tobytes(), f"Design array changed: {name}")
        arrays[name] = {"dtype": str(right.dtype), "shape": list(right.shape),
                        "sha256": hashlib.sha256(right.tobytes()).hexdigest()}
    require(design_ids(original) == design_ids(current) == metadata["ids"],
            "State identity/order changed")
    require(fingerprint(original) == fingerprint(current) == metadata["fingerprint"],
            "Fit fingerprint mismatch")
    require(check(path, current) == metadata, "Normal fit compatibility check failed")
    require(initial == {"archive": digest(path), "metadata": digest(sidecar)},
            "Fit changed during audit")
    return {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "audit_source_sha256": digest(Path(__file__)),
        "fit_revision": revision, "fit_identity": metadata["created_utc"],
        "fit_fingerprint": metadata["fingerprint"], "fit_hashes": initial,
        "original_source_and_data_sha256": hashes,
        "unchanged_sampling_functions": list(functions),
        "original_design_sha256": hashlib.sha256(historical("f1rank/design.py")).hexdigest(),
        "current_design_sha256": digest(ROOT / "f1rank/design.py"),
        "exact_model_arrays": arrays, "state_identity_equal": True,
        "compatible_for_qualifying_reexport": True,
        "requires_export_convergence_check": True,
        "accepts_overall_model": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.fit)
    atomic_json(args.output, result)
    print("Published qualifying fit is compatible for checked re-export.")
