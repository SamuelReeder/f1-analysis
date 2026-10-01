"""Lightweight, fail-closed reader for the portable race-pace export (no JAX)."""
import json
from pathlib import Path
import math

from .artifacts import require, require_convergence


def check(data):
    if data.get("schema_version") != 1 or data.get("model") != "total-dry-race-pace-v1":
        raise ValueError("Unsupported race pace export")
    diag = data["diagnostics"]
    require_convergence(diag)
    if not (math.isfinite(diag["rhat_max"]) and diag["rhat_max"] < 1.05
            and diag["n_draws"] >= 1600 and 0 <= diag["divergences"] <= diag["n_draws"] / 1000):
        raise ValueError("Race pace fit failed publication diagnostics")
    validation = data["validation"]
    if validation.get("folds") != ["2024-10", "2025-10", "2026-07"]:
        raise ValueError("Race pace validation has incomplete test windows")
    for end in validation["folds"]:
        for mode in ("full", "no_driver", "no_car"):
            fit = validation["fits"][end][mode]
            d = fit["diagnostics"]
            require_convergence(d)
            if (fit["data_as_of"] > end or d["rhat_max"] >= 1.05 or not math.isfinite(d["rhat_max"])
                    or d["n_draws"] < 1600 or not 0 <= d["divergences"] <= d["n_draws"] / 1000):
                raise ValueError("Invalid race pace validation fit")
    for kind in ("drivers", "cars"):
        gate = data["validation"]["metrics"][kind]
        passed = (gate["n_races"] >= 12 and gate["mse_difference_ci95"][1] < 0
                  and .85 <= gate["coverage90"] <= .95
                  and gate["mean_interval_width"] < gate["baseline_interval_width"])
        if bool(gate["passed"]) != passed:
            raise ValueError("Race pace gate disagrees with its evidence")
        if data[kind] and not passed:
            raise ValueError("Unsupported race pace ranking")
        for row in data[kind]:
            e = row["pace"]
            if not all(math.isfinite(e[q]) for q in ("q05", "median", "q95")):
                raise ValueError("Nonfinite race pace")
            if not (e["q05"] <= e["median"] <= e["q95"] and 1 <= e["rank_lo"] <= e["rank_hi"] <= len(data[kind])):
                raise ValueError("Invalid race pace intervals")
    return data


def load(root: Path):
    directory = root / "outputs/race_total"
    if not directory.exists() or not (directory / "pace.json").exists():
        return None
    require(directory, required_outputs=[directory / "pace.json"])
    return check(json.loads((directory / "pace.json").read_text()))
