"""Race-only evidence boundaries and the inherited joint state prior."""
import numpy as np

from f1rank.overall.design import build
from test_overall import tables


def test_race_evidence_remains_usable_without_qualifying_observations():
    source, clean = tables()
    source["quali_times"] = source["quali_times"].iloc[:0].copy()
    design = build("2022-02", tables=source, clean=clean)
    q = design.qualifying.arrays()
    assert q["n_sessions"] == q["n_car_segments"] == 0
    assert q["y"].size == q["train"].size == 0
    assert len(design.laps["y"]) > 0 and design.outcomes["mask"].any()
    assert set(design.catalog["field"]) == {f"d{i}" for i in range(6)}
    np.testing.assert_array_equal(q["circuit_factor"], np.zeros(q["n_circuits"]))


def test_diagnostics_record_empty_sites_without_changing_existing_decisions():
    from f1rank.overall.diagnostics import check, MIN_CHAINS, MIN_ESS
    draws = np.random.default_rng(0).normal(size=(MIN_CHAINS, MIN_ESS))
    original = check({"parameter": draws}, 0)
    extended = check({"parameter": draws, "mu": np.empty(draws.shape + (0,))}, 0)
    assert extended.pop("empty_sites") == [dict(parameter="mu", shape=list(draws.shape + (0,)))]
    original.pop("empty_sites", None)
    assert extended == original
