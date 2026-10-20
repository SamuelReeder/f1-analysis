"""Canonical export statistics: paired arithmetic, identity, inventory and bounds.

Passing these software checks supplies no scientific acceptance or real ranking.
"""
import copy
import json
from pathlib import Path

import pytest

from f1rank.overall.publication import CONTRACT, check_payload

FIXTURE = Path(__file__).parent / "fixtures/overall_current_contract.json"


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text())


def test_shared_browser_fixture_is_structurally_valid_without_becoming_scientific_evidence(payload):
    before = copy.deepcopy(payload)
    assert check_payload(payload) is payload
    assert payload == before
    # This fixture is explicitly invented and is never passed to a writer.
    assert "Synthetic fixture only" in payload["limitations"][0]


@pytest.mark.parametrize("key", CONTRACT["required_checks"])
@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_every_recorded_check_must_be_exactly_true(payload, key, value):
    payload["evidence"]["checks"][key]["passed"] = value
    with pytest.raises(ValueError, match="acceptance check"):
        check_payload(payload)


@pytest.mark.parametrize("path,value", [
    (("schema_version",), True),
    (("status",), "not established"),
    (("target",), "current_team"),
    (("evidence", "posterior_key"), "missing"),
    (("evidence", "posterior_draws"), True),
    (("evidence", "replicates_per_draw"), 200),
    (("standings", 0, "expected_points", "mean"), True),
    (("standings", 0, "expected_points", "mean"), float("nan")),
    (("standings", 0, "expected_points", "q05"), 25),
    (("standings", 0, "expected_points", "posterior_sd"), -1),
    (("standings", 0, "expected_points", "conditional_precision_passed"), False),
    (("standings", 0, "expected_points", "conditional_mcse_bound"), 0),
    (("standings", 0, "p_title"), .9),
    (("standings", 0, "season_rank", "median"), 3),
    (("standings", 0, "driver_id"), "alpha"),
    (("contributions", 0, "posterior_key"), "d" * 64),
    (("contributions", 0, "simulation_key"), "d" * 64),
    (("contributions", 0, "losses", "all"), 1.7),
    (("contributions", 0, "losses", "all", "conditional_precision_passed"), False),
    (("contributions", 0, "losses", "all", "max_conditional_mcse"), .1),
    (("pairwise",), []),
    (("pairwise", 0, "driver_b"), "alpha"),
    (("pairwise", 0, "driver_b"), "outsider"),
    (("pairwise", 0, "scenario_id"), "other"),
    (("pairwise", 0, "simulation_key"), "d" * 64),
    (("pairwise", 0, "points_difference", "mean"), 3),
    (("pairwise", 0, "points_difference", "conditional_mcse_bound"), 0),
])
def test_incomplete_mixed_or_inconsistent_statistics_are_rejected(payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        check_payload(payload)


def test_all_driver_pairs_are_required_and_reversed_duplicates_are_not_extra_pairs(payload):
    e = payload["evidence"]
    e["field"].append("gamma")
    row = copy.deepcopy(payload["standings"][0])
    row.update(driver_id="gamma", name="Fixture Gamma", p_title=0.)
    row["expected_points"].update(mean=10., q05=9.9, q95=10.1)
    payload["standings"].append(row)
    explanation = copy.deepcopy(payload["contributions"][0])
    explanation["driver_id"] = "gamma"
    payload["contributions"].append(explanation)
    for first, mean in (("alpha", 12.5), ("beta", 8.5)):
        pair = copy.deepcopy(payload["pairwise"][0])
        pair.update(driver_a=first, driver_b="gamma")
        pair["points_difference"].update(mean=mean, q05=mean - .1, q95=mean + .1)
        payload["pairwise"].append(pair)
    check_payload(payload)
    payload["pairwise"][2] = copy.deepcopy(payload["pairwise"][1])
    pair = payload["pairwise"][2]
    pair["driver_a"], pair["driver_b"] = pair["driver_b"], pair["driver_a"]
    points = pair["points_difference"]
    points.update(mean=-points["mean"], q05=-points["q95"], q95=-points["q05"])
    with pytest.raises(ValueError, match="Duplicate"):
        check_payload(payload)


def test_explanation_inventory_cannot_add_or_drop_unregistered_components(payload):
    payload["contributions"][0]["losses"]["unregistered"] = payload["contributions"][0]["losses"].pop("all")
    with pytest.raises(ValueError, match="components"):
        check_payload(payload)

