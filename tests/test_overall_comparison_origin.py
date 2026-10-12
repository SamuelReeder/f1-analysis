"""Same origin labels do not establish identical information or scoring targets."""
import copy

import pytest

from analysis import overall_predictive_scores as scores
from test_overall_predictive_scores import inventory_and_records


@pytest.mark.parametrize("field,changed", [
    ("origin_at", "2001-03-01T13:00:00Z"),
    ("snapshot_sha256", "a" * 64),
    ("training_history_sha256", "b" * 64),
    ("target_definition_sha256", "c" * 64),
])
def test_different_information_or_conditioning_rule_cannot_pass(field, changed):
    inventory, rows = inventory_and_records()
    rows[0][field] = changed
    result = scores.compare_arms(rows, inventory)
    assert not result["complete"]
    assert not result["comparison_rule_passed"]
    assert not result["comparisons"]
    assert inventory[0]["event_id"] in result["required_events"]


@pytest.mark.parametrize("field", ["origin_at", "snapshot_sha256", "training_history_sha256", "target_definition_sha256"])
def test_missing_record_identity_remains_a_required_unresolved_result(field):
    inventory, rows = inventory_and_records()
    rows[0].pop(field)
    result = scores.compare_arms(rows, inventory)
    assert not result["complete"] and not result["comparison_rule_passed"]


@pytest.mark.parametrize("field,bad", [
    ("origin_at", "2001-03-01T12:00:00"),
    ("origin_at", "not a date"),
    ("snapshot_sha256", ""),
    ("training_history_sha256", "not a content digest"),
    ("target_definition_sha256", None),
])
def test_invalid_common_inventory_cannot_establish_comparability(field, bad):
    inventory, rows = inventory_and_records()
    inventory[0][field] = bad
    with pytest.raises(ValueError):
        scores.compare_arms(rows, inventory)


def test_timezone_equivalence_is_not_a_different_information_origin():
    inventory, rows = inventory_and_records()
    expected = scores.compare_arms(rows, inventory)
    rows[0]["origin_at"] = "2001-03-01T07:00:00-05:00"
    actual = scores.compare_arms(rows, inventory)
    assert actual == expected


def test_every_arm_has_to_match_the_same_registered_definition():
    inventory, rows = inventory_and_records()
    for row in rows:
        row["target_definition_sha256"] = "d" * 64
    result = scores.compare_arms(rows, inventory)
    assert not result["complete"] and not result["comparison_rule_passed"]


def test_receipt_binds_comparison_inventory_without_certifying_claimed_provenance():
    inventory, rows = inventory_and_records()
    result = scores.compare_arms(rows, inventory)
    assert result["complete"] and result["comparison_rule_passed"]
    assert not result["publication_allowed"]
    assert not result["source_availability_verified"]
    changed, changed_rows = copy.deepcopy(inventory), copy.deepcopy(rows)
    changed[0]["snapshot_sha256"] = "e" * 64
    for row in changed_rows:
        if row["event_id"] == changed[0]["event_id"]:
            row["snapshot_sha256"] = "e" * 64
    other = scores.compare_arms(changed_rows, changed)
    assert result["comparisons"] == other["comparisons"]
    assert result["comparison_inventory_sha256"] != other["comparison_inventory_sha256"]
