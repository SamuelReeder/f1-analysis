"""Guard the distinction between a successful replay and a changed decision."""
import pytest

from analysis.championship_diagnosis import assert_same


def test_replay_rejects_changed_gate_even_when_numeric_values_match():
    with pytest.raises(AssertionError):
        assert_same({"gate": True, "mean": 0.1}, {"gate": False, "mean": 0.1})
    with pytest.raises(AssertionError):
        assert_same({"gate": 0}, {"gate": False})


def test_replay_rejects_missing_evidence_or_material_score_changes():
    assert_same({"ci": [-0.03, 0.293000001]}, {"ci": [-0.03, 0.293]})
    with pytest.raises(AssertionError):
        assert_same({"ci": [-0.01, 0.293]}, {"ci": [-0.03, 0.293]})
    with pytest.raises(AssertionError):
        assert_same({"ci": [-0.03, 0.293]}, {"ci": [-0.03, 0.293], "gate": False})
