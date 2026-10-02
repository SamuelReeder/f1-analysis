"""The pre-registered weekend covariates (docs/race_features.md) on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from f1rank import race_features as rf


def practice_laps(stints):
    """stints: (driver, team, compound, lap times); laps get a green status and no pit."""
    rows = []
    for k, (driver, team, compound, times) in enumerate(stints):
        for n, t in enumerate(times):
            rows.append(dict(event_id="2025-01", driver_id=driver, team=team, stint=float(k), compound=compound,
                             lap_time=t, pit_in_time=np.nan, pit_out_time=np.nan, deleted=False,
                             is_accurate=True, track_status="1", lap_number=n, speed_st=300.0))
    return pd.DataFrame(rows)


def test_longrun_uses_clean_runs_of_five_laps_relative_to_the_compound_median():
    base = [90.0] * 6
    laps = practice_laps([
        ("a1", "A", "MEDIUM", [89.0] * 5 + [100.0]),   # a slow cool-down lap (>7%) is dropped
        ("a2", "A", "MEDIUM", [90.0] * 6),
        ("b1", "B", "MEDIUM", [91.0] * 6),
        ("b2", "B", "MEDIUM", [91.0] * 4),             # too short: not a run
        ("c1", "C", "MEDIUM", base),
        ("c2", "C", "HARD", base),                      # only one driver on HARD: not used
    ])
    x_car, x_drv = rf.longrun(laps)
    rel = lambda t: 100 * (90.0 - t) / 90.0  # noqa: E731 - median of the MEDIUM runs is 90.0
    team = {"A": (rel(89.0) + rel(90.0)) / 2, "B": rel(91.0), "C": rel(90.0)}
    mean = np.mean(list(team.values()))
    assert x_car.longrun.to_dict() == pytest.approx({f"2025-01|{t}": v - mean for t, v in team.items()})
    # the teammate split exists only where both teammates have runs
    assert x_drv.longrun_split.to_dict() == pytest.approx(
        {"2025-01|a1": rel(89.0) - team["A"], "2025-01|a2": rel(90.0) - team["A"]})


def test_longrun_ignores_pit_laps_and_neutralised_laps():
    laps = practice_laps([(d, t, "SOFT", [90.0] * 6) for d, t in (("a1", "A"), ("b1", "B"), ("c1", "C"))])
    slow = laps.driver_id == "a1"
    laps.loc[slow & (laps.lap_number == 0), "pit_out_time"] = 1.0
    laps.loc[slow & (laps.lap_number == 1), "track_status"] = "12"
    x_car, _ = rf.longrun(laps)
    assert "2025-01|A" not in x_car.index  # 4 clean laps left: no run


def test_traps_are_the_teams_90th_percentile_against_the_median_team():
    laps = pd.DataFrame(dict(event_id="2025-01", team=["A"] * 10 + ["B"] * 10 + ["C"] * 10,
                             lap_time=90.0, pit_in_time=np.nan, pit_out_time=np.nan,
                             speed_st=np.r_[np.arange(300, 310), np.arange(310, 320), np.arange(290, 300)]))
    x = rf.traps(laps).traps
    q = {t: np.quantile(v, .9) for t, v in (("A", np.arange(300, 310)), ("B", np.arange(310, 320)),
                                            ("C", np.arange(290, 300)))}
    assert x.to_dict() == pytest.approx({f"2025-01|{t}": (v - q["A"]) / 10 for t, v in q.items()})


def test_upgrades_accumulate_within_a_season_and_are_centred():
    entries = pd.DataFrame(dict(event_id=["2024-24"] * 2 + ["2025-01"] * 2 + ["2025-02"] * 2,
                                team=["A", "B"] * 3))
    supplement = {"events": {"2024-24": {"teams": {"A": {"performance": 10}}},
                             "2025-02": {"teams": {"A": {"performance": 3}, "B": {"performance": 1}}}}}
    x = rf.upgrades(entries, supplement).upgrades.to_dict()
    assert x == pytest.approx({"2024-24|A": .5, "2024-24|B": -.5,       # 10 vs 0
                               "2025-01|A": 0., "2025-01|B": 0.,        # a new season starts at 0
                               "2025-02|A": .1, "2025-02|B": -.1})      # 3 vs 1


def fia():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "extract" / "fia_upgrades.py"
    spec = importlib.util.spec_from_file_location("fia_upgrades", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# pdfplumber's rows of the 2024 Chinese GP Haas table (descriptions shortened): '' is a cell
# boundary, None the continuation of a merged cell, the number centred in its row's cells
CHINA_2024_HAAS = [
    ["", "Updated", "Primary reason for", "", ""],
    ["", None, None, "Geometric differences", "Brief description"],
    [None, "component", "update", None, None],
    ["", None, None, "", ""],
    ["", "", "", "", "x"],
    [None, None, "Performance - Flow", None, None],
    ["1", "Floor Fences", None, "x", "x"],
    [None, None, "Conditioning", None, None],
    ["", "", None, "", "x"],
    [None, None, "", None, None],
    ["", "", "", "", "x"],
    [None, None, "Performance - Local", None, None],
    ["2", "Floor Edge", None, "x", "x"],
    [None, None, "Load", None, None],
    ["", "", None, "", "x"],
    [None, None, "", None, None],
    ["", "", "", "", "x"],
    [None, "Coke/Engine", "Performance - Drag", None, None],
    ["3", None, None, "x", "x"],
    [None, "Cover", "reduction", None, None],
    ["", None, None, "", "x"],
    [None, "", "", None, None],
]


def test_fia_tables_read_wrapped_reasons_and_names():
    assert fia().components(CHINA_2024_HAAS) == [
        {"component": "Floor Fences", "reason": "Performance - Flow Conditioning"},
        {"component": "Floor Edge", "reason": "Performance - Local Load"},
        {"component": "Coke/Engine Cover", "reason": "Performance - Drag reduction"},
    ]


def test_fia_tables_apply_a_merged_reason_and_skip_blank_rows_and_fragments():
    f = fia()
    head = [["", "", "Updated", "", "", "Primary reason", ""],
            [None, None, "component", None, None, "for update", None]]
    table = head + [
        ["1", "Front Wing", None, None, "Performance - Local Load", None, None],
        [None, None, None, None, None, None, None],
        ["2", "Nose", None, None, None, None, None],                      # the reason cell is merged
        [None, None, None, None, None, None, None],
        ["3", "Cooling Louvres", None, None, "Circuit specific", None, None],  # reads like a reason
        [None, None, None, None, None, None, None],
        ["4", "", None, None, "", None, None],                             # a blank template row
    ]
    assert f.components(table) == [
        {"component": "Front Wing", "reason": "Performance - Local Load"},
        {"component": "Nose", "reason": "Performance - Local Load"},
        {"component": "Cooling Louvres", "reason": "Circuit specific"},
    ]
    assert f.components([["Performance -"], ["Local Load"]]) == []  # a text fragment, not a table


def test_fia_counts_performance_subcategories_written_without_the_prefix():
    f = fia()
    reasons = ["Performance - Drag reduction", "Local load", "Flow conditioning", "Drag Reduction",
               "Circuit specific - Drag Range", "Reliability", "Cooling Range"]
    assert [bool(f.PERFORMANCE.match(r)) for r in reasons] == [True, True, True, True, False, False, False]
    assert f.team_of("Visa Cash App RB F1 Team") == "faenza" and f.team_of("Oracle Red Bull Racing") == "red_bull"
