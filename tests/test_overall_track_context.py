"""Unknown neutralisation is not negative evidence; VSC ending is neutralised."""
import numpy as np
import pandas as pd
import pytest

from f1rank.overall.design import build
from test_overall import tables


def source_tables():
    source, clean = tables()
    source["race_track_status"] = pd.DataFrame({
        "event_id": source["events"].event_id, "time": 0., "status": "1"})
    return source, clean


@pytest.mark.parametrize("missing", ["absent_event", "null_only", "mixed_clear_unknown"])
def test_missing_track_status_cannot_enter_the_outcome_likelihood_as_green(missing):
    source, clean = source_tables()
    status = source["race_track_status"]
    if missing == "absent_event":
        source["race_track_status"] = status[status.event_id != "2022-02"]
    else:
        status.loc[status.event_id == "2022-02", "status"] = None
        if missing == "mixed_clear_unknown":
            source["race_track_status"] = pd.concat([status, pd.DataFrame([
                {"event_id": "2022-02", "time": 5., "status": "1"}])], ignore_index=True)
    design = build("2022-02", tables=source, clean=clean)
    assert "2022-02" in design.catalog["excluded_context_events"]
    assert "2022-02" not in design.catalog["outcome_events"]
    risk_events = design.qualifying.entries.iloc[design.outcomes["risk_entry"]].event_id
    assert "2022-02" not in set(risk_events)


def test_vsc_ending_is_neutralisation_even_without_a_derived_timeline_row():
    source, clean = source_tables()
    source["race_track_status"].loc[lambda x: x.event_id == "2022-02", "status"] = "7"
    design = build("2022-02", tables=source, clean=clean)
    index = design.catalog["outcome_events"].index("2022-02")
    assert design.outcomes["context_sc"][design.outcomes["context"][index]] == 1


def test_observed_clear_and_future_status_data_remain_separate():
    source, clean = source_tables()
    source["race_track_status"].loc[lambda x: x.event_id > "2022-02", "status"] = "unrecognised_future_code"
    design = build("2022-02", tables=source, clean=clean)
    assert design.catalog["excluded_context_events"] == []
    assert not design.outcomes["context_sc"][design.outcomes["context"]].any()
