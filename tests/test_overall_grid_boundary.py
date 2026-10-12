"""A known grid feature must not depend on an entrant's eventual result status."""
import numpy as np
import pytest

from f1rank.overall.design import build
from test_overall import tables


@pytest.mark.parametrize("later_status", ["Disqualified", "Did not start"])
def test_pit_lane_grid_feature_does_not_use_later_field_exclusions(later_status):
    source, clean = tables()
    event = "2022-02"
    pit_driver = (source["race"].event_id == event) & (source["race"].driver_id == "d0")
    source["race"].loc[pit_driver, "grid"] = 0
    before = build(event, tables=source, clean=clean)
    source["race"].loc[(source["race"].event_id == event) & (source["race"].driver_id == "d1"), "status"] = later_status
    after = build(event, tables=source, clean=clean)

    def grid_feature(design):
        entry = design.qualifying.entries
        index = int(entry.loc[(entry.event_id == event) & (entry.driver_id == "d0"), "entry_idx"].iloc[0])
        where = (design.outcomes["entry"] == index) & design.outcomes["mask"]
        return design.outcomes["log_grid"][where]

    # The excluded driver's result is not a new starting position for the pit-lane
    # driver. Match the complete declared field used at forecast time.
    np.testing.assert_array_equal(grid_feature(before), np.log([6]))
    np.testing.assert_array_equal(grid_feature(after), grid_feature(before))
