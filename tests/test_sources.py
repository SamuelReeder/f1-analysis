"""Stage-2 data sources on small hand-made inputs: Jolpica lap data, the power-unit table."""

import numpy as np
import pandas as pd

from f1rank import oldlaps, powerunits


def test_lap_times_parse():
    assert oldlaps.seconds("1:32.456") == 92.456
    assert oldlaps.seconds("58.1") == 58.1
    assert oldlaps.seconds("1:02:03.5") == 3723.5


def field(times_by_lap, drivers=("a", "b", "c", "d")):
    rows = [(d, lap, t + 0.1 * i) for lap, t in times_by_lap.items() for i, d in enumerate(drivers)]
    return pd.DataFrame(rows, columns=["driver_id", "lap_number", "lap_time"])


def test_neutralised_laps_are_the_slow_field_laps_and_the_next_one():
    times = {lap: 90.0 for lap in range(1, 21)}
    times[8] = times[9] = 130.0  # safety car
    assert oldlaps.neutral_laps(field(times)) == {8, 9, 10}


def test_pit_stop_is_the_in_lap_of_the_slowest_pair():
    L = field({lap: 90.0 for lap in range(1, 21)})
    m = (L.driver_id == "b") & L.lap_number.isin([11, 12, 13])
    L.loc[m, "lap_time"] += np.array([4.0, 6.0, 18.0])  # slow in-lap 12, out-lap 13
    P = oldlaps.infer_pits(L, set())
    assert list(zip(P.driver_id, P.lap)) == [("b", 12)]


def test_table_grid_expands_rowspan_and_colspan():
    html = ("<table><tr><th>Entrant</th><th>Constructor</th><th colspan=2>Drivers</th></tr>"
            "<tr><td rowspan=2>Team X</td><td rowspan=2>X-Ferrari</td><td>1</td><td>A</td></tr>"
            "<tr><td>2</td><td>B<sup>[1]</sup></td></tr></table>")
    p = powerunits.Tables()
    p.feed(html)
    assert p.tables[0] == [["Entrant", "Constructor", "Drivers", "Drivers"],
                           ["Team X", "X-Ferrari", "1", "A"], ["Team X", "X-Ferrari", "2", "B"]]


def test_lineage_from_the_chassis_part():
    assert powerunits.team_of("McLaren-Mercedes", 2014) == "mclaren"
    assert powerunits.team_of("Mercedes", 2014) == "brackley"
    assert powerunits.team_of("Racing Bulls-Red Bull Ford", 2026) == "faenza"
    assert powerunits.team_of("Red Bull Racing-TAG Heuer", 2016) == "red_bull"
    assert powerunits.team_of("Lotus-Renault", 2011) == "leafield"   # Team Lotus
    assert powerunits.team_of("Lotus-Renault", 2013) == "enstone"    # the former Renault team
    assert powerunits.match("Red Bull RBPTH001", powerunits.SUPPLIERS) == "Honda"
    assert powerunits.match("TAG Heuer F1-2016", powerunits.SUPPLIERS) == "Renault"
