"""Battle episodes and on-track passes, 2018 onward (stage 2, step 7: overtaking and defending).

    python -m f1rank.battles episodes      # build episodes (outputs/battles/episodes.parquet)
    python -m f1rank.battles feasibility   # synthetic check at real sample sizes
    python -m f1rank.battles fit           # model + held-out test (only if feasible)

Episodes: on each lap, cars that completed the same lap are ordered by the time they crossed
the line. A battle episode starts when a car finishes a lap within 1.0 s of the car ahead,
with both on green-flag laps and neither pitting. It continues while they stay within 2.0 s,
and ends with a pass (the car behind crosses the line first on the next lap, neither
pitting), the gap growing beyond 2 s, a pit stop, a neutralisation, or the end of the race.
Teammates are excluded (team orders), as are lap 1 and the laps after a neutralisation.

Each episode lap is one observation: did the attacker pass on the next lap? Covariates:
the race-pace difference (stage A of racepace.py; attacker minus defender), the tyre-age
difference, a softer-compound indicator, the gap, and the straight-line speed difference
(race medians of the speed trap). The model adds a circuit effect and attacker and
defender effects, plus a pair effect for repeated laps against the same car.

docs/racing_approach.md asks for a synthetic feasibility check first: at the real number
of episodes per driver, can attacker and defender effects of plausible size be recovered?
If not, overtaking is reported only as opportunity counts (episodes and passes), not rated.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "battles"
START_S, KEEP_S = 1.0, 2.0
SOFTNESS = {"HYPERSOFT": 5, "ULTRASOFT": 4, "SUPERSOFT": 3, "SOFT": 2, "MEDIUM": 1, "HARD": 0, "SUPERHARD": -1}


def race_battles(L: pd.DataFrame, neutral_laps: set, pace: pd.Series) -> pd.DataFrame:
    """Episode laps for one race: (attacker, defender, lap, gap, passed next lap, covariates)."""
    L = L[L.driver_id.notna() & L.time.notna()].copy()
    L["pit"] = L.pit_in_time.notna() | L.pit_out_time.notna()
    L["green"] = L.track_status.astype(str) == "1"
    L = L.sort_values(["lap_number", "time"])
    L["ahead"] = L.groupby("lap_number").driver_id.shift()
    L["ahead_team"] = L.groupby("lap_number").team.shift()
    L["gap"] = L.groupby("lap_number").time.diff()
    speed = L.groupby("driver_id").speed_st.median()
    by = L.set_index(["lap_number", "driver_id"])
    order = {lap: dict(zip(g.driver_id, range(len(g)))) for lap, g in L.groupby("lap_number")}
    rows = []
    active = {}  # (attacker, defender) -> episode id
    next_id = 0
    for lap in sorted(order)[1:]:
        cur = L[L.lap_number == lap]
        nxt = order.get(lap + 1)
        seen = set()
        for r in cur.itertuples():
            if pd.isna(r.ahead) or r.team == r.ahead_team or pd.isna(r.gap):
                continue
            key = (r.driver_id, r.ahead)
            d = by.loc[(lap, r.ahead)] if (lap, r.ahead) in by.index else None
            ok = (r.green and not r.pit and d is not None and bool(d.green) and not bool(d.pit)
                  and lap not in neutral_laps and nxt is not None and r.driver_id in nxt and r.ahead in nxt)
            if not ok or r.gap > (KEEP_S if key in active else START_S):
                continue
            nxt_rows = [(lap + 1, x) for x in key]
            if any(k not in by.index or bool(by.loc[k].pit) or not bool(by.loc[k].green) for k in nxt_rows):
                continue  # a pit stop or neutralisation next lap ends the episode without an outcome
            if key not in active:
                active[key] = next_id
                next_id += 1
            seen.add(key)
            passed = nxt[r.driver_id] < nxt[r.ahead]
            rows.append({"episode": active[key], "attacker": r.driver_id, "defender": r.ahead, "lap": lap,
                         "gap": r.gap, "passed": passed,
                         "pace_diff": pace.get(r.driver_id, np.nan) - pace.get(r.ahead, np.nan),
                         "tyre_age_diff": r.tyre_life - d.tyre_life,
                         "softer": SOFTNESS.get(r.compound, 0) - SOFTNESS.get(d.compound, 0),
                         "speed_diff": speed.get(r.driver_id, np.nan) - speed.get(r.ahead, np.nan)})
            if passed:
                seen.discard(key)
        active = {k: v for k, v in active.items() if k in seen}
    return pd.DataFrame(rows)


def episodes() -> pd.DataFrame:
    laps = pd.read_parquet(PROCESSED / "race_laps.parquet")
    T = pd.read_parquet(PROCESSED / "timeline.parquet")
    pace = pd.read_csv(ROOT / "outputs" / "race" / "stage_a_drivers.csv")
    out = []
    for event_id, L in laps.groupby("event_id"):
        tl = T[(T.event_id == event_id) & T.kind.isin(["safety_car", "vsc", "red_flag"])]
        neutral = set()
        for r in tl.itertuples():
            end = r.lap_end if pd.notna(r.lap_end) else r.lap_start
            neutral |= set(range(int(r.lap_start), int(end) + 3))  # and 2 laps after it ends
        p = pace[pace.event_id == event_id].set_index("driver_id").pace
        b = race_battles(L, neutral | {1}, p)
        if len(b):
            b.insert(0, "event_id", event_id)
            b["episode"] = event_id + "#" + b.episode.astype(str)
            out.append(b)
        print(event_id, len(b), flush=True)
    E = pd.concat(out, ignore_index=True)
    E["season"] = E.event_id.str[:4].astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    E.to_parquet(OUT / "episodes.parquet", index=False)
    return E


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["episodes", "feasibility", "fit"])
    args = p.parse_args()
    if args.step == "episodes":
        E = episodes()
        print(len(E), "episode laps;", E.episode.nunique(), "episodes;", int(E.passed.sum()), "passes")
    elif args.step == "feasibility":
        from .overtaking import feasibility
        feasibility()
    else:
        from .overtaking import fit_and_test
        fit_and_test()


if __name__ == "__main__":
    main()
