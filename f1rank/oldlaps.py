"""Lap-by-lap data from Jolpica, for the races before FastF1's timing (2010-2017).

    python -m f1rank.oldlaps           # build data/processed/jolpica_{laps,pitstops}.parquet
    python -m f1rank.oldlaps check     # check the inferences against FastF1 (2018-2019)
    python -m f1rank.oldlaps dump      # check the Jolpica database dump against the API pages
    python -m f1rank.oldlaps download  # fetch the newest delayed CSV dump (sha256 checked)

Sources. Lap-by-lap pages from the API (fetch.py --laps, data/raw/jolpica/laps/) and, for
races the API pages do not cover, Jolpica's CSV database dump (the free "delayed" dump,
data/raw/jolpica/dumps/, its sha256 checked against the value Jolpica publishes). `dump`
compares the two on every race both have; the dump is used only after that check.

Jolpica gives each car's lap time and position on every lap, and pit stops from 2011. It has no tyre compounds, track status or speed traps, so docs/racing_approach.md
treats these races through a weaker observation model. This module builds the tables and
the flags the racing models need, inferring what Jolpica lacks:

- Crossing time: the sum of a car's lap times (race time at the end of each lap), so the
  gap to the car ahead at the line is a difference of sums, as with FastF1's session time.
- Neutralised laps (safety car, VSC, red flag): laps whose field median lap time is more
  than NEUTRAL_SLOW times the race's median lap; the laps after one, until the field is back
  to speed, are also flagged, as are the next NEUTRAL_AFTER laps.
- Pit stops in 2010 (not recorded): a lap and the next one together at least PIT_LOSS_S slower than
  the car's median green lap, when the field was not neutralised; of consecutive such laps,
  the in-lap starts the slowest pair. Stops under a neutralisation lose too little time to
  be seen and are missed (their laps are excluded as neutralised anyway, but the car's tyre
  age is then not reset).
- Tyre age: laps since the car's last pit stop (or since the start); compound unknown.

`check` compares each inference with FastF1 on the races both sources have (2018 on: lap
positions, neutralised laps, pit laps, gaps at the line) and, for 2011-2017, the inferred
pit laps against Jolpica's recorded stops; it writes
outputs/analysis/oldlaps_check.json. Those rates are what the downstream models' weaker
observation model for these seasons is judged against.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "jolpica" / "laps"
PROCESSED = ROOT / "data" / "processed"
NEUTRAL_SLOW = 1.12
NEUTRAL_AFTER = 1
PIT_LOSS_S = 12.0


def seconds(t: str) -> float:
    parts = [float(x) for x in t.split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


FIRST_SEASON = 2010


def build(use_dump: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(PROCESSED / "events.parquet")
    eid = {(int(s), int(r)): e for s, r, e in zip(events.season, events["round"], events.event_id)}
    laps, stops = [], []
    for f in sorted(RAW.glob("*_laps.json")):
        doc = json.loads(f.read_text())
        e = eid.get((int(doc["season"]), int(doc["round"])))
        if e is None:
            continue
        for lap in doc["Laps"]:
            for t in lap["Timings"]:
                if "time" not in t:
                    continue
                pos = int(t["position"]) if "position" in t else np.nan  # a few rows lack it
                laps.append((e, t["driverId"], int(lap["number"]), pos, seconds(t["time"])))
    for f in sorted(RAW.glob("*_pitstops.json")):
        doc = json.loads(f.read_text())
        e = eid.get((int(doc["season"]), int(doc["round"])))
        if e is None:
            continue
        for s in doc["PitStops"]:
            dur = s.get("duration", "")
            stops.append((e, s["driverId"], int(s["stop"]), int(s["lap"]),
                          seconds(dur) if dur else np.nan))
    L = pd.DataFrame(laps, columns=["event_id", "driver_id", "lap_number", "position", "lap_time"]).assign(source="api")
    P = pd.DataFrame(stops, columns=["event_id", "driver_id", "stop", "lap", "duration"]).assign(source="api")
    if use_dump and any(DUMPS.glob("*.zip")):  # every other race from 2010 from the database dump
        D, DP = dump_tables()
        keep = D.event_id.str[:4].astype(int).ge(FIRST_SEASON) & ~D.event_id.isin(L.event_id)
        L = pd.concat([L, D[keep].assign(source="dump")], ignore_index=True)
        P = pd.concat([P, DP[DP.event_id.isin(D[keep].event_id)].assign(source="dump")], ignore_index=True)
    L = L.sort_values(["event_id", "driver_id", "lap_number"], ignore_index=True)
    L["time"] = L.groupby(["event_id", "driver_id"]).lap_time.cumsum()
    return L, P


DUMPS = ROOT / "data" / "raw" / "jolpica" / "dumps"
DUMP_LISTING = "https://api.jolpi.ca/data/dumps/download/"


def download_dump() -> Path:
    """The newest free delayed CSV dump, saved with a manifest entry; refused unless its
    sha256 matches the hash Jolpica lists for it."""
    import datetime as dt
    import hashlib

    import requests
    meta = requests.get(DUMP_LISTING, timeout=60).json()["delayed_dumps"]["csv"]
    path = DUMPS / f"delayed_csv_{meta['uploaded_at'][:10]}.zip"
    manifest_path = DUMPS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if path.exists() and manifest.get(path.name, {}).get("sha256") == meta["file_hash"]:
        print(f"{path.name} already present")
        return path
    body = requests.get(meta["download_url"], timeout=600).content
    h = hashlib.sha256(body).hexdigest()
    if h != meta["file_hash"]:
        raise RuntimeError(f"dump sha256 {h} does not match the published {meta['file_hash']}")
    DUMPS.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    manifest[path.name] = {"url": meta["download_url"], "uploaded_at": meta["uploaded_at"],
                           "published_sha256": meta["file_hash"], "sha256": h,
                           "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "terms": "free delayed dump, non-commercial use (Jolpica docs/database_dumps.md)"}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"saved {path.name} ({len(body):,} bytes, sha256 matches)")
    return path


def dump_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Race laps and pit stops from the newest CSV dump, keyed like the API pages."""
    import zipfile
    path = sorted(DUMPS.glob("*.zip"))[-1]
    z = zipfile.ZipFile(path)
    read = lambda name, cols: pd.read_csv(z.open(f"formula_one_{name}.csv"), usecols=cols)  # noqa: E731
    season = read("season", ["id", "year"]).rename(columns={"id": "season_id"})
    rnd = read("round", ["id", "number", "season_id"]).rename(columns={"id": "round_id", "number": "round"})
    sess = read("session", ["id", "round_id", "type"]).rename(columns={"id": "session_id"})
    sess = sess[sess.type == "R"].merge(rnd, on="round_id").merge(season, on="season_id")
    drv = read("driver", ["id", "reference"]).rename(columns={"id": "driver_pk"})
    td = read("teamdriver", ["id", "driver_id"]).rename(columns={"id": "team_driver_id", "driver_id": "driver_pk"})
    re_ = read("roundentry", ["id", "team_driver_id"]).rename(columns={"id": "round_entry_id"})
    se = read("sessionentry", ["id", "round_entry_id", "session_id"]).rename(columns={"id": "session_entry_id"})
    se = (se.merge(sess[["session_id", "year", "round"]], on="session_id").merge(re_, on="round_entry_id")
          .merge(td, on="team_driver_id").merge(drv, on="driver_pk"))
    events = pd.read_parquet(PROCESSED / "events.parquet")
    se = se.merge(events[["event_id", "season", "round"]], left_on=["year", "round"], right_on=["season", "round"])
    key = se.set_index("session_entry_id")[["event_id", "reference"]]
    lap = read("lap", ["id", "number", "position", "session_entry_id", "time"])
    lap = lap[lap.session_entry_id.isin(key.index) & lap.number.notna() & lap.time.notna()]
    L = lap.join(key, on="session_entry_id")
    L = pd.DataFrame({"event_id": L.event_id, "driver_id": L.reference, "lap_number": L.number.astype(int),
                      "position": L.position, "lap_time": L.time.map(seconds), "lap_pk": L["id"]})
    pit = read("pitstop", ["duration", "lap_id", "number", "session_entry_id"])
    pit = pit[pit.session_entry_id.isin(key.index)].join(key, on="session_entry_id")
    pit = pit.merge(L[["lap_pk", "lap_number"]], left_on="lap_id", right_on="lap_pk", how="left")
    P = pd.DataFrame({"event_id": pit.event_id, "driver_id": pit.reference, "stop": pit.number,
                      "lap": pit.lap_number, "duration": pit.duration.map(lambda d: seconds(d) if isinstance(d, str) else np.nan)})
    return L.drop(columns="lap_pk"), P.dropna(subset=["lap"]).astype({"lap": int})


def neutral_laps(L: pd.DataFrame) -> set:
    """Race laps flagged as neutralised, from the field's lap times."""
    med = L[L.lap_number > 1].groupby("lap_number").lap_time.median()
    slow = med[med > NEUTRAL_SLOW * med.median()].index
    out = set(int(x) for x in slow)
    for lap in sorted(out):
        out |= set(range(lap + 1, lap + 1 + NEUTRAL_AFTER))
    return out


def infer_pits(L: pd.DataFrame, neutral: set) -> pd.DataFrame:
    """Inferred pit stops (driver_id, lap = the in-lap) from lap-time losses."""
    ok = ~L.lap_number.isin(neutral) & (L.lap_number > 1)
    base = L[ok].groupby("driver_id").lap_time.median()
    L = L.assign(excess=L.lap_time - L.driver_id.map(base))
    nxt = L.groupby("driver_id").excess.shift(-1)
    pair = (L.excess + nxt).where(ok & ~L.lap_number.add(1).isin(neutral))
    cand = L[pair >= PIT_LOSS_S].copy()
    cand["loss"] = pair[pair >= PIT_LOSS_S]
    keep = []
    for d, g in cand.groupby("driver_id"):
        # consecutive candidate laps are one stop: the in-lap starts the slowest pair of laps
        g = g.sort_values("lap_number")
        run = (g.lap_number.diff() != 1).cumsum()
        for _, r in g.groupby(run):
            keep.append((d, int(r.lap_number.iloc[int(np.argmax(r.loss.to_numpy()))])))
    return pd.DataFrame(keep, columns=["driver_id", "lap"])


def race_frame(L: pd.DataFrame, stops: pd.DataFrame, team: pd.Series) -> pd.DataFrame:
    """One race's laps in the column layout of race_laps.parquet (FastF1), as far as Jolpica
    allows: pit in/out flags, a green flag (not neutralised), tyre age; no compound."""
    neutral = neutral_laps(L)
    if stops.empty:
        stops = infer_pits(L, neutral)
    L = L.copy()
    pit_in = set(zip(stops.driver_id, stops.lap))
    L["pit_in_time"] = np.where([k in pit_in for k in zip(L.driver_id, L.lap_number)], 1.0, np.nan)
    L["pit_out_time"] = np.where([(d, n - 1) in pit_in for d, n in zip(L.driver_id, L.lap_number)], 1.0, np.nan)
    L["track_status"] = np.where(L.lap_number.isin(neutral), "4", "1")
    stint = L.groupby("driver_id").pit_out_time.transform(lambda s: s.notna().cumsum())
    L["stint"] = stint + 1
    L["tyre_life"] = L.groupby(["driver_id", "stint"]).cumcount() + 1.0
    L["compound"] = "UNKNOWN"
    L["team"] = L.driver_id.map(team)
    L["speed_st"] = np.nan
    L["is_accurate"] = True
    return L


def check() -> dict:
    L, P = build()
    F = pd.read_parquet(PROCESSED / "race_laps.parquet")
    both = sorted(set(L.event_id) & set(F.event_id))
    out = {"races_compared": len(both)}
    pos, gaps, neut, pits = [], [], [], []
    for e in both:
        j, f = L[L.event_id == e], F[(F.event_id == e) & F.driver_id.notna()]
        m = j.merge(f[["driver_id", "lap_number", "position", "time", "track_status", "pit_in_time"]],
                    on=["driver_id", "lap_number"], suffixes=("_j", "_f"))
        pos.append((m.position_j == m.position_f).mean())
        m = m.sort_values(["lap_number", "time_f"])
        gj = m.sort_values(["lap_number", "time_j"]).groupby("lap_number").time_j.diff()
        gf = m.groupby("lap_number").time_f.diff()
        gaps.append(pd.DataFrame({"j": gj, "f": gf}).dropna())
        truth = set(f[f.track_status.astype(str).str.contains("[4567]", regex=True)].lap_number)
        flagged = neutral_laps(j)
        laps = set(j.lap_number)
        neut.append((len(flagged & truth), len(flagged), len(truth & laps)))
        pj = infer_pits(j, flagged)
        pf = f[f.pit_in_time.notna()]
        truth_p = set(zip(pf.driver_id, pf.lap_number))
        inf_p = set(zip(pj.driver_id, pj.lap))
        pits.append((len(inf_p & truth_p), len(inf_p), len(truth_p)))
    G = pd.concat(gaps)
    close_j, close_f = G.j < 1.0, G.f < 1.0
    tp, fl, tr = np.sum(neut, 0)
    out["position_agreement"] = float(np.mean(pos))
    out["gap_at_line"] = {"median_abs_diff_s": float((G.j - G.f).abs().median()),
                          "within_1s_agreement": float((close_j == close_f).mean()),
                          "within_1s_precision": float((close_j & close_f).sum() / close_j.sum()),
                          "within_1s_recall": float((close_j & close_f).sum() / close_f.sum())}
    out["neutralised_laps"] = {"precision": float(tp / max(fl, 1)), "recall": float(tp / max(tr, 1)),
                               "definition": "FastF1 track status 4-7 (SC, red, VSC) at any point in the lap"}
    tp, inf, tr = np.sum(pits, 0)
    out["pit_laps_vs_fastf1"] = {"precision": float(tp / max(inf, 1)), "recall": float(tp / max(tr, 1))}
    # 2012-2017: inferred pit laps against Jolpica's recorded stops
    rec = []
    for e, g in L[L.event_id.str[:4].astype(int).between(2011, 2017)].groupby("event_id"):
        truth_p = set(zip(P[P.event_id == e].driver_id, P[P.event_id == e].lap))
        if not truth_p:
            continue
        inf_p = set(map(tuple, infer_pits(g, neutral_laps(g))[["driver_id", "lap"]].to_numpy()))
        rec.append((len(inf_p & truth_p), len(inf_p), len(truth_p)))
    if rec:
        tp, inf, tr = np.sum(rec, 0)
        out["pit_laps_vs_recorded_2011_17"] = {"races": len(rec), "precision": float(tp / max(inf, 1)),
                                   "recall": float(tp / max(tr, 1))}
    path = ROOT / "outputs" / "analysis" / "oldlaps_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def dump_check() -> dict:
    """The dump against the API pages, on every race both have."""
    A, AP = build(use_dump=False)
    D, DP = dump_tables()
    both = sorted(set(A.event_id) & set(D.event_id))
    m = A[A.event_id.isin(both)].merge(D[D.event_id.isin(both)], on=["event_id", "driver_id", "lap_number"],
                                       how="outer", suffixes=("_api", "_dump"), indicator=True)
    b = m[m._merge == "both"]
    pm = AP[AP.event_id.isin(both)].merge(DP[DP.event_id.isin(both)], on=["event_id", "driver_id", "stop"],
                                          how="outer", suffixes=("_api", "_dump"), indicator=True)
    pb = pm[pm._merge == "both"]
    out = {"dump": json.loads((DUMPS / "manifest.json").read_text()), "races_compared": len(both),
           "laps": {k: int(v) for k, v in m._merge.value_counts().items()},
           "lap_time_equal": float(np.isclose(b.lap_time_api, b.lap_time_dump).mean()),
           "position_equal": float(((b.position_api == b.position_dump)
                                    | (b.position_api.isna() & b.position_dump.isna())).mean()),
           "pit_stops": {k: int(v) for k, v in pm._merge.value_counts().items()},
           "pit_lap_equal": float((pb.lap_api == pb.lap_dump).mean()),
           "pit_duration_equal": float(np.isclose(pb.duration_api, pb.duration_dump, equal_nan=True).mean())}
    path = ROOT / "outputs" / "analysis" / "oldlaps_dump_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    if sys.argv[1:] == ["check"]:
        check()
        return
    if sys.argv[1:] == ["dump"]:
        dump_check()
        return
    if sys.argv[1:] == ["download"]:
        download_dump()
        return
    L, P = build()
    L.to_parquet(PROCESSED / "jolpica_laps.parquet", index=False)
    P.to_parquet(PROCESSED / "jolpica_pitstops.parquet", index=False)
    print(f"jolpica_laps {len(L)} rows, {L.event_id.nunique()} races "
          f"({L.groupby('source').event_id.nunique().to_dict()}); pit stops {len(P)}")


if __name__ == "__main__":
    main()
