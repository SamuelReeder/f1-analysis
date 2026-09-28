"""Race conditions for the racing models: wet races and traffic.

    python -m f1rank.conditions

Wet races. From 2018, FastF1 says directly: rainfall recorded by the track's weather station
at any point in the race, or intermediate/wet tyres used (the rule racepace.py uses). Before
2018 neither source records weather, and lap times cannot tell (wet races usually follow
wet qualifying, so race pace relative to pole is not raised). Instead: hourly precipitation
at the circuit from the Open-Meteo historical archive (ERA5 reanalysis), summed over the race
window (start hour to start + WINDOW_H hours, UTC, start time from Jolpica). A race is wet
when that sum is at least WET_MM. The rule is checked against FastF1 on 2018 onward
(precision and recall at several thresholds in the check file); it is a weak proxy (about
0.6 precision and recall), so models give proxy-wet races (wet_source "precipitation")
their own coefficient rather than pooling them with FastF1's wet races. Responses are cached
in data/raw/openmeteo/ with their retrieval time and sha256.

Traffic. Per driver and race, the share of green-flag laps after lap 1 the car finished
within 1 s of the car ahead (FastF1 session times from 2018; Jolpica cumulative lap times
before, with neutralised laps inferred as in oldlaps.py).

Writes data/processed/race_conditions.parquet (event_id, wet, wet_source, precip_mm) and
data/processed/race_traffic.parquet (event_id, driver_id, green_laps, traffic_share), and
outputs/analysis/conditions_check.json.
"""

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .oldlaps import neutral_laps

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
JOLPICA = ROOT / "data" / "raw" / "jolpica"
RAW = ROOT / "data" / "raw" / "openmeteo"
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
       "&start_date={d0}&end_date={d1}&hourly=precipitation&timezone=UTC")
FIRST_SEASON = 2010
WINDOW_H = 2
WET_MM = 0.2  # chosen on 2018 onward, where precision and recall against FastF1 balance (sweep in the check)
CLOSE_S = 1.0


def race_info() -> pd.DataFrame:
    rows = []
    for f in sorted(JOLPICA.glob("*_results.json")):
        for r in json.loads(f.read_text()):
            loc = r["Circuit"]["Location"]
            rows.append((int(r["season"]), int(r["round"]), float(loc["lat"]), float(loc["long"]),
                         r["date"], r.get("time", "12:00:00Z")))
    info = pd.DataFrame(rows, columns=["season", "round", "lat", "lon", "date", "time"])
    ev = pd.read_parquet(PROCESSED / "events.parquet")[["event_id", "season", "round"]]
    return ev.merge(info, on=["season", "round"])


def precipitation(r) -> float:
    """Precipitation (mm) at the circuit over the race window."""
    path = RAW / f"{r.event_id}.json"
    start = pd.Timestamp(f"{r.date}T{r.time.rstrip('Z')}", tz="UTC").floor("h")
    if not path.exists():
        d0, d1 = start.date(), (start + pd.Timedelta(hours=WINDOW_H + 1)).date()
        for attempt in range(6):
            try:
                resp = requests.get(URL.format(lat=r.lat, lon=r.lon, d0=d0, d1=d1), timeout=60)
            except requests.RequestException:
                time.sleep(30 * 2 ** attempt)
                continue
            if resp.status_code != 429:
                break
            time.sleep(30 * 2 ** attempt)
        resp.raise_for_status()
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text)
        man = RAW / "manifest.json"
        m = json.loads(man.read_text()) if man.exists() else {}
        m[path.name] = {"retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "sha256": hashlib.sha256(resp.text.encode()).hexdigest()}
        man.write_text(json.dumps(m, indent=1, sort_keys=True))
        time.sleep(0.5)
    h = json.loads(path.read_text())["hourly"]
    s = pd.Series(h["precipitation"], index=pd.to_datetime(h["time"]).tz_localize("UTC"), dtype=float)
    return float(s[(s.index >= start) & (s.index <= start + pd.Timedelta(hours=WINDOW_H))].sum())


def fastf1_wet() -> pd.Series:
    laps = pd.read_parquet(PROCESSED / "race_laps.parquet")
    rain = pd.read_parquet(PROCESSED / "race_weather.parquet").groupby("event_id").rainfall.any()
    tyres = laps.groupby("event_id").compound.apply(lambda c: c.isin(["INTERMEDIATE", "WET"]).any())
    return (rain.reindex(tyres.index, fill_value=False) | tyres).rename("wet")


def traffic() -> pd.DataFrame:
    out = []
    F = pd.read_parquet(PROCESSED / "race_laps.parquet")
    F = F[F.driver_id.notna() & F.time.notna()]
    for e, L in F.groupby("event_id"):
        L = L.sort_values(["lap_number", "time"])
        gap = L.groupby("lap_number").time.diff()
        green = (L.track_status.astype(str) == "1") & (L.lap_number > 1) & L.pit_in_time.isna() & L.pit_out_time.isna()
        out.append(pd.DataFrame({"event_id": e, "driver_id": L.driver_id, "green": green, "close": green & (gap < CLOSE_S)}))
    old = PROCESSED / "jolpica_laps.parquet"
    if old.exists():
        J = pd.read_parquet(old)
        P = pd.read_parquet(PROCESSED / "jolpica_pitstops.parquet")
        for e, L in J[J.event_id < F.event_id.min()].groupby("event_id"):
            L = L.sort_values(["lap_number", "time"])
            gap = L.groupby("lap_number").time.diff()
            n = neutral_laps(L)
            s = P[P.event_id == e]
            pit = set(zip(s.driver_id, s.lap)) | set(zip(s.driver_id, s.lap + 1))
            green = (~L.lap_number.isin(n) & (L.lap_number > 1)
                     & ~pd.Series([k in pit for k in zip(L.driver_id, L.lap_number)], index=L.index))
            out.append(pd.DataFrame({"event_id": e, "driver_id": L.driver_id, "green": green,
                                     "close": green & (gap < CLOSE_S)}))
    T = pd.concat(out).groupby(["event_id", "driver_id"]).agg(green_laps=("green", "sum"), close=("close", "sum"))
    T["traffic_share"] = T.close / T.green_laps.where(T.green_laps > 0)
    return T.drop(columns="close").reset_index()


def main() -> None:
    info = race_info()
    info = info[info.season >= FIRST_SEASON]
    info["precip_mm"] = [precipitation(r) for r in info.itertuples()]
    ff = fastf1_wet()
    info["wet_fastf1"] = info.event_id.map(ff)
    info["wet_precip"] = info.precip_mm >= WET_MM
    both = info.dropna(subset=["wet_fastf1"])
    truth, rule = both.wet_fastf1.astype(bool), both.wet_precip
    check = {"races_compared": int(len(both)), "rule": f"precipitation >= {WET_MM} mm from start to +{WINDOW_H} h",
             "precision": float((truth & rule).sum() / max(rule.sum(), 1)),
             "recall": float((truth & rule).sum() / max(truth.sum(), 1)),
             "agreement": float((truth == rule).mean()),
             "fastf1_wet": int(truth.sum()), "rule_wet": int(rule.sum())}
    laps = pd.read_parquet(PROCESSED / "race_laps.parquet")
    share = both.event_id.map(laps.groupby("event_id").compound.apply(lambda c: c.isin(["INTERMEDIATE", "WET"]).mean()))
    sweep = []
    for name, t in (("fastf1_flag", truth), ("wet_tyres_10pct_of_laps", share >= 0.1)):
        for thr in (0.1, 0.2, 0.5, 1.0):
            r = both.precip_mm >= thr
            sweep.append({"truth": name, "threshold_mm": thr, "truth_wet": int(t.sum()), "rule_wet": int(r.sum()),
                          "precision": round(float((t & r).sum() / max(r.sum(), 1)), 3),
                          "recall": round(float((t & r).sum() / max(t.sum(), 1)), 3)})
    check["threshold_sweep"] = sweep
    has = info.wet_fastf1.notna()
    info["wet"] = np.where(has, info.wet_fastf1.fillna(False).astype(bool), info.wet_precip)
    info["wet_source"] = np.where(has, "fastf1", "precipitation")
    check["wet_races_by_season"] = info.groupby("season").wet.sum().astype(int).to_dict()
    info[["event_id", "wet", "wet_source", "precip_mm"]].to_parquet(PROCESSED / "race_conditions.parquet", index=False)
    T = traffic()
    T.to_parquet(PROCESSED / "race_traffic.parquet", index=False)
    check["traffic_driver_races"] = int(len(T))
    path = ROOT / "outputs" / "analysis" / "conditions_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(check, indent=1))
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
