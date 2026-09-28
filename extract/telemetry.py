"""Per-lap telemetry summaries for the dry races used by race pace (FastF1 environment).

    .venv-fastf1/bin/python extract/telemetry.py [--refresh]

docs/racing_approach.md (Granularity, telemetry use 1): a coasting / not-pushing flag for
race pace, entered as a covariate and kept only if it changes held-out results. Car data
(about 4 samples a second: speed, throttle, brake) come from FastF1's live-timing API
(`fastf1.api.car_data`), which does not query Jolpica. Per lap:

    coast_s          seconds with throttle <= 1%, brake off and speed >= 150 km/h
                     (lifting before a braking zone: lift-and-coast)
    full_throttle_s  seconds with throttle >= 99%
    samples          car-data samples in the lap

Alignment. Car-data timestamps are dates; laps are in session time. Session time zero is
taken as FastF1 does (the largest Date - Time offset over the cars' streams), then a shift
of up to +-1 s is chosen per race to match the car's speed at each lap's end with FastF1's
finish-line speed trap (the median absolute difference is recorded; on 2024 Bahrain the
best shift was +0.25 to +0.5 s with a median difference of about 1 km/h).

Races: those in outputs/race/stage_a_pairs.csv. Writes
data/raw/fastf1_tables/<event_id>_R/telemetry_laps.parquet and telemetry.json (shift,
speed-trap agreement, retrieval time, sha256); `python -m f1rank.racedata` combines them
into data/processed/race_telemetry.parquet.
"""

import argparse
import datetime as dt
import hashlib
import json
import warnings

import fastf1
import numpy as np
import pandas as pd

from ff1 import ROOT, enable_cache, patient

warnings.filterwarnings("ignore", message=".*fastf1.api.*")
from fastf1 import api  # noqa: E402

TABLES = ROOT / "data" / "raw" / "fastf1_tables"
SHIFTS = np.round(np.arange(-1.0, 1.01, 0.05), 2)


def session_path(season: int, rnd: int, schedules: dict) -> str:
    if season not in schedules:
        schedules[season] = fastf1.get_event_schedule(season, include_testing=False)
    ev = schedules[season]
    ev = ev[ev.RoundNumber == rnd].iloc[0]
    # the archive's paths use the local date (a night race can end after midnight UTC)
    race = next(ev[f"Session{i}Date"] for i in range(5, 0, -1) if ev[f"Session{i}"] == "Race")
    return api.make_path(ev.EventName, ev.EventDate.strftime("%Y-%m-%d"), "Race", race.strftime("%Y-%m-%d"))


def summarise(car: dict, laps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    t0 = max((d.Date - d.Time).max() for d in car.values())
    st = {n: (d.Date - t0).dt.total_seconds().to_numpy() for n, d in car.items()}
    # shift that best matches the finish-line speed trap
    err = []
    for s in SHIFTS:
        e = []
        for n, d in car.items():
            lap = laps[(laps.DriverNumber == n) & laps.SpeedFL.notna() & laps.Time.notna()]
            if len(lap):
                e.append(np.abs(np.interp(lap.Time.to_numpy(), st[n] + s, d.Speed.to_numpy()) - lap.SpeedFL.to_numpy()))
        err.append(np.median(np.concatenate(e)))
    shift = float(SHIFTS[int(np.argmin(err))])
    rows = []
    for n, d in car.items():
        t = st[n] + shift
        dtime = np.clip(np.diff(t, append=t[-1]), 0, 0.5)
        coast = (d.Throttle.to_numpy() <= 1) & ~d.Brake.to_numpy().astype(bool) & (d.Speed.to_numpy() >= 150)
        full = d.Throttle.to_numpy() >= 99
        for lap in laps[laps.DriverNumber == n].itertuples():
            if pd.isna(lap.LapStartTime) or pd.isna(lap.Time):
                continue
            m = (t >= lap.LapStartTime) & (t < lap.Time)
            rows.append({"DriverNumber": n, "LapNumber": lap.LapNumber, "coast_s": float(dtime[m & coast].sum()),
                         "full_throttle_s": float(dtime[m & full].sum()), "samples": int(m.sum())})
    return pd.DataFrame(rows), {"shift_s": shift, "speed_trap_median_abs_diff_kmh": float(min(err)),
                                "speed_trap_median_abs_diff_at_zero_shift_kmh": float(err[int(np.argmin(np.abs(SHIFTS)))])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    enable_cache()
    events = sorted(set(pd.read_csv(ROOT / "outputs" / "race" / "stage_a_pairs.csv").event_id))
    schedules = {}
    for e in events:
        out = TABLES / f"{e}_R"
        if (out / "telemetry.json").exists() and not args.refresh:
            continue
        season, rnd = int(e[:4]), int(e[5:])
        try:
            car = patient(api.car_data, session_path(season, rnd, schedules))
            laps = pd.read_parquet(out / "laps.parquet")
            T, info = summarise(car, laps)
        except Exception as exc:  # noqa: BLE001 - record and continue
            print(f"{e}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        path = out / "telemetry_laps.parquet"
        T.to_parquet(path, index=False)
        info.update({"fastf1_version": fastf1.__version__, "rows": int(len(T)),
                     "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (out / "telemetry.json").write_text(json.dumps(info, indent=1))
        print(f"{e}: {len(T)} laps, shift {info['shift_s']} s, trap diff {info['speed_trap_median_abs_diff_kmh']:.1f} km/h",
              flush=True)


if __name__ == "__main__":
    main()
