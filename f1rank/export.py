"""Write dashboard-ready outputs from the main fit.

outputs/ratings/
  driver_series.parquet   every driver at every event (revised = uses all data)
  car_series.parquet      every team at every event: track-neutral pace, pace at that circuit
  current_drivers.csv     leaderboard at the latest event, with rank ranges
  current_cars.csv        car-package leaderboard at the latest event
  pairwise_drivers.csv    P(row driver faster than column driver) at the latest event
  current_draws.npz       thinned joint draws (drivers, cars) for a swap calculator
  meta.json               data as-of, model version, fit diagnostics
outputs/snapshots/<event_id>.json   append-only record of what was published
"""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .design import build_design
from .fit import FITS, load
from .ratings import (SEC_PER_PCT, car_leaderboard, car_series, driver_leaderboard,
                      driver_series, flat)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "ratings"
SNAPSHOTS = ROOT / "outputs" / "snapshots"
MODEL_VERSION = "quali-v1"
N_DRAWS_EXPORT = 400
SECONDS_COLS = ["q05", "q25", "median", "q75", "q95"]


def to_seconds(df: pd.DataFrame,
               prefixes=("", "gap_", "at_circuit_", "track_", "in_team_", "compat_")) -> pd.DataFrame:
    df = df.copy()
    for p in prefixes:
        for c in SECONDS_COLS:
            if p + c in df and p != "track_":
                df[p + c + "_s"] = df[p + c] * SEC_PER_PCT
    return df


def export(fit_name: str = "main", start: int = 2010) -> dict:
    design = build_design(start)
    post, info = load(FITS / f"{fit_name}.npz")
    last = design.events.iloc[-1]
    OUT.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)

    to_seconds(driver_series(design, post)).to_parquet(OUT / "driver_series.parquet", index=False)
    to_seconds(car_series(design, post)).to_parquet(OUT / "car_series.parquet", index=False)

    drivers = to_seconds(driver_leaderboard(design, post))
    cars = to_seconds(car_leaderboard(design, post))
    drivers.to_csv(OUT / "current_drivers.csv", index=False)
    cars.to_csv(OUT / "current_cars.csv", index=False)

    # joint draws at the latest event, relative to that field
    e_rows = np.flatnonzero(design.entries.event_idx.to_numpy() == last.event_idx)
    c_rows = np.flatnonzero(design.cars.event_idx.to_numpy() == last.event_idx)
    sk = flat(post, "skill")[:, e_rows]
    sk -= sk.mean(1, keepdims=True)
    cr = flat(post, "car")[:, c_rows]
    cr -= cr.mean(1, keepdims=True)
    ids = design.entries.driver_id.to_numpy()[e_rows]
    teams = design.cars.team.to_numpy()[c_rows]
    ahead = (sk[:, :, None] > sk[:, None, :]).mean(0)
    pd.DataFrame(ahead, index=ids, columns=ids).to_csv(OUT / "pairwise_drivers.csv")
    pick = np.random.default_rng(0).choice(sk.shape[0], N_DRAWS_EXPORT, replace=False)
    np.savez_compressed(OUT / "current_draws.npz", driver_ids=ids, skill_s=sk[pick] * SEC_PER_PCT,
                        teams=teams, car_s=cr[pick] * SEC_PER_PCT)

    meta = {
        "model_version": MODEL_VERSION, "fit": fit_name,
        "data_as_of": {"event_id": last.event_id, "race_name": last.race_name, "date": last.date},
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "scope": "one-lap qualifying pace; ratings relative to the field at each event",
        "units": "seconds per 90-second lap (positive = faster); model units are % of lap time",
        "window": f"{start}-{int(design.events.season.max())}",
        "n_lap_times": int(len(design.obs)), "diagnostics": info,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))

    snap = {
        "meta": meta,
        "drivers": drivers[["driver_id", "name", "team", "median_s", "q05_s", "q95_s",
                            "rank_median", "rank_lo", "rank_hi", "p_fastest"]].to_dict("records"),
        "cars": cars[["team", "constructor", "median_s", "q05_s", "q95_s", "gap_median_s",
                      "rank_lo", "rank_hi"]].to_dict("records"),
    }
    snap_path = SNAPSHOTS / f"{last.event_id}_{MODEL_VERSION}.json"
    snap_path.write_text(json.dumps(snap, indent=1, default=float))
    return meta


if __name__ == "__main__":
    print(json.dumps(export(), indent=1))
