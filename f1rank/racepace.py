"""Race pace and tyre degradation, 2018 onward (stage 2, step 3 of docs/racing_approach.md).

    python -m f1rank.racepace stage-a     # per-race estimates (outputs/race/stage_a_*.csv)
    python -m f1rank.racepace stage-b     # model across races + held-out test (outputs/race/)
    python -m f1rank.racepace stage-a-old # stage A for 2010-2017 races from Jolpica laps (*_old.*)

Stage A, per dry race: a robust (Huber) regression of clean laps,

    y = -100 log(lap / race median clean lap)          (percent, positive = faster)
      = driver pace at tyre age 10 laps
      + driver degradation deviation x (tyre age - 10)
      + car lap number (fuel burn and track evolution, common to all cars in the race)
      + compound (race-specific) + compound x tyre age (common degradation per compound)
      + dirty air (car ahead < 1 s; 1-2 s) + unpressured (no car within 5 s ahead or behind)

Pace and degradation are relative to the race's field (centred over drivers); degradation
is relative to the race's common slope per compound, which is what the data identify
(within a stint, tyre age and lap number rise together; only stint resets and compound
changes separate fuel burn from the common part of degradation). Standard errors and the
pace/degradation covariance come from a moving-block residual bootstrap within stints, so
correlated laps are not treated as independent.

Clean laps: green for the whole lap (FastF1 track status), not lap 1, not a car's last
lap, not a pit in or out lap, FastF1 marks the timing accurate, slick tyres, not more than
10% slower than the race's median clean lap, and outside the event timeline's
exclusions for that car: from 2 race laps before a retirement; 1 lap either side of a
contact incident; the lap of and after a spin or off-track moment; from a contact incident
to the pit stop that follows (suspected damage); the 2 laps after a neutralisation ends.
Races with rain or intermediate/wet tyres are excluded.

Before 2018 (stage-a-old), laps come from Jolpica (oldlaps.race_frame): compounds are
unknown (one "UNKNOWN" compound per race, so a common degradation slope and no compound
offsets), tyre age counts laps since the car's last pit stop (recorded from 2011, inferred
before), green-flag laps are those not flagged as neutralised from the field's lap times,
and FastF1's accuracy flag does not exist. Races are wet by the precipitation proxy
(conditions.py), which misses about 40% of wet races; the 10%-slower cut removes the
slowest wet laps of the races it misses. This is the weaker observation model the approach
specifies for these seasons.

Stage B, across races (NumPyro): for teammates in the same race,

    (pace gap, degradation gap) ~ bivariate Student-t
        mean  = (gamma x qualifying gap + race-specific gap, degradation-specific gap)
        scale = stage-A covariance + extra race-to-race variation

The qualifying gap is stage 1's pace in the current car at that event (posterior mean). Each
driver has a race-specific pace effect and a degradation effect, shrunk towards zero, with
a correlation between them. A held-out test (seasons after the training seasons) decides
whether the race-specific part improves on the qualifying link alone.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "race"

SLICKS = {"HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD"}
WET = {"INTERMEDIATE", "WET"}
MAX_SLOW = 1.10       # laps this much slower than the race's median clean lap are not representative
REF_AGE = 10.0        # tyre age (laps) at which driver pace is stated
CLOSE_S, NEAR_S = 1.0, 2.0
UNPRESSURED_S = 5.0
MIN_LAPS = 15         # clean laps a driver needs in a race for a stage-A estimate
BLOCK = 5             # laps per bootstrap block
N_BOOT = 200


# --------------------------------------------------------------------------- clean laps

def leader_lap(laps: pd.DataFrame):
    ends = np.sort(laps.groupby("lap_number").time.min().dropna().to_numpy())
    return lambda t: np.searchsorted(ends, np.asarray(t, float), side="right") + 1


def exclusions(tl: pd.DataFrame) -> tuple[dict, set]:
    """Race-lap windows to drop per driver, and race-wide race laps to drop."""
    per = {}
    add = lambda d, a, b: per.setdefault(d, []).append((a, b))  # noqa: E731
    wide = set()
    for r in tl.itertuples():
        if r.kind == "retirement":
            add(r.driver_id, r.lap_start - 2, 10 ** 6)
        elif r.kind == "incident_noted" and r.detail.startswith("contact") and pd.notna(r.lap_start):
            add(r.driver_id, r.lap_start - 1, r.lap_start + 1)
        elif r.kind == "off_track" and ("spun" in r.detail or "off track" in r.detail):
            add(r.driver_id, r.lap_start, r.lap_start + 1)
        elif r.kind == "suspected_damage":
            ev = json.loads(r.evidence)[0]
            add(r.driver_id, ev["incident_lap"], ev["pit_lap"])
        elif r.kind in ("safety_car", "vsc", "red_flag") and pd.notna(r.lap_end):
            wide |= {int(r.lap_end), int(r.lap_end) + 1}
    return per, wide


def clean_laps(laps: pd.DataFrame, tl: pd.DataFrame, compounds=SLICKS) -> pd.DataFrame:
    L = laps.sort_values(["lap_number", "position"]).copy()
    lap_at = leader_lap(L)
    L["race_lap"] = lap_at(L.time.to_numpy() - 1e-3)
    L["gap_ahead"] = L.groupby("lap_number").time.diff()
    L["gap_behind"] = -L.groupby("lap_number").time.diff(-1)
    last = L.groupby("driver_id").lap_number.transform("max")
    ok = ((L.track_status.astype(str) == "1") & (L.lap_number > 1) & (L.lap_number < last)
          & L.pit_in_time.isna() & L.pit_out_time.isna() & L.is_accurate.astype(bool)
          & L.compound.isin(compounds) & L.tyre_life.notna() & L.lap_time.notna() & L.driver_id.notna())
    per, wide = exclusions(tl)
    drop = L.race_lap.isin(wide).to_numpy().copy()
    for d, windows in per.items():
        m = (L.driver_id == d).to_numpy()
        for a, b in windows:
            drop |= m & (L.race_lap >= a).to_numpy() & (L.race_lap <= b).to_numpy()
    C = L[ok & ~drop].copy()
    C = C[C.lap_time <= MAX_SLOW * C.lap_time.median()]
    C["y"] = -100 * np.log(C.lap_time / C.lap_time.median())
    C["close"] = (C.gap_ahead < CLOSE_S).astype(float)
    C["near"] = ((C.gap_ahead >= CLOSE_S) & (C.gap_ahead < NEAR_S)).astype(float)
    C["unpressured"] = ((C.gap_ahead.isna() | (C.gap_ahead > UNPRESSURED_S))
                        & (C.gap_behind.isna() | (C.gap_behind > UNPRESSURED_S))).astype(float)
    C["stint_key"] = C.driver_id + "|" + C.stint.astype(str)
    n = C.groupby("driver_id").size()
    return C[C.driver_id.isin(n[n >= MIN_LAPS].index)]


# --------------------------------------------------------------------------- stage A

def huber(X: np.ndarray, y: np.ndarray, k: float = 1.345, iters: int = 50) -> np.ndarray:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
        sw = np.sqrt(np.minimum(1.0, k * scale / np.maximum(np.abs(r), 1e-12)))
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-7:
            return new
        beta = new
    return beta


def design(C: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    drivers = sorted(C.driver_id.unique())
    compounds = C.compound.value_counts().index.tolist()  # most used first = reference
    age = C.tyre_life.to_numpy(float)
    D = (C.driver_id.to_numpy()[:, None] == np.array(drivers)[None, :]).astype(float)
    cols = [D, (D[:, 1:] * (age - REF_AGE)[:, None]),
            ((C.lap_number.to_numpy(float) - C.lap_number.mean()) / 10)[:, None],
            C[["close", "near", "unpressured"]].to_numpy(float)]
    cols += [(C.compound == c).to_numpy(float)[:, None] for c in compounds[1:]]
    cols += [((C.compound == c).to_numpy(float) * (age - REF_AGE))[:, None] for c in compounds]
    return np.hstack(cols), drivers, compounds


def stage_a_race(C: pd.DataFrame, rng) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Driver pace and degradation (centred over the field) with bootstrap draws."""
    X, drivers, compounds = design(C)
    y = C.y.to_numpy()
    n = len(drivers)

    def summary(beta):
        pace = beta[:n] - beta[:n].mean()
        dev = np.concatenate([[0.0], beta[n:2 * n - 1]])
        return pace, dev - dev.mean()

    beta = huber(X, y)
    pace, deg = summary(beta)
    resid = y - X @ beta
    # moving-block residual bootstrap within stints: blocks of BLOCK consecutive laps
    idx = C.reset_index(drop=True).groupby("stint_key").indices
    draws = np.zeros((N_BOOT, 2, n))
    for b in range(N_BOOT):
        e = np.empty_like(resid)
        for rows in idx.values():
            rows = np.sort(rows)
            m = len(rows)
            starts = rng.integers(0, max(1, m - BLOCK + 1), size=int(np.ceil(m / BLOCK)))
            take = np.concatenate([rows[s:s + BLOCK] for s in starts])[:m]
            e[rows] = resid[take] if len(take) == m else resid[rows]
        draws[b] = summary(huber(X, X @ beta + e))
    common = {"lap_trend_per10": float(beta[2 * n - 1]), "close": float(beta[2 * n]),
              "near": float(beta[2 * n + 1]), "unpressured": float(beta[2 * n + 2]),
              "compounds": compounds, "median_lap_s": float(C.lap_time.median())}
    out = pd.DataFrame({"driver_id": drivers, "pace": pace, "deg": deg,
                        "pace_se": draws[:, 0].std(0), "deg_se": draws[:, 1].std(0),
                        "n_laps": C.groupby("driver_id").size().reindex(drivers).to_numpy(),
                        "n_stints": C.groupby("driver_id").stint.nunique().reindex(drivers).to_numpy()})
    return out, draws, common


OLD_COMPOUNDS = {"UNKNOWN"}


def fastf1_races():
    laps = pd.read_parquet(PROCESSED / "race_laps.parquet")
    weather = pd.read_parquet(PROCESSED / "race_weather.parquet")
    rain = weather.groupby("event_id").rainfall.any()
    wet_tyres = laps.groupby("event_id").compound.apply(lambda c: c.isin(WET).any())
    for event_id, L in laps.groupby("event_id"):
        yield event_id, L, bool(rain.get(event_id, False) or wet_tyres.get(event_id, False))


def jolpica_races():
    """Pre-FastF1 races from Jolpica laps, in race_laps' layout (oldlaps.race_frame)."""
    from .oldlaps import race_frame
    J = pd.read_parquet(PROCESSED / "jolpica_laps.parquet")
    P = pd.read_parquet(PROCESSED / "jolpica_pitstops.parquet")
    first = pd.read_parquet(PROCESSED / "race_laps.parquet").event_id.min()
    wet = pd.read_parquet(PROCESSED / "race_conditions.parquet").set_index("event_id").wet
    team = pd.read_parquet(PROCESSED / "race.parquet").set_index(["event_id", "driver_id"]).team
    for event_id, L in J[J.event_id < first].groupby("event_id"):
        F = race_frame(L, P[P.event_id == event_id], team.loc[event_id])
        yield event_id, F, bool(wet.get(event_id, False))


def stage_a(old: bool = False) -> None:
    timeline = pd.read_parquet(PROCESSED / "timeline.parquet")
    rng = np.random.default_rng(0)
    drivers, pairs, races = [], [], []
    for event_id, L, wet in (jolpica_races() if old else fastf1_races()):
        info = {"event_id": event_id, "wet": wet}
        if wet:
            races.append(info)
            continue
        C = clean_laps(L, timeline[timeline.event_id == event_id], OLD_COMPOUNDS if old else SLICKS)
        if C.driver_id.nunique() < 10:
            races.append({**info, "skipped": "fewer than 10 drivers with enough clean laps"})
            continue
        est, draws, common = stage_a_race(C, rng)
        team = L.groupby("driver_id").team.first()
        est = est.assign(event_id=event_id, team=est.driver_id.map(team))
        drivers.append(est)
        races.append({**info, "n_clean_laps": int(len(C)), "n_drivers": int(len(est)), **common})
        # teammate gaps with the bootstrap covariance of (pace gap, degradation gap)
        pos = {d: i for i, d in enumerate(est.driver_id)}
        for tm, g in est.groupby("team"):
            if len(g) != 2:
                continue
            a, b = sorted(g.driver_id)
            ia, ib = pos[a], pos[b]
            dd = draws[:, :, ia] - draws[:, :, ib]
            cov = np.cov(dd.T)
            ea, eb = est.set_index("driver_id").loc[a], est.set_index("driver_id").loc[b]
            pairs.append({"event_id": event_id, "team": tm, "a": a, "b": b,
                          "pace_gap": ea.pace - eb.pace, "deg_gap": ea.deg - eb.deg,
                          "pace_gap_se": float(np.sqrt(cov[0, 0])), "deg_gap_se": float(np.sqrt(cov[1, 1])),
                          "pace_deg_cov": float(cov[0, 1]), "n_laps_a": int(ea.n_laps), "n_laps_b": int(eb.n_laps)})
        print(event_id, len(C), "clean laps", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = "_old" if old else ""
    pd.concat(drivers).to_csv(OUT / f"stage_a_drivers{sfx}.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / f"stage_a_pairs{sfx}.csv", index=False)
    (OUT / f"stage_a_races{sfx}.json").write_text(json.dumps(races, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["stage-a", "stage-b", "stage-a-old"])
    args = p.parse_args()
    if args.step in ("stage-a", "stage-a-old"):
        stage_a(old=args.step == "stage-a-old")
    else:
        from .racemodel import stage_b
        stage_b()


if __name__ == "__main__":
    main()
