"""How much racing signal is in the 2018+ timing data? A reproducible check.

Runs in its own environment, because FastF1 requires pandas < 3:

    python3 -m venv .venv-fastf1 && .venv-fastf1/bin/pip install fastf1 pyarrow
    .venv-fastf1/bin/python analysis/race_signal.py [--first 2018] [--last 2025]

Reads the stage-1 qualifying tables in data/processed/ (run f1rank.build first).
Downloads race timing with FastF1 (cached in data/raw/fastf1/) and writes to
outputs/analysis/race_signal/:

    driver_races.csv  one row per driver-race: lap counts, battles, passes, race pace
    pairs.csv         teammate race-pace and qualifying gaps per race
    summary.json      signal volumes, stability and race-vs-qualifying statistics

Definitions:
- clean lap: green flag, not lap 1 or the last lap, no pit in/out, FastF1 IsAccurate
- clean-air lap: clean and more than 2 s behind the car ahead at the line
- battle lap: green, no pit, within 1 s of the car ahead at the line
- pass: two cars swap order between consecutive laps with neither pitting on either
  lap and the second lap green; teammate swaps are excluded
- race pace (dry races only): driver effect from a Huber regression of
  100*log(lap / race median) on clean-air slick laps within 107% of the median, with
  lap number, compound and compound x tyre age; positive = faster, % of lap time
- qualifying gap: median over shared segments of the teammate pace difference, with
  the stage-1 definitions (segment median, laps > 5% off dropped)

Pair-season statistics use medians (not means), so one compromised session cannot
dominate, and bootstrap 95% intervals over pair-seasons.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "fastf1"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "analysis" / "race_signal"
SLICKS = ["SOFT", "MEDIUM", "HARD"]
MIN_PACE_LAPS = 10       # per driver per race, for a race-pace estimate
MIN_PAIR_RACES = 8       # per pair-season, for pair-season statistics
N_BOOT = 2000


def patient(fn, *args, waits: int = 12, **kwargs):
    """Call fn, waiting out FastF1's API rate limit (500 calls/h; downloads are cached,
    so a rerun resumes where it stopped)."""
    for _ in range(waits):
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededError:
            print("FastF1 rate limit reached: waiting 10 minutes", flush=True)
            time.sleep(600)
    return fn(*args, **kwargs)


def huber_fit(X: np.ndarray, y: np.ndarray, k: float = 1.345, iters: int = 50) -> np.ndarray:
    """Huber M-estimate by iteratively reweighted least squares."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
        w = np.minimum(1.0, k * scale / np.maximum(np.abs(r), 1e-12))
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-8:
            return new
        beta = new
    return beta


def race_pace(laps: pd.DataFrame) -> pd.DataFrame:
    C = laps[laps.clean_air & laps.Compound.isin(SLICKS)].dropna(subset=["LapNumber", "TyreLife"])
    C = C[C.lap_s < 1.07 * C.lap_s.median()]
    n = C.groupby("Driver").size()
    C = C[C.Driver.isin(n[n >= MIN_PACE_LAPS].index)]
    if C.Driver.nunique() < 10:
        return pd.DataFrame(columns=["Driver", "pace", "pace_laps"])
    y = 100 * np.log(C.lap_s / C.lap_s.median())
    drivers = sorted(C.Driver.unique())
    compounds = sorted(C.Compound.unique())
    cols = [pd.get_dummies(C.Driver).reindex(columns=drivers, fill_value=0).to_numpy(float),
            C.LapNumber.to_numpy(float)[:, None] / 10]
    cols += [(C.Compound == c).to_numpy(float)[:, None] for c in compounds[1:]]
    cols += [((C.Compound == c) * C.TyreLife).to_numpy(float)[:, None] / 10 for c in compounds]
    beta = huber_fit(np.hstack(cols), y.to_numpy())
    return pd.DataFrame({"Driver": drivers, "pace": -beta[:len(drivers)],
                         "pace_laps": n.reindex(drivers).to_numpy()})


def passes(laps: pd.DataFrame) -> pd.DataFrame:
    out, prev = [], None
    for n, cur in laps.groupby("LapNumber"):
        cur = cur.set_index("Driver")
        if prev is not None and n > 1:
            ok = [d for d in prev.index.intersection(cur.index)
                  if not (prev.at[d, "pit"] or cur.at[d, "pit"])]
            if ok and cur.loc[ok, "green"].all():
                p0, p1 = prev.loc[ok, "Position"], cur.loc[ok, "Position"]
                for a in ok:
                    for b in ok:
                        if p0[a] > p0[b] and p1[a] < p1[b] and cur.at[a, "Team"] != cur.at[b, "Team"]:
                            out.append((a, b))
        prev = cur
    return pd.DataFrame(out, columns=["attacker", "defender"])


def load_race(season: int, rnd: int) -> pd.DataFrame | None:
    s = fastf1.get_session(season, rnd, "R")
    s.load(laps=True, telemetry=False, weather=True, messages=False)
    L = s.laps.copy()
    if L.empty:
        return None
    L["t"] = L.Time.dt.total_seconds()
    L["lap_s"] = L.LapTime.dt.total_seconds()
    L["pit"] = L.PitInTime.notna() | L.PitOutTime.notna()
    L["green"] = L.TrackStatus.astype(str) == "1"
    L = L.sort_values(["LapNumber", "Position"])
    L["gap_ahead"] = L.groupby("LapNumber").t.diff()
    last = L.LapNumber.max()
    L["clean"] = (L.green & ~L.pit & (L.LapNumber > 1) & (L.LapNumber < last)
                  & L.IsAccurate.astype(bool) & L.lap_s.notna())
    L["clean_air"] = L.clean & (L.gap_ahead.isna() | (L.gap_ahead > 2.0))
    L["battle"] = L.green & ~L.pit & (L.LapNumber > 1) & (L.gap_ahead < 1.0)
    wet = bool(s.weather_data.Rainfall.any()) or L.Compound.isin(["INTERMEDIATE", "WET"]).any()

    per = L.groupby("Driver").agg(team=("Team", "first"), clean=("clean", "sum"),
                                  clean_air=("clean_air", "sum"), battle=("battle", "sum"))
    P = passes(L)
    per["passes_made"] = P.attacker.value_counts().reindex(per.index).fillna(0).astype(int)
    per["passes_suffered"] = P.defender.value_counts().reindex(per.index).fillna(0).astype(int)
    if not wet:
        per = per.join(race_pace(L).set_index("Driver"))
    ids = s.results.set_index("Abbreviation").DriverId
    per = per.reset_index().assign(driver_id=lambda x: x.Driver.map(ids), season=season, round=rnd,
                                   event_id=f"{season}-{rnd:02d}", wet=wet)
    return per


def quali_gaps() -> pd.DataFrame:
    """Teammate qualifying gaps per event, stage-1 definitions (positive = a faster)."""
    q = pd.read_parquet(PROCESSED / "quali_times.parquet")
    ent = pd.read_parquet(PROCESSED / "entries.parquet")[["event_id", "driver_id", "team"]]
    q["y"] = -100 * np.log(q.time_s / q.groupby(["event_id", "segment"]).time_s.transform("median"))
    q = q[q.y >= -5.0].merge(ent, on=["event_id", "driver_id"])
    m = q.merge(q, on=["event_id", "segment", "team"])
    m = m[m.driver_id_x < m.driver_id_y]
    g = m.assign(gap=m.y_x - m.y_y).groupby(["event_id", "driver_id_x", "driver_id_y"]).gap.median()
    return g.rename("quali_gap").reset_index().rename(columns={"driver_id_x": "a", "driver_id_y": "b"})


def teammate_pairs(R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (event_id, team), x in R[R.pace.notna()].groupby(["event_id", "team"]):
        if len(x) != 2:
            continue
        x = x.set_index("driver_id")
        a, b = sorted(x.index)
        rows.append(dict(event_id=event_id, season=int(x.season.iloc[0]), round=int(x["round"].iloc[0]),
                         a=a, b=b, race_gap=x.at[a, "pace"] - x.at[b, "pace"]))
    return pd.DataFrame(rows).merge(quali_gaps(), on=["event_id", "a", "b"], how="left")


def spearman_brown(r: float) -> float:
    """Full-season reliability from a split-half correlation (floored at 0)."""
    r = max(r, 0.0)
    return 2 * r / (1 + r)


def pair_season_stats(P: pd.DataFrame, rng: np.random.Generator) -> dict:
    P = P.dropna(subset=["quali_gap"]).assign(half=lambda x: x["round"] % 2,
                                             race_minus_quali=lambda x: x.race_gap - x.quali_gap)
    key = ["season", "a", "b"]
    n = P.groupby(key).size()
    P = P.set_index(key).loc[n[n >= MIN_PAIR_RACES].index].reset_index()
    cols = ["race_gap", "quali_gap", "race_minus_quali"]
    whole = P.groupby(key)[cols].median()
    halves = P.groupby(key + ["half"])[cols].median().unstack("half").dropna()

    def stats(idx) -> dict:
        w, h = whole.loc[idx], halves.loc[halves.index.intersection(idx)]
        out = {"race_vs_quali_pearson": w.race_gap.corr(w.quali_gap),
               "race_vs_quali_spearman": w.race_gap.corr(w.quali_gap, method="spearman")}
        for c in cols:
            out[f"split_half_{c}"] = spearman_brown(h[(c, 0)].corr(h[(c, 1)]))
        return out

    idx = whole.index
    point = stats(idx)
    boot = pd.DataFrame([stats(idx[rng.integers(0, len(idx), len(idx))]) for _ in range(N_BOOT)])
    return {"pair_seasons": int(len(idx)), "driver_races": int(len(P)),
            **{k: {"estimate": round(float(v), 3),
                   "ci95": [round(float(boot[k].quantile(0.025)), 3), round(float(boot[k].quantile(0.975)), 3)]}
               for k, v in point.items()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--first", type=int, default=2018)
    p.add_argument("--last", type=int, default=2025)
    args = p.parse_args()
    logging.disable(logging.INFO)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    rows = []
    for season in range(args.first, args.last + 1):
        sched = patient(fastf1.get_event_schedule, season, include_testing=False)
        for rnd in sched.RoundNumber:
            try:
                r = patient(load_race, season, int(rnd))
            except RateLimitExceededError:
                raise
            except Exception as e:  # noqa: BLE001 - a missing race is reported, not fatal
                print(f"skip {season}-{int(rnd):02d}: {e}", flush=True)
                continue
            if r is not None:
                rows.append(r)
                print(f"{season}-{int(rnd):02d} done", flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(OUT / "driver_races.csv", index=False)
    P = teammate_pairs(R)
    P.to_csv(OUT / "pairs.csv", index=False)

    races = R.groupby("event_id").agg(wet=("wet", "first"))
    season_driver = R.groupby(["season", "driver_id"]).agg(
        races=("event_id", "nunique"), clean=("clean", "sum"), clean_air=("clean_air", "sum"),
        battle=("battle", "sum"), passes_made=("passes_made", "sum"), passes_suffered=("passes_suffered", "sum"))
    full = season_driver[season_driver.races >= 0.8 * season_driver.groupby("season").races.transform("max")]
    both = R.assign(ok20=R.clean >= 20, ok10=R.clean_air >= 10).groupby(["event_id", "team"])[["ok20", "ok10"]].sum()
    summary = {
        "seasons": [args.first, args.last],
        "races": int(len(races)), "wet_races": int(races.wet.sum()),
        "per_full_season_driver_median": {c: float(full[c].median()) for c in
                                          ["clean", "clean_air", "battle", "passes_made", "passes_suffered"]},
        "team_races_both_20_clean": round(float((both.ok20 == 2).mean()), 3),
        "team_races_both_10_clean_air": round(float((both.ok10 == 2).mean()), 3),
        "teammate_race_comparisons": int(P.race_gap.notna().sum()),
        "pair_season_stats_dry": pair_season_stats(P, np.random.default_rng(0)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
