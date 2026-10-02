"""Pre-registered tests of weekend information for race pace (docs/race_features.md).

    python -m f1rank.race_features build                 # practice laps -> data/processed
    python -m f1rank.race_features validate longrun      # or traps, upgrades (GPU)

`build` combines data/raw/fastf1_tables/<event>_FP/ (extract/practice.py) into
data/processed/practice_laps.parquet and practice_sources.json, with driver ids from the
abbreviations in that weekend's race laps and the team lineage from the race entries.

`validate FEATURE` fits v1 plus the feature's covariates (race_features_model.py) on v1's
training folds and scores the remaining dry races of each season against v1's full
model, with v1's targets, rows and race-block bootstrap (race_total.improvement). Writes
outputs/race_features/<feature>/validation.json and the per-row predictions.
"""
import argparse
import json

import numpy as np
import pandas as pd

from .artifacts import atomic_json, digest, record
from .race_total import FOLDS, ROOT, improvement, inputs, point_prediction

OUT = ROOT / "outputs" / "race_features"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "data" / "raw" / "fastf1_tables"
UPGRADES = ROOT / "data" / "supplements" / "fia_upgrades.json"
FEATURES = ("longrun", "traps", "upgrades")
DRY = ("SOFT", "MEDIUM", "HARD")


def build() -> pd.DataFrame:
    from .racedata import snake
    race_laps = pd.read_parquet(PROCESSED / "race_laps.parquet", columns=["event_id", "driver", "driver_id"])
    ids = race_laps.dropna().drop_duplicates(["event_id", "driver"]).set_index(["event_id", "driver"]).driver_id
    entries = pd.read_parquet(PROCESSED / "race.parquet", columns=["event_id", "driver_id", "team"])
    frames, sources = [], {}
    for d in sorted(TABLES.glob("*_FP")):
        if not (d / "manifest.json").exists():
            continue
        event_id = d.name[:-3]
        manifest = json.loads((d / "manifest.json").read_text())
        sources[event_id] = {k: manifest[k] for k in ("session", "fastf1_version", "retrieved_utc", "event_name")}
        sources[event_id]["sha256"] = manifest["tables"]["laps"]["sha256"]
        laps = pd.read_parquet(d / "laps.parquet")
        laps.columns = [snake(c) for c in laps.columns]
        laps = laps.rename(columns={"team": "team_name"})  # FastF1's name; `team` is the lineage
        laps.insert(0, "event_id", event_id)
        laps.insert(1, "driver_id", [ids.get((event_id, a)) for a in laps.driver])
        frames.append(laps)
    out = pd.concat(frames, ignore_index=True)
    out = out.merge(entries, on=["event_id", "driver_id"], how="left", validate="many_to_one")
    out["driver_number"] = out.driver_number.astype(str)
    out.to_parquet(PROCESSED / "practice_laps.parquet", index=False)
    failures = TABLES / "practice_failures.json"
    (PROCESSED / "practice_sources.json").write_text(json.dumps({
        "sessions": sources,
        "not_available": json.loads(failures.read_text()) if failures.exists() else {},
    }, indent=1))
    print(f"practice_laps: {len(out)} laps from {len(sources)} weekends; "
          f"{int(out.driver_id.isna().sum())} laps by drivers not in the race")
    return out


def _centred(values: pd.Series) -> pd.Series:
    return values - values.groupby(level="event_id").transform("mean")


def longrun(practice: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Race-fuel practice pace (docs/race_features.md): team values (centred) and the
    teammate split, in percent (positive = faster)."""
    p = practice.dropna(subset=["driver_id", "team", "lap_time", "stint"])
    p = p[p.pit_in_time.isna() & p.pit_out_time.isna() & ~p.deleted.fillna(False).astype(bool)
          & p.is_accurate.fillna(False).astype(bool) & (p.track_status.astype(str) == "1")
          & p.compound.isin(DRY)]
    p = p[p.lap_time <= 1.07 * p.groupby(["event_id", "driver_id", "stint"]).lap_time.transform("median")]
    runs = p.groupby(["event_id", "driver_id", "team", "stint", "compound"]).lap_time.agg(["size", "mean"])
    runs = runs[runs["size"] >= 5].reset_index()
    drivers_per = runs.groupby(["event_id", "compound"]).driver_id.transform("nunique")
    runs = runs[drivers_per >= 3]
    median = runs.groupby(["event_id", "compound"])["mean"].transform("median")
    runs["rel"] = 100 * (median - runs["mean"]) / median
    driver = runs.groupby(["event_id", "team", "driver_id"]).rel.mean()
    team = driver.groupby(level=["event_id", "team"]).mean()
    has_pair = driver.groupby(level=["event_id", "team"]).size() == 2
    split = (driver - team.reindex(driver.index.droplevel("driver_id")).to_numpy()).reset_index()
    split = split[has_pair.reindex(pd.MultiIndex.from_frame(split[["event_id", "team"]])).to_numpy()]
    x_car = _centred(team).rename("longrun").reset_index()
    x_car.index = x_car.event_id + "|" + x_car.team
    x_drv = split.assign(longrun_split=split.rel)
    x_drv.index = x_drv.event_id + "|" + x_drv.driver_id
    return x_car[["longrun"]], x_drv[["longrun_split"]]


def traps(practice: pd.DataFrame) -> pd.DataFrame:
    """Straight-line speed in practice: the team's 90th percentile speed trap minus the
    event's median team value, per 10 km/h."""
    p = practice.dropna(subset=["team", "lap_time", "speed_st"])
    p = p[p.pit_in_time.isna() & p.pit_out_time.isna()]
    team = p.groupby(["event_id", "team"]).speed_st.quantile(.9)
    x = ((team - team.groupby(level="event_id").transform("median")) / 10).rename("traps").reset_index()
    x.index = x.event_id + "|" + x.team
    return x[["traps"]]


def upgrades(entries: pd.DataFrame, supplement: dict) -> pd.DataFrame:
    """Performance components declared so far in the season (this race included), centred
    across the race's teams, per 10 components."""
    declared = {(e, team): t["performance"] for e, doc in supplement["events"].items()
                for team, t in doc["teams"].items()}
    teams = entries[["event_id", "team"]].drop_duplicates().sort_values(["event_id", "team"])
    teams["n"] = [declared.get((e, t), 0) for e, t in zip(teams.event_id, teams.team)]
    teams["cum"] = teams.groupby([teams.event_id.str[:4], "team"]).n.cumsum()
    teams = teams.set_index(["event_id", "team"])
    x = (_centred(teams.cum) / 10).rename("upgrades").reset_index()
    x.index = x.event_id + "|" + x.team
    return x[["upgrades"]]


def covariates(feature: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if feature in ("longrun", "traps"):
        practice = pd.read_parquet(PROCESSED / "practice_laps.parquet")
        return longrun(practice) if feature == "longrun" else (traps(practice), None)
    entries = pd.read_parquet(PROCESSED / "race.parquet", columns=["event_id", "team"])
    return upgrades(entries, json.loads(UPGRADES.read_text())), None


def sources(feature: str) -> list:
    data = [PROCESSED / "practice_laps.parquet"] if feature != "upgrades" else [UPGRADES]
    return inputs() + data + [ROOT / "f1rank/race_features.py", ROOT / "f1rank/race_features_model.py",
                              ROOT / "docs/race_features.md"]


def validate(feature: str) -> dict:
    from . import race_total_model as v1
    from . import race_features_model as variant
    from .race_targets import target
    snapshot = {str(p): digest(p) for p in sources(feature)}
    x_car, x_drv = covariates(feature)
    C = v1.laps()
    entries = pd.read_parquet(PROCESSED / "race.parquet")
    car_rows, driver_rows, checks = [], [], {}
    for end in FOLDS:
        season = int(end[:4])
        train = C[C.event_id <= end]
        test_events = sorted(set(C.loc[(C.event_id > end) & (C.season == season), "event_id"]))
        if len(test_events) < 3:
            raise ValueError(f"Insufficient held-out dry races after {end}")
        fits = {"full": variant.fit(train, f"{feature}_{end}", x_car, x_drv),
                "v1": v1.fit(train, f"{end}_full", driver=True, car=True)}
        checks[end] = {m: {k: meta[k] for k in ("key", "diagnostics", "data_as_of", "n_laps", "n_races")
                           if k in meta} | {"attempts": meta.get("attempts")} for m, (_, meta) in fits.items()}
        post, meta = fits["full"]
        checks[end]["full"]["beta"] = {
            name: dict(mean=float(post[key][:, j].mean()), q05=float(np.quantile(post[key][:, j], .05)),
                       q95=float(np.quantile(post[key][:, j], .95)))
            for key, names in (("beta_car", meta["catalog"]["car_covariates"]),
                               ("beta_drv", meta["catalog"]["driver_covariates"]))
            for j, name in enumerate(names)}
        for event in test_events:
            t, target_draws = target(C[C.event_id == event])
            t = t.assign(event_id=event).merge(entries[["event_id", "driver_id", "team"]],
                          on=["event_id", "driver_id"], validate="one_to_one").sort_values("driver_id").reset_index(drop=True)
            pred = {"full": variant.predict(*fits["full"], t, x_car, x_drv, noise=True, seed=season + 1),
                    "v1": v1.predict(*fits["v1"], t, noise=True, seed=season + 1)}
            point = {"full": point_prediction(*fits["full"], t)
                     + variant.covariate_effect(*fits["full"], t, x_car, x_drv, mean=True),
                     "v1": point_prediction(*fits["v1"], t)}
            point = {m: x - x.mean() for m, x in point.items()}
            pred = {m: x - x.mean(1, keepdims=True) for m, x in pred.items()}
            error = target_draws - target_draws.mean(0, keepdims=True)
            observed_pred = {m: x + error[np.random.default_rng(1).integers(len(error), size=len(x))]
                             for m, x in pred.items()}

            def add(row, idx, combine):
                for m in ("full", "v1"):
                    x = combine(observed_pred[m], idx)
                    lo, hi = np.quantile(x, [.05, .95])
                    row[m] = float(combine(point[m][None], idx)[0])
                    row["lo" if m == "full" else m + "_lo"] = float(lo)
                    row["hi" if m == "full" else m + "_hi"] = float(hi)
                return row
            for team, tteam in t.groupby("team"):
                idx = tteam.index.to_numpy()
                car_rows.append(add(dict(event_id=event, team=team, observed=float(tteam.pace.mean())),
                                    idx, lambda x, i: x[:, i].mean(1)))
                if x_drv is not None and len(idx) == 2:
                    ia, ib = sorted(idx, key=lambda i: t.driver_id[i])
                    driver_rows.append(add(dict(event_id=event, a=t.driver_id[ia], b=t.driver_id[ib],
                                                observed=float(t.pace[ia] - t.pace[ib])),
                                           (ia, ib), lambda x, i: x[:, i[0]] - x[:, i[1]]))
        print(f"Validated {end}: {len(test_events)} held-out races", flush=True)
    tables = {"cars": improvement(car_rows, "v1")}
    if driver_rows:
        tables["drivers"] = improvement(driver_rows, "v1")
    for result in tables.values():
        result["passed"] = bool(result["n_races"] >= 12 and result["improves"] and result["calibrated"])
    out = OUT / feature
    summary = dict(feature=feature, design="Pre-registered in docs/race_features.md; v1 folds, targets and rows; "
                   "baseline: v1's full model", folds=list(FOLDS), fits=checks, metrics=tables,
                   gate="n_races >= 12, MSE difference (variant - v1) 95% CI below 0, coverage 85-95%",
                   units="percent of lap time; MSE in percent squared")
    if {str(p): digest(p) for p in sources(feature)} != snapshot:
        raise RuntimeError("Inputs or code changed during the run; rerun")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "validation.json", summary)
    files = [out / "validation.json", out / "car_predictions.csv"]
    pd.DataFrame(car_rows).to_csv(files[1], index=False)
    if driver_rows:
        files.append(out / "driver_predictions.csv")
        pd.DataFrame(driver_rows).to_csv(files[2], index=False)
    record(out, files, model=f"race-feature-{feature}-validation-v1", inputs=sources(feature),
           name="validation.manifest.json")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "validate"])
    p.add_argument("feature", nargs="?", choices=FEATURES)
    args = p.parse_args()
    if args.command == "build":
        build()
    else:
        if args.feature is None:
            p.error("validate needs a feature")
        print(json.dumps(validate(args.feature)["metrics"], indent=1))


if __name__ == "__main__":
    main()
