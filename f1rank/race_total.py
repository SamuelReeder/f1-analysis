"""Fit, validate and export total driver/car race pace.

python -m f1rank.race_total --validate
python -m f1rank.race_total --export

The fixed outer tests train through rounds 10 of 2024/2025 and round 7 of
2026, and predict the remaining dry races in that season. Each ablation is
refitted on the same training laps. Qualifying data do not enter this model.

The export also writes season-by-season estimates (revised with all data) and a
write-once snapshot, outputs/snapshots/race/<event>_<model>_<fit id>.json, of the
tables as published for each fit. Failed tables are withheld from both.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import atomic_json, digest, record, require, require_convergence, write_once

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "race_total"
SNAPSHOTS = ROOT / "outputs" / "snapshots" / "race"
FOLDS = ("2024-10", "2025-10", "2026-07")
BOOTSTRAPS = 4000


def inputs():
    return [ROOT / p for p in (
        "f1rank/race_total_model.py", "f1rank/race_total.py", "f1rank/race_publication.py", "f1rank/race_targets.py",
        "f1rank/racepace.py", "f1rank/artifacts.py",
        "data/processed/race_laps.parquet", "data/processed/race_weather.parquet",
        "data/processed/timeline.parquet", "data/processed/entries.parquet",
        "data/processed/race.parquet", "data/processed/race_classification.parquet",
        "data/processed/events.parquet", "data/processed/drivers.parquet")]


def source_snapshot():
    return {str(path): digest(path) for path in inputs()}


def check_snapshot(snapshot):
    if source_snapshot() != snapshot:
        raise RuntimeError("Race inputs or scoring code changed during the run; rerun before publishing")


def improvement(rows, baseline):
    """Race-block bootstrap: laps and teammates never count as independent trials."""
    rows = pd.DataFrame(rows)
    d = rows.groupby("event_id").apply(
        lambda g: float(np.mean((g.observed - g.full) ** 2 - (g.observed - g[baseline]) ** 2)),
        include_groups=False).to_numpy()
    rng = np.random.default_rng(0)
    boot = d[rng.integers(len(d), size=(BOOTSTRAPS, len(d)))].mean(1)
    interval = np.quantile(boot, [.025, .975])
    coverage = float(((rows.observed >= rows.lo) & (rows.observed <= rows.hi)).mean())
    width = float((rows.hi - rows.lo).mean())
    baseline_width = float((rows[baseline + "_hi"] - rows[baseline + "_lo"]).mean())
    # Calibration counts predictions, whereas the improvement CI resamples races.
    return dict(n_races=len(d), n_predictions=len(rows), mse_difference=float(d.mean()),
                mse_difference_ci95=interval.tolist(), improves=bool(interval[1] < 0),
                rmse=float(np.sqrt(np.mean((rows.observed - rows.full) ** 2))),
                baseline_rmse=float(np.sqrt(np.mean((rows.observed - rows[baseline]) ** 2))),
                coverage90=coverage, mean_interval_width=width, baseline_interval_width=baseline_width,
                calibrated=bool(.85 <= coverage <= .95), sharper=bool(width < baseline_width))


def point_prediction(post, meta, entries):
    """Conditional means: future random effects and unseen states have mean zero."""
    catalog = meta["catalog"]
    maps = {key: dict(zip(catalog[names], post[key].mean(0))) if key in post else {}
            for key, names in (("skill", "drivers"), ("form", "driver_seasons"), ("package", "cars"))}
    return np.array([maps["skill"].get(r.driver_id, 0.)
                     + maps["form"].get(r.event_id[:4] + "|" + r.driver_id, 0.)
                     + maps["package"].get(r.event_id[:4] + "|" + r.team, 0.) for r in entries.itertuples()])


def validate():
    snapshot = source_snapshot()
    from .race_total_model import laps, fit, predict
    from .race_targets import target
    C = laps()
    entries = pd.read_parquet(ROOT / "data/processed/race.parquet")
    driver_rows, car_rows, checks = [], [], {}
    for end in FOLDS:
        season = int(end[:4])
        train = C[C.event_id <= end]
        test_events = sorted(set(C.loc[(C.event_id > end) & (C.season == season), "event_id"]))
        if len(test_events) < 3:
            raise ValueError(f"Insufficient held-out dry races after {end}")
        fits = {}
        checks[end] = {}
        for mode in ("full", "no_driver", "no_car"):
            post, meta = fit(train, f"{end}_{mode}", driver=mode != "no_driver", car=mode != "no_car")
            fits[mode] = (post, meta)
            checks[end][mode] = {k: meta[k] for k in ("key", "diagnostics", "data_as_of", "n_laps", "n_races")}
        for event in test_events:
            t, target_draws = target(C[C.event_id == event])
            t = t.assign(event_id=event).merge(entries[["event_id", "driver_id", "team"]],
                          on=["event_id", "driver_id"], validate="one_to_one").sort_values("driver_id").reset_index(drop=True)
            pred = {m: predict(*fits[m], t, noise=True, seed=season + 1) for m in fits}
            point = {m: point_prediction(*fits[m], t) for m in fits}
            point = {m: x - x.mean() for m, x in point.items()}
            # Stage A pace estimates are centred on this race's eligible drivers.
            pred = {m: x - x.mean(1, keepdims=True) for m, x in pred.items()}
            by_id = {d: i for i, d in enumerate(t.driver_id)}
            # Resample entire bootstrap rows: compound/fuel fits induce shared
            # errors across entrants, so independent marginal SEs are inadequate.
            error = target_draws - target_draws.mean(0, keepdims=True)
            observed_pred = {m: x + error[np.random.default_rng(1).integers(len(error), size=len(x))]
                             for m, x in pred.items()}
            for _, team_entries in t.groupby("team"):
                if len(team_entries) != 2:
                    continue
                a, b = sorted(team_entries.driver_id)
                ia, ib = by_id[a], by_id[b]
                row = dict(event_id=event, a=a, b=b, observed=float(t.pace.iloc[ia] - t.pace.iloc[ib]))
                for m in ("full", "no_driver"):
                    x = observed_pred[m][:, ia] - observed_pred[m][:, ib]
                    lo, hi = np.quantile(x, [.05, .95])
                    row[m] = float(point[m][ia] - point[m][ib])
                    row["lo" if m == "full" else m + "_lo"] = float(lo)
                    row["hi" if m == "full" else m + "_hi"] = float(hi)
                driver_rows.append(row)
            for team, tteam in t.groupby("team"):
                idx = tteam.index.to_numpy()
                # Between-team test uses observed team pace, not a target constructed
                # by subtracting the model's own driver ratings.
                row = dict(event_id=event, team=team, observed=float(tteam.pace.mean()))
                for m in ("full", "no_car"):
                    x = observed_pred[m][:, idx].mean(1)
                    lo, hi = np.quantile(x, [.05, .95])
                    row[m] = float(point[m][idx].mean())
                    row["lo" if m == "full" else m + "_lo"] = float(lo)
                    row["hi" if m == "full" else m + "_hi"] = float(hi)
                car_rows.append(row)
        print(f"Validated {end}: {len(test_events)} held-out races", flush=True)
    results = {"drivers": improvement(driver_rows, "no_driver"), "cars": improvement(car_rows, "no_car")}
    for result in results.values():
        result["passed"] = bool(result["n_races"] >= 12 and result["improves"]
                                and result["calibrated"] and result["sharper"])
    summary = dict(design="Later races of the same season, excluded from all fitting; refitted ablations",
                   folds=list(FOLDS), fits=checks, metrics=results,
                   units="percent of lap time; MSE in percent squared",
                   calibration_note="Whole Stage-A bootstrap rows preserve shared target uncertainty across drivers and teams.")
    OUT.mkdir(parents=True, exist_ok=True)
    check_snapshot(snapshot)
    atomic_json(OUT / "validation.json", summary)
    pd.DataFrame(driver_rows).to_csv(OUT / "driver_predictions.csv", index=False)
    pd.DataFrame(car_rows).to_csv(OUT / "car_predictions.csv", index=False)
    check_snapshot(snapshot)
    record(OUT, [OUT / n for n in ("validation.json", "driver_predictions.csv", "car_predictions.csv")],
           model="total-dry-race-pace-validation-v1", inputs=inputs(), name="validation.manifest.json")
    return summary


def estimates(draws):
    if draws.ndim != 2 or not np.isfinite(draws).all() or not draws.shape[1]:
        raise ValueError("Invalid race pace draws")
    centred = (draws - draws.mean(1, keepdims=True)) * .9
    ranks = (-centred).argsort(1).argsort(1) + 1
    q = np.quantile(centred, [.05, .25, .5, .75, .95], axis=0)
    result = []
    for i in range(centred.shape[1]):
        result.append(dict(zip(("q05", "q25", "median", "q75", "q95"), q[:, i].tolist()),
                      rank_lo=int(np.quantile(ranks[:, i], .05, method="lower")),
                      rank_hi=int(np.quantile(ranks[:, i], .95, method="higher")),
                      p_fastest=float((ranks[:, i] == 1).mean()), p_top3=float((ranks[:, i] <= 3).mean())))
    return result


def season_rows(C, post, meta, names, classification, grid=None):
    """Estimates for one season's drivers and cars with at least two clean dry races.

    A past season is centred on its own eligible field. The current season uses the
    latest race's grid, exactly as the headline table does.
    """
    from .race_total_model import predict
    season = int(C.season.iloc[0])
    if grid is None:  # each driver with the team of their last clean race that season
        grid = C.sort_values("event_id").groupby("driver_id").tail(1)[["event_id", "driver_id", "team"]]
        grid = grid.merge(classification[["event_id", "driver_id", "team_name"]], on=["event_id", "driver_id"],
                          how="left", validate="one_to_one").rename(columns={"team_name": "constructor_name"})
        grid = grid.assign(event_id=C.event_id.max())  # predictions use the season's states
    races = C.groupby("driver_id").event_id.nunique()
    team_races = C.groupby("team").event_id.nunique()
    drivers = grid[grid.driver_id.isin(races[races >= 2].index)].sort_values("driver_id")
    cars = grid.drop_duplicates("team").sort_values("team")
    cars = cars[cars.team.isin(team_races[team_races >= 2].index)]
    out = {"drivers": [], "cars": []}
    for kind, rows, keep in (("drivers", drivers, lambda k: k != "package"),
                             ("cars", cars, lambda k: k not in ("skill", "form"))):
        if not len(rows):
            continue
        draws = predict({k: v for k, v in post.items() if keep(k)}, meta, rows)
        for r, estimate in zip(rows.itertuples(), estimates(draws)):
            obs = C[C.driver_id == r.driver_id] if kind == "drivers" else C[C.team == r.team]
            if kind == "drivers":
                row = dict(id=r.driver_id, name=names.loc[r.driver_id, "name"],
                           code=names.loc[r.driver_id, "code"], lineage=r.team)
                row["team"] = r.constructor_name if isinstance(r.constructor_name, str) else r.team
            else:
                row = dict(id=r.team, name=r.constructor_name if isinstance(r.constructor_name, str) else r.team)
            out[kind].append(dict(row, pace=estimate, races=int(obs.event_id.nunique()),
                                  laps=int(len(obs)), last_race=str(obs.event_id.max())))
        out[kind].sort(key=lambda row: -row["pace"]["median"])
    return season, out


def export():
    snapshot = source_snapshot()
    from .race_total_model import laps, fit, predict
    from .race_publication import check
    C = laps()
    post, meta = fit(C, "full")
    require_convergence(meta["diagnostics"])
    validation_manifest = require(OUT, name="validation.manifest.json", required_outputs=[OUT / "validation.json"])
    validation = json.loads((OUT / "validation.json").read_text())
    events = pd.read_parquet(ROOT / "data/processed/events.parquet")
    as_of = events[events.event_id == meta["data_as_of"]].iloc[0].to_dict()
    entries = pd.read_parquet(ROOT / "data/processed/race.parquet")
    grid_event = str(entries.event_id.max())
    grid = entries[entries.event_id == grid_event].sort_values("driver_id").copy()
    classification = pd.read_parquet(ROOT / "data/processed/race_classification.parquet")
    grid = grid.merge(classification[["event_id", "driver_id", "team_name"]],
                      on=["event_id", "driver_id"], how="left", validate="one_to_one").rename(columns={"team_name": "constructor_name"})
    grid["constructor_name"] = grid.constructor_name.fillna(grid.constructor_id.str.replace("_", " ").str.title())
    names = pd.read_parquet(ROOT / "data/processed/drivers.parquet").set_index("driver_id")
    current = C[C.season == int(grid_event[:4])]
    eligible = set(current.groupby("driver_id").event_id.nunique().loc[lambda x: x >= 2].index)
    eligible_cars = set(current.groupby("team").event_id.nunique().loc[lambda x: x >= 2].index)
    _, now = season_rows(current, post, meta, names, classification, grid)
    driver_rows, car_rows = now["drivers"], now["cars"]
    passed = {k: validation["metrics"][k]["passed"] for k in ("drivers", "cars")}
    # Revised estimates for every season; a table that failed its gate is withheld here too.
    seasons = []
    for season, S in C.groupby("season"):
        if season == int(grid_event[:4]):
            rows = now
        else:
            _, rows = season_rows(S, post, meta, names, classification)
        seasons.append(dict(season=int(season), **{k: rows[k] if passed[k] else [] for k in rows}))
    data = dict(schema_version=1, model=meta["model"], fit_id=meta["key"][:20],
                data_as_of=as_of, grid_as_of=grid_event, first_event=meta["first_event"], season=int(grid_event[:4]),
                n_laps=meta["n_laps"], n_races=meta["n_races"], diagnostics=meta["diagnostics"],
                validation=validation, validation_fit_id=validation_manifest["fit_id"],
                drivers=driver_rows if passed["drivers"] else [],
                cars=car_rows if passed["cars"] else [],
                seasons=seasons,
                unrated_drivers=grid.loc[~grid.driver_id.isin(eligible), "driver_id"].tolist(),
                unrated_cars=grid.loc[~grid.team.isin(eligible_cars), "team"].drop_duplicates().tolist())
    check(data)
    check_snapshot(snapshot)
    # Inspectable research values remain separate from the gated public tables.
    atomic_json(OUT / "research_estimates.json", {"drivers": driver_rows, "cars": car_rows, "fit": meta})
    atomic_json(OUT / "pace.json", data)
    check_snapshot(snapshot)
    manifest = record(OUT, [OUT / "pace.json"], model=meta["model"],
                      inputs=inputs() + [OUT / "validation.json", OUT / "validation.manifest.json"],
                      details={"posterior_key": meta["key"], "validation_fit_id": validation_manifest["fit_id"]})
    snap = dict(model=data["model"], fit_id=data["fit_id"], data_as_of=data["data_as_of"],
                grid_as_of=grid_event, generated_at=manifest["generated_at"],
                validation_fit_id=data["validation_fit_id"], passed=passed,
                drivers=data["drivers"], cars=data["cars"])
    path = SNAPSHOTS / f"{grid_event}_{data['model']}_{data['fit_id']}.json"
    print(("snapshot written: " if write_once(path, snap) else "snapshot already exists: ") + path.name)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--export", action="store_true")
    args = p.parse_args()
    if not args.validate and not args.export:
        p.error("Select --validate and/or --export")
    if args.validate:
        validate()
    if args.export:
        export()
