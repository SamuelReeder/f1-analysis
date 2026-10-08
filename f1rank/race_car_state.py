"""Validate the pre-registered within-season car state (docs/race_car_state.md).

python -m f1rank.race_car_state --validate

Same folds, targets, team-level rows, baseline and gate as the v1 car table in
race_total.validate; only the full model differs.
"""
import argparse

import numpy as np
import pandas as pd

from .artifacts import atomic_json, record
from .race_total import FOLDS, ROOT, improvement, inputs, point_prediction

OUT = ROOT / "outputs" / "race_car_state"


def point(post, meta, entries):
    """Conditional mean, as for v1: the team's last state; the walk has mean zero."""
    last = {}
    for i, name in enumerate(meta["catalog"]["car_races"]):
        event, team = name.split("|")
        last[event[:4] + "|" + team] = i
    drift = post["drift"].mean(0)
    return point_prediction(post, meta, entries) + np.array(
        [drift[last[k]] if k in last else 0. for k in entries.event_id.str[:4] + "|" + entries.team])


def validate():
    from . import race_total_model as v1
    from .race_car_state_model import MODEL, fit, predict
    from .race_targets import target
    C = v1.laps()
    entries = pd.read_parquet(ROOT / "data/processed/race.parquet")
    rows, checks = [], {}
    for end in FOLDS:
        season = int(end[:4])
        train = C[C.event_id <= end]
        test_events = sorted(set(C.loc[(C.event_id > end) & (C.season == season), "event_id"]))
        if len(test_events) < 3:
            raise ValueError(f"Insufficient held-out dry races after {end}")
        steps = {event: k for k, event in enumerate(test_events, 1)}
        full = fit(train, f"{end}_full")
        baseline = v1.fit(train, f"{end}_no_car", driver=True, car=False)
        checks[end] = {name: {k: meta[k] for k in ("key", "diagnostics", "attempts", "data_as_of", "n_laps", "n_races")}
                       for name, (_, meta) in (("full", full), ("no_car", baseline))}
        for event in test_events:
            t, target_draws = target(C[C.event_id == event])
            t = t.assign(event_id=event).merge(entries[["event_id", "driver_id", "team"]],
                          on=["event_id", "driver_id"], validate="one_to_one").sort_values("driver_id").reset_index(drop=True)
            pred = {"full": predict(*full, t, steps, noise=True, seed=season + 1),
                    "no_car": v1.predict(*baseline, t, noise=True, seed=season + 1)}
            mean = {"full": point(*full, t), "no_car": point_prediction(*baseline, t)}
            points = {m: x - x.mean() for m, x in mean.items()}
            pred = {m: x - x.mean(1, keepdims=True) for m, x in pred.items()}
            error = target_draws - target_draws.mean(0, keepdims=True)
            observed_pred = {m: x + error[np.random.default_rng(1).integers(len(error), size=len(x))]
                             for m, x in pred.items()}
            for team, tteam in t.groupby("team"):
                idx = tteam.index.to_numpy()
                row = dict(event_id=event, team=team, k=steps[event], observed=float(tteam.pace.mean()))
                for m in ("full", "no_car"):
                    x = observed_pred[m][:, idx].mean(1)
                    lo, hi = np.quantile(x, [.05, .95])
                    row[m] = float(points[m][idx].mean())
                    row["lo" if m == "full" else m + "_lo"] = float(lo)
                    row["hi" if m == "full" else m + "_hi"] = float(hi)
                rows.append(row)
        print(f"Validated {end}: {len(test_events)} held-out races", flush=True)
    result = improvement(rows, "no_car")
    result["passed"] = bool(result["n_races"] >= 12 and result["improves"]
                            and result["calibrated"] and result["sharper"])
    summary = dict(model=MODEL, design="Pre-registered in docs/race_car_state.md; v1 folds, targets, baseline and gate",
                   folds=list(FOLDS), fits=checks, cars=result,
                   units="percent of lap time; MSE in percent squared")
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "validation.json", summary)
    pd.DataFrame(rows).to_csv(OUT / "car_predictions.csv", index=False)
    record(OUT, [OUT / "validation.json", OUT / "car_predictions.csv"], model="total-dry-race-pace-carstate-validation-v1",
           inputs=inputs() + [ROOT / "f1rank/race_car_state_model.py", ROOT / "f1rank/race_car_state.py",
                              ROOT / "docs/race_car_state.md"], name="validation.manifest.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", required=True)
    parser.parse_args()
    print(validate()["cars"])
