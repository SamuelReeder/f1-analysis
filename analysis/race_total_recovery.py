"""Diagnostic recovery on the observed team-switch network, three independent truths.

python -m analysis.race_total_recovery

The first and last four clean laps per driver-race keep this check manageable
while retaining adjacent laps that identify the serial correlation. The
model generates the nuisance effects; residuals follow its stated Student-t AR(1)
likelihood. This checks implementation and identifiability under the model, not
whether real F1 confounding follows that model. It is separate from the real-data
publication gates fixed in race_total.py.
"""
import json

import jax
import numpy as np
import numpyro.handlers as handlers
import pandas as pd

from f1rank import race_total_model as m
from f1rank.artifacts import atomic_json


def simulate(C, seed):
    d, catalog = m.design(C)
    trace = handlers.trace(handlers.seed(m.model, jax.random.PRNGKey(seed + 100))).get_trace(d)
    truth = {k: np.asarray(v["value"]) for k, v in trace.items() if v["type"] in ("sample", "deterministic")}
    a = {k: np.asarray(v) for k, v in d.items()}
    mu = (truth["intercept"][a["race"]] + truth["trend"][a["race"]] * a["lap"]
          + np.where(a["nonref"], truth["compound"][a["rc"]], 0)
          + (truth["slope"][a["rc"]] + truth["wear"][a["dr"]]) * a["age"]
          + (truth["traffic"][a["race"]] * a["traffic"]).sum(-1)
          + truth["day"][a["dr"]] + truth["car_day"][a["cr"]]
          + truth["driver_pace"][a["ds"]] + truth["package"][a["car"]])
    rng = np.random.default_rng(seed)
    r = np.zeros(len(C))
    rho, sigma, nu = (float(truth[k]) for k in ("rho", "sigma", "nu"))
    for i in range(len(C)):
        r[i] = (rho * r[i - 1] if a["prev"][i] else 0) + rng.standard_t(nu) * sigma / (
            1 if a["prev"][i] else np.sqrt(1 - rho ** 2))
    return C.assign(y=mu + r), truth, catalog


def compare(draws, truth):
    draws = draws - draws.mean(1, keepdims=True)
    truth = truth - truth.mean()
    lo, hi = np.quantile(draws, [.05, .95], axis=0)
    median = np.median(draws, axis=0)
    corr = np.corrcoef(median.argsort().argsort(), truth.argsort().argsort())[0, 1]
    return dict(n=len(truth), rank_correlation=float(corr),
                coverage90=float(np.mean((truth >= lo) & (truth <= hi))),
                rmse=float(np.sqrt(np.mean((median - truth) ** 2))),
                mean_width90=float(np.mean(hi - lo)))


def main():
    C = m.laps()
    selected = [np.unique(np.r_[np.arange(min(4, len(g))), np.arange(max(0, len(g) - 4), len(g))])
                for _, g in C.groupby(["event_id", "driver_id"], sort=True)]
    C = pd.concat([g.iloc[index] for (_, g), index in zip(
        C.groupby(["event_id", "driver_id"], sort=True), selected)], ignore_index=True)
    C = C.sort_values(["event_id", "stint_key", "lap_number"], ignore_index=True)
    entries = pd.read_parquet(m.PROCESSED / "race.parquet")
    grid = entries[entries.event_id == C.event_id.max()].sort_values("driver_id")
    results = {}
    for seed in (0, 1, 2):
        fake, truth, cat = simulate(C, seed)
        post, meta = m.fit(fake, f"recovery_{seed}", warmup=500, samples=500)
        driver = m.predict({k: v for k, v in post.items() if k != "package"}, meta, grid)
        cars = grid.drop_duplicates("team").sort_values("team")
        car = m.predict({k: v for k, v in post.items() if k not in ("skill", "form")}, meta, cars)
        sd = {name: truth["skill"][i] for i, name in enumerate(cat["drivers"])}
        ds = {name: truth["form"][i] for i, name in enumerate(cat["driver_seasons"])}
        cs = {name: truth["package"][i] for i, name in enumerate(cat["cars"])}
        season = C.event_id.max()[:4]
        driver_truth = np.array([sd[r.driver_id] + ds[season + "|" + r.driver_id] for r in grid.itertuples()])
        car_truth = np.array([cs[season + "|" + r.team] for r in cars.itertuples()])
        results[seed] = dict(drivers=compare(driver, driver_truth), cars=compare(car, car_truth),
                             diagnostics=meta["diagnostics"], fit_key=meta["key"], laps=len(fake))
        atomic_json(m.ROOT / "outputs/race_total/recovery.json", {"design": __doc__, "seeds": results})
        print(json.dumps(results[seed], indent=1), flush=True)


if __name__ == "__main__":
    main()
