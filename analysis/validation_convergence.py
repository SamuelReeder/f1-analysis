"""Convergence of the qualifying validation fits made before the 2026-10 retry rule.

    .venv/bin/python -m analysis.validation_convergence

Applies the main fit's check (report.convergence: R-hat below 1.05 for skill, the
team-specific effect, car and the hyperparameters; at most one divergence per 1,000
draws) to every lfo_* and sens_* fit in outputs/fits/ that was made by the old
single-attempt rule (no retry history in its metadata), and splits the car and skill
R-hat by where each state lies: training period, test season, or later seasons that
have no data in the fit and are not forecast. Those fits are replaced when the
validation is rerun, so the result is kept in
outputs/analysis/validation_convergence/before_retry_rule.json.
"""
import json
from pathlib import Path

import numpy as np
from numpyro.diagnostics import split_gelman_rubin

from f1rank.fit import FITS, load, load_meta
from f1rank.report import convergence

OUT = Path(__file__).resolve().parent.parent / "outputs" / "analysis" / "validation_convergence"


def by_period(post: dict, meta: dict, name: str) -> dict:
    cut = meta["last_train_event"]
    test_season = int(cut[:4]) + (1 if name.startswith("lfo_end") else 0)
    out = {}
    for site, key, position in (("car", "cars", 1), ("skill", "entries", 0)):
        events = np.array([i.split("|")[position] for i in meta["ids"][key]])
        seasons = np.array([int(e[:4]) for e in events])
        rhat = np.asarray(split_gelman_rubin(post[site]))
        periods = {"train": events <= cut, "test_season": (events > cut) & (seasons == test_season),
                   "later": seasons > test_season}
        out[site] = {k: round(float(rhat[m].max()), 3) if m.any() else None for k, m in periods.items()}
    return out


def main() -> None:
    rows = {}
    for path in sorted([*FITS.glob("lfo_*.npz"), *FITS.glob("sens_*.npz")]):
        meta = load_meta(path)
        if meta is None or "attempts" in meta:
            continue  # made under the retry rule
        name = path.stem
        try:
            worst, ok = convergence(path)
        except KeyError as exc:  # a variant without that parameter (sens_nocompat)
            rows[name] = {"checked": False, "reason": f"no {exc} parameter"}
            continue
        row = {"checked": True, "passes": ok, "divergences": worst["info"]["divergences"],
               "n_draws": worst["n_draws"],
               **{f"rhat_{k}": round(worst[k]["rhat_max"], 3) for k in ("skill", "compat", "car")},
               "rhat_hyper": round(worst["hyper_rhat_max"], 3), "created_utc": meta["created_utc"]}
        if meta.get("last_train_event"):
            row["rhat_by_period"] = by_period(load(path)[0], meta, name)
        rows[name] = row
        print(name, row, flush=True)
    checked = [r for r in rows.values() if r["checked"]]
    summary = {"n_checked": len(checked), "n_fail": sum(not r["passes"] for r in checked),
               "fits": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "before_retry_rule.json").write_text(json.dumps(summary, indent=1))
    print({k: v for k, v in summary.items() if k != "fits"})


if __name__ == "__main__":
    main()
