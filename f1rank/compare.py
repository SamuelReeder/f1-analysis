"""Compare forecasting variants with the main model on identical forecast targets.

    python -m f1rank.compare

Variants (jobs.LFO_VARIANTS), each fitted at some of the leave-future-out cutoffs:
  lfo2006   the data window starting in 2006 instead of 2010
  lfospell  one team-specific effect per spell with a team, instead of per lineage
  lfoera    one team-specific effect per regulation era with a team

At each cutoff the main model's lfo_* fit and the variant forecast the same teammate
pairings and segments. The difference in squared error per matched pairing (variant
minus main; negative = variant better) gets a bootstrap 95% interval over pairings.
Pairings at the same cutoff share a fit, so the interval is somewhat too narrow. The
`after_reset` subset is the first season after a regulation reset (2017, 2022, 2026),
where the per-era variant starts every team-specific effect afresh.

Writes outputs/validation/compare.json.
"""

import json

import numpy as np
import pandas as pd

from .design import build_design
from .evaluate import REPORTS, SEC_PER_PCT, evaluate_cutoff
from .fit import FITS, common_data_as_of
from .jobs import LFO_VARIANTS, lfo_cutoffs
from .lineage import REGULATION_RESETS

N_BOOT = 2000


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)) * SEC_PER_PCT)


def paired(m: pd.DataFrame, rng) -> dict:
    """Variant vs main on matched pairings: RMSEs and the mean difference in squared error
    (s^2) with a bootstrap 95% interval over pairings."""
    if not len(m):
        return {"n": 0}
    d = ((m.gap - m.pred_var) ** 2 - (m.gap - m.pred_main) ** 2).to_numpy() * SEC_PER_PCT ** 2
    boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": int(len(m)), "rmse_main_s": _rmse(m.gap, m.pred_main), "rmse_variant_s": _rmse(m.gap, m.pred_var),
            "in90_main": float(m.in90_main.mean()), "in90_variant": float(m.in90_var.mean()),
            "mse_diff_s2": float(d.mean()), "mse_diff_ci95": [float(lo), float(hi)]}


def compare(variant: str, rng) -> dict | None:
    spec = LFO_VARIANTS[variant]
    cuts = [c for c in spec["cuts"] if (FITS / f"{variant}_{c}.npz").exists()]
    if not cuts:
        return None
    as_of = common_data_as_of([FITS / f"{w}_{c}.npz" for c in cuts for w in ("lfo", variant)])
    d_main = build_design(2010, end_event=as_of)
    d_var = build_design(spec.get("start", 2010), end_event=as_of)
    pairs, sessions = [], []
    for c in cuts:
        for label, design, fit_name in (("main", d_main, f"lfo_{c}"), ("var", d_var, f"{variant}_{c}")):
            r = evaluate_cutoff(design, f"lfo_{c}", lfo_cutoffs(design)[f"lfo_{c}"], np.random.default_rng(0),
                                fit_file=FITS / f"{fit_name}.npz")
            pairs += [{**p, "fit": label, "cut": c, "test_season": r["test_season"]} for p in r["pairs"]]
            sessions.append({"fit": label, "cut": c, **r["teammate"],
                             "pace_rmse": r["pace"]["rmse_pred"], "pace_spearman": r["pace"]["spearman_pred"]})
    p = pd.DataFrame(pairs)
    key = ["cut", "driver_a", "driver_b"]
    m = p[p.fit == "main"].merge(p[p.fit == "var"][key + ["pred", "in90"]], on=key, suffixes=("_main", "_var"))
    s = pd.DataFrame(sessions)
    out = {"cutoffs": cuts, "model_kw": {k: repr(v) for k, v in spec.get("model_kw", {}).items()},
           "start": spec.get("start", 2010),
           "pairings": paired(m, rng), "pairings_new": paired(m[m.new_pair], rng),
           "pairings_after_reset": paired(m[m.test_season.isin(REGULATION_RESETS)], rng)}
    for label in ("main", "var"):
        g = s[s.fit == label]
        out[f"sessions_{label}"] = {"rmse_s": float(g.rmse_pred.mean() * SEC_PER_PCT),
                                    "crps_s": float(g.crps.mean() * SEC_PER_PCT), "cov90": float(g.cov90.mean()),
                                    "pace_rmse_s": float(g.pace_rmse.mean() * SEC_PER_PCT),
                                    "pace_spearman": float(g.pace_spearman.mean())}
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    out = {v: r for v in LFO_VARIANTS if (r := compare(v, rng)) is not None}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
