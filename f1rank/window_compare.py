"""Compare the 2010 and 2006 data windows on identical forecast targets.

    python -m f1rank.window_compare

Uses the lfo_* (2010 window) and lfo2006_* fits that exist for the same cutoff.
Writes outputs/validation/window_compare.json.
"""

import json

import numpy as np
import pandas as pd

from .design import build_design
from .evaluate import REPORTS, SEC_PER_PCT, evaluate_cutoff
from .fit import FITS
from .jobs import lfo_cutoffs


def main() -> None:
    designs = {2010: build_design(2010), 2006: build_design(2006)}
    cuts = {w: lfo_cutoffs(d) for w, d in designs.items()}
    names = sorted(f.stem[len("lfo2006_"):] for f in FITS.glob("lfo2006_*.npz"))
    pairs, sessions = [], []
    for n in names:
        for w, fit_name in ((2010, f"lfo_{n}"), (2006, f"lfo2006_{n}")):
            r = evaluate_cutoff(designs[w], f"lfo_{n}", cuts[w][f"lfo_{n}"], np.random.default_rng(0),
                                fit_file=FITS / f"{fit_name}.npz")
            pairs += [{**p, "window": w, "cut": n} for p in r["pairs"]]
            sessions.append({"window": w, "cut": n, **r["teammate"],
                             "pace_rmse": r["pace"]["rmse_pred"], "pace_spearman": r["pace"]["spearman_pred"]})
    p = pd.DataFrame(pairs)
    key = ["cut", "driver_a", "driver_b"]
    m = p[p.window == 2010].merge(p[p.window == 2006], on=key, suffixes=("_10", "_06"))
    s = pd.DataFrame(sessions)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) * SEC_PER_PCT)  # noqa: E731
    out = {"cutoffs": names, "n_pairings": int(len(m))}
    for w, suf in ((2010, "_10"), (2006, "_06")):
        new = m[m["new_pair" + suf]]
        out[str(w)] = {
            "pairing_rmse_s": rmse(m["gap" + suf], m["pred" + suf]),
            "pairing_new_rmse_s": rmse(new["gap" + suf], new["pred" + suf]) if len(new) else None,
            "pairing_in90": float(m["in90" + suf].mean()),
            "session_rmse_s": float(s[s.window == w].rmse_pred.mean() * SEC_PER_PCT),
            "session_crps_s": float(s[s.window == w].crps.mean() * SEC_PER_PCT),
            "pace_rmse_s": float(s[s.window == w].pace_rmse.mean() * SEC_PER_PCT),
            "pace_spearman": float(s[s.window == w].pace_spearman.mean()),
        }
    (REPORTS / "window_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
