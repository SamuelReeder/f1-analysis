"""Posterior checks on the main fit.

    python -m f1rank.diagnostics [fit_name]

teammate residual correlation  residuals of two teammates in the same segment
                               should be uncorrelated if shared car effects are
                               fully captured; positive correlation means a
                               segment-level team effect is missing
residual autocorrelation       a driver's residual teammate gap should not
                               persist across events if skill dynamics are right
"""

import ast
import sys

import numpy as np
import pandas as pd

from .fit import FITS, design_for, load, load_meta
from .ratings import flat


def fitted_terms(car_track=True, car_transient=True, car_segment=True, driver_form=True,
                 compat=True, placebo=False, **_) -> dict[str, str]:
    """Terms in the model's mean for these model options, with the state each is indexed by."""
    if placebo:
        raise ValueError("residuals are not defined for the placebo diagnostic (its term is not saved)")
    terms = {"mu": "session_idx", "car": "car_idx", "skill": "entry_idx"}
    if car_transient:
        terms["car_event"] = "car_idx"
    if car_track:
        terms["car_track"] = "car_idx"
    if compat:
        terms["compat"] = "entry_idx"
    if driver_form:
        terms["driver_form"] = "entry_idx"
    if car_segment:
        terms["car_segment"] = "car_segment_idx"
    return terms


def residuals(design, post, **model_kw) -> pd.DataFrame:
    """Observed minus fitted pace, using every term of the fitted model."""
    o = design.obs
    terms = fitted_terms(**model_kw)
    missing = [k for k in terms if k not in post]
    if missing:
        raise KeyError(f"fit lacks saved term(s) {', '.join(missing)}; refit with the current fit.py "
                       "(which saves every term) before computing residuals")
    fitted = sum(flat(post, k).mean(0)[o[idx].to_numpy()] for k, idx in terms.items())
    sigma = (post["sigma0"].reshape(-1, 1) * np.exp(post["sigma_tau"].reshape(-1, 1)
                                                     * flat(post, "sigma_u"))).mean(0)
    e = design.entries.set_index("entry_idx")
    return o.assign(resid=o.y - fitted, z=(o.y - fitted) / sigma[o.session_idx],
                    driver=e.driver_id.loc[o.entry_idx].to_numpy())


def main(fit_name: str = "main") -> None:
    path = FITS / f"{fit_name}.npz"
    design = design_for(path)
    post, _ = load(path, design)
    model_kw = {k: ast.literal_eval(v) if isinstance(v, str) else v
                for k, v in (load_meta(path).get("model_kw") or {}).items()}
    r = residuals(design, post, **model_kw)
    p = r.merge(r, on=["session_idx", "car_idx"])
    p = p[p.driver_x < p.driver_y]
    print(f"teammate residual correlation (same segment): {np.corrcoef(p.z_x, p.z_y)[0, 1]:.3f} "
          f"(n={len(p)})")
    by_seg = p.merge(design.sessions[["session_idx", "segment"]], on="session_idx")
    for seg, g in by_seg.groupby("segment"):
        print(f"   {seg}: {np.corrcoef(g.z_x, g.z_y)[0, 1]:.3f}")
    # teammate gap residual persistence across a pairing's consecutive events
    p["gap_res"] = p.resid_x - p.resid_y
    ev = p.groupby(["driver_x", "driver_y", "event_idx_x"]).gap_res.mean().reset_index()
    ev["prev"] = ev.groupby(["driver_x", "driver_y"]).gap_res.shift()
    ev = ev.dropna()
    print(f"lag-1 autocorrelation of teammate-gap residuals across events: "
          f"{np.corrcoef(ev.gap_res, ev.prev)[0, 1]:.3f} (n={len(ev)})")
    q = np.quantile(r.z, [0.01, 0.05, 0.5, 0.95, 0.99])
    print("standardised residual quantiles 1/5/50/95/99%:", np.round(q, 2))


if __name__ == "__main__":
    main(*sys.argv[1:])
