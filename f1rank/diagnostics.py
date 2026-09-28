"""Posterior checks on the main fit.

    python -m f1rank.diagnostics [fit_name]

teammate residual correlation  residuals of two teammates in the same segment
                               should be uncorrelated if shared car effects are
                               fully captured; positive correlation means a
                               segment-level team effect is missing
residual autocorrelation       a driver's residual teammate gap should not
                               persist across events if skill dynamics are right
"""

import sys

import numpy as np
import pandas as pd

from .design import build_design
from .fit import FITS, load
from .ratings import flat


def residuals(design, post) -> pd.DataFrame:
    o = design.obs
    car = flat(post, "car") + flat(post, "car_track")
    if "car_event" in post:
        car = car + flat(post, "car_event")
    fitted = (flat(post, "mu")[:, o.session_idx] + car[:, o.car_idx]
              + flat(post, "skill")[:, o.entry_idx]).mean(0)
    if "car_segment" in post:
        fitted = fitted + flat(post, "car_segment").mean(0)[o.car_segment_idx]
    sigma = (post["sigma0"].reshape(-1, 1) * np.exp(post["sigma_tau"].reshape(-1, 1)
                                                     * flat(post, "sigma_u"))).mean(0)
    e = design.entries.set_index("entry_idx")
    return o.assign(resid=o.y - fitted, z=(o.y - fitted) / sigma[o.session_idx],
                    driver=e.driver_id.loc[o.entry_idx].to_numpy())


def main(fit_name: str = "main") -> None:
    design = build_design(2010)
    post, _ = load(FITS / f"{fit_name}.npz")
    r = residuals(design, post)
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
