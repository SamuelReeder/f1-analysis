"""New-candidate diagnostics; never relabel the legacy fits retrospectively."""
import numpy as np

RHAT_LIMIT = 1.01
MIN_ESS = 400
MAX_MCSE_SD_RATIO = .05
MAX_DIVERGENCE_RATE = .001
MIN_CHAINS = 4
DETERMINISTIC_SITES = frozenset({"skill", "car", "car_event", "car_track", "track_load",
    "compat", "driver_form", "car_segment", "overall_race_skill", "overall_team",
    "overall_car", "overall_day", "overall_grid", "overall_pace", "overall_hazard"})


def check(grouped, divergences):
    from arviz_stats.base import array_stats
    if not grouped:
        raise ValueError("Empty posterior")
    shape = np.asarray(next(iter(grouped.values()))).shape[:2]
    if len(shape) != 2 or shape[0] < MIN_CHAINS or shape[1] < MIN_ESS // MIN_CHAINS:
        return dict(converged=False, reason="Insufficient chains/draws", shape=list(shape))
    rows, failures, constants = [], [], []
    for name, raw in grouped.items():
        a = np.asarray(raw, float)
        if a.shape[:2] != shape or not np.isfinite(a).all():
            failures.append(f"{name}: invalid shape or nonfinite values")
            continue
        a = a.reshape(*shape, -1)
        varying = np.ptp(a.reshape(-1, a.shape[-1]), axis=0) > 0
        if not varying.all():
            constants.append(dict(parameter=name, coordinates=int((~varying).sum()),
                                  deterministic=name in DETERMINISTIC_SITES))
        if not varying.all() and name not in DETERMINISTIC_SITES:
            failures.append(f"{name}: constant stochastic coordinate")
        if not varying.any():
            continue
        a = a[:, :, varying]
        kwargs = dict(chain_axis=0, draw_axis=1)
        rhat = np.asarray(array_stats.rhat(a, **kwargs, method="rank"))
        bulk = np.asarray(array_stats.ess(a, **kwargs, method="bulk"))
        tail = np.asarray(array_stats.ess(a, **kwargs, method="tail", prob=(.05, .95)))
        mcse = np.asarray(array_stats.mcse(a, **kwargs, method="mean"))
        ratio = mcse / a.reshape(-1, a.shape[-1]).std(axis=0, ddof=1)
        finite = all(np.isfinite(x).all() for x in (rhat, bulk, tail, ratio))
        passed = bool(finite and (rhat < RHAT_LIMIT).all() and (bulk >= MIN_ESS).all()
                      and (tail >= MIN_ESS).all() and (ratio <= MAX_MCSE_SD_RATIO).all())
        if not passed:
            failures.append(f"{name}: convergence/precision threshold failed")
        rows.append(dict(parameter=name, passed=passed,
            rhat_max=float(rhat.max()) if finite else None,
            bulk_ess_min=float(bulk.min()) if finite else None,
            tail_ess_min=float(tail.min()) if finite else None,
            mcse_sd_max=float(ratio.max()) if finite else None,
            constant_coordinates=int((~varying).sum())))
    n = int(np.prod(shape))
    if divergences < 0 or divergences > n * MAX_DIVERGENCE_RATE:
        failures.append("Divergence threshold failed")
    if not rows:
        failures.append("No varying posterior coordinates")
    return dict(converged=not failures, failures=failures, parameters=rows, constants=constants,
                n_draws=n, n_chains=shape[0], divergences=int(divergences))
