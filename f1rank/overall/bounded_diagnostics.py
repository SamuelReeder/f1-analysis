"""Production copy of the verified overall_bounded_diagnostics.py.

The original research implementation and its evidence remain unchanged.
"""
import math

import numpy as np

from .diagnostics import (DETERMINISTIC_SITES, MAX_DIVERGENCE_RATE,
    MAX_MCSE_SD_RATIO, MIN_CHAINS, MIN_ESS, RHAT_LIMIT)

COORDINATE_BLOCK = 128


def _columns(raw, start, stop):
    """C-order trailing coordinates without flattening/copying the whole site."""
    if raw.ndim == 2:
        return np.asarray(raw, dtype=float)[..., None]
    indices = np.unravel_index(np.arange(start, stop), raw.shape[2:])
    return np.asarray(raw[(slice(None), slice(None), *indices)], dtype=float)


def check(grouped, divergences, *, coordinate_block=COORDINATE_BLOCK):
    from arviz_stats.base import array_stats
    if type(coordinate_block) is not int or coordinate_block < 1:
        raise ValueError('A positive coordinate block size is required')
    if not grouped:
        raise ValueError('Empty posterior')
    shape = np.asarray(next(iter(grouped.values()))).shape[:2]
    if len(shape) != 2 or shape[0] < MIN_CHAINS or shape[1] < MIN_ESS // MIN_CHAINS:
        return dict(converged=False, reason='Insufficient chains/draws', shape=list(shape))
    rows, failures, constants, empty = [], [], [], []
    for name, raw in grouped.items():
        raw = np.asarray(raw)
        if raw.shape[:2] != shape:
            failures.append(f'{name}: invalid shape or nonfinite values')
            continue
        if raw.size == 0:
            empty.append(dict(parameter=name, shape=list(raw.shape)))
            continue
        ncoords = math.prod(raw.shape[2:])
        varying = np.zeros(ncoords, dtype=bool)
        valid = True
        for start in range(0, ncoords, coordinate_block):
            stop = min(start + coordinate_block, ncoords)
            a = _columns(raw, start, stop)
            if not np.isfinite(a).all():
                valid = False
                break
            varying[start:stop] = np.ptp(a.reshape(-1, stop-start), axis=0) > 0
        if not valid:
            failures.append(f'{name}: invalid shape or nonfinite values')
            continue
        constant_count = int((~varying).sum())
        if constant_count:
            constants.append(dict(parameter=name, coordinates=constant_count,
                                  deterministic=name in DETERMINISTIC_SITES))
            if name not in DETERMINISTIC_SITES:
                failures.append(f'{name}: constant stochastic coordinate')
        if not varying.any():
            continue
        finite = True
        rhat_max, bulk_min, tail_min, ratio_max = -np.inf, np.inf, np.inf, -np.inf
        for start in range(0, ncoords, coordinate_block):
            stop = min(start + coordinate_block, ncoords)
            selected = varying[start:stop]
            if not selected.any():
                continue
            a = _columns(raw, start, stop)[:, :, selected]
            kwargs = dict(chain_axis=0, draw_axis=1)
            rhat = np.asarray(array_stats.rhat(a, **kwargs, method='rank'))
            bulk = np.asarray(array_stats.ess(a, **kwargs, method='bulk'))
            tail = np.asarray(array_stats.ess(a, **kwargs, method='tail', prob=(.05, .95)))
            mcse = np.asarray(array_stats.mcse(a, **kwargs, method='mean'))
            ratio = mcse / a.reshape(-1, a.shape[-1]).std(axis=0, ddof=1)
            finite = finite and all(np.isfinite(x).all() for x in (rhat, bulk, tail, ratio))
            if finite:
                rhat_max = max(rhat_max, float(rhat.max()))
                bulk_min = min(bulk_min, float(bulk.min()))
                tail_min = min(tail_min, float(tail.min()))
                ratio_max = max(ratio_max, float(ratio.max()))
        passed = bool(finite and rhat_max < RHAT_LIMIT and bulk_min >= MIN_ESS
                      and tail_min >= MIN_ESS and ratio_max <= MAX_MCSE_SD_RATIO)
        if not passed:
            failures.append(f'{name}: convergence/precision threshold failed')
        rows.append(dict(parameter=name, passed=passed,
            rhat_max=rhat_max if finite else None, bulk_ess_min=bulk_min if finite else None,
            tail_ess_min=tail_min if finite else None, mcse_sd_max=ratio_max if finite else None,
            constant_coordinates=constant_count))
    n = int(np.prod(shape))
    if divergences < 0 or divergences > n * MAX_DIVERGENCE_RATE:
        failures.append('Divergence threshold failed')
    if not rows:
        failures.append('No varying posterior coordinates')
    return dict(converged=not failures, failures=failures, parameters=rows, constants=constants, empty_sites=empty,
                n_draws=n, n_chains=shape[0], divergences=int(divergences))
