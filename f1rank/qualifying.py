"""Qualifying features with an explicit information cutoff for racing validation.

python -m f1rank.qualifying --seasons 2012 2013 ...

Held-out season S uses a fit trained through S-1 for BOTH training and test
features. Test features are season-ahead forecasts conditional on the entered
drivers and circuits, not estimates updated with that weekend's qualifying.
Current/full-data racing fits may use the checked main fit. Never fall back to
it when a historical fold is missing.
"""

import argparse
from functools import lru_cache

import numpy as np

from .artifacts import diagnostics, require_convergence
from .design import build_design
from .fit import FITS, design_for, fit, load, load_meta, save
from .ratings import flat

POLICY = "qualifying-trained-before-test-season-v1"


def fit_path(season: int | None):
    return FITS / ("main.npz" if season is None else f"lfo_end{season - 1}.npz")


def check_cutoff(design, season: int) -> None:
    train = design.obs if design.train is None else design.obs.loc[design.train]
    if train.empty or int(train.event_id.str[:4].astype(int).max()) >= season:
        raise ValueError(f"Qualifying fold for {season} contains observations from its test season or later")
    if not (design.events.season == season).any():
        raise ValueError(f"Qualifying fit does not cover forecast season {season}; prepare its fold again")


@lru_cache(maxsize=2)
def features(season: int | None = None):
    path = fit_path(season)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}; run python -m f1rank.qualifying --seasons {season}")
    design = build_design(2010) if season is None else design_for(path)
    if season is not None:
        check_cutoff(design, season)
    meta = load_meta(path)
    if meta.get("model_kw", {}) != {} or meta.get("start_season") != 2010:
        raise ValueError(f"{path.name} is not the standard qualifying model")
    post, info = load(path, design)
    require_convergence(diagnostics(post, info["divergences"]))
    e = design.entries[["event_id", "driver_id"]].copy()
    e["driver"] = np.median(flat(post, "skill") + flat(post, "compat"), axis=0)
    c = design.cars[["team", "event_idx"]].copy()
    c["event_id"] = design.events.event_id.to_numpy()[c.event_idx]
    c["car"] = np.median(flat(post, "car") + flat(post, "car_track"), axis=0)
    return e, c.drop(columns="event_idx")


def dependencies(seasons=()):
    from .fit import meta_path
    return [p for S in seasons for p in (fit_path(S), meta_path(fit_path(S)))]


def prepare(season: int) -> None:
    path = fit_path(season)
    if path.exists():
        # A corrected historical source or a newly added event requires a new fit;
        # do not silently replace a valid existing fold.
        features.cache_clear()
        try:
            e, _ = features(season)
            events = build_design(2010).events
            expected = set(events.loc[events.season == season, "event_id"])
            if expected <= set(e.event_id):
                print(f"{path.name}: checked, already available", flush=True)
                return
        except (ValueError, RuntimeError, FileNotFoundError):
            pass
    design = build_design(2010)
    cutoff = int(design.events.loc[design.events.season < season, "event_idx"].max())
    design = design.with_cutoff(cutoff)
    check_cutoff(design, season)
    # Fixed retry rule; failed attempts never overwrite the previous fit.
    attempts = []
    for attempt in range(3):
        settings = dict(warmup=700 * (attempt + 1), samples=400 * 2 ** attempt,
                        chains=4, target_accept=0.9, seed=attempt)
        post, info = fit(design, **settings)
        checked = diagnostics(post, info["divergences"])
        attempts.append({"settings": settings, "diagnostics": checked})
        print(f"{path.name}, attempt {attempt + 1}: {checked}", flush=True)
        if checked["converged"]:
            save(path, post, info, design, job=path.stem, model_kw={}, settings=settings, attempts=attempts)
            break
    else:
        require_convergence(checked)
    features.cache_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for S in args.seasons:
        prepare(S)


if __name__ == "__main__":
    main()
