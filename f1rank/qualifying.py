"""Qualifying features with an explicit information cutoff for racing validation.

python -m f1rank.qualifying --seasons 2012 2013 ...

Held-out season S uses a fit trained through S-1 for BOTH training and test
features. Test features are season-ahead forecasts conditional on the entered
drivers and circuits, not estimates updated with that weekend's qualifying.
Current/full-data racing fits may use the checked main fit. Never fall back to
it when a historical fold is missing. A season whose fold failed every attempt of the
fixed retry rule is left out of the racing held-out tests (unconverged_folds).
"""

import argparse
from functools import lru_cache

import numpy as np

from .artifacts import diagnostics, require_convergence
from .design import build_design
from .fit import FITS, design_for, load, load_meta, save
from .ratings import flat

POLICY = "qualifying-trained-before-test-season-v1"


def fit_path(season: int | None):
    # Not lfo_end{season - 1}.npz: that is the qualifying validation fit (jobs.py), whose
    # design keeps every later season, so sharing the name made each overwrite the other.
    return FITS / ("main.npz" if season is None else f"quali_fold{season}.npz")


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


def failure_path(season: int):
    return FITS / f"quali_fold{season}.failed.json"


def unconverged_folds() -> list[int]:
    """Seasons whose racing fold has no fit because every attempt of the fixed retry rule
    failed (quali_fold<season>.failed.json). The racing held-out tests leave these test
    seasons out and list them, as the qualifying validation does with its cutoffs; the
    rule depends only on the qualifying fit, not on any racing result."""
    seasons = (int(f.name.removeprefix("quali_fold").removesuffix(".failed.json"))
               for f in FITS.glob("quali_fold*.failed.json"))
    return sorted(S for S in seasons if not fit_path(S).exists())


def dependencies(seasons=()):
    """A fold's fit and metadata, or for an unconverged fold its failure record."""
    from .fit import meta_path
    failed = set(unconverged_folds())
    return [p for S in seasons
            for p in ((failure_path(S),) if S in failed else (fit_path(S), meta_path(fit_path(S))))]


def fold_design(season: int):
    """The racing fold's design: trained through the season before `season`, ending with it.

    Forecast only the season being tested. Keeping every later season adds unobserved car
    paths (and regulation resets) to NUTS, although none of those states is used by this
    fold. Marginalising those future paths does not alter the model for training or
    test-season observations."""
    events = build_design(2010).events
    test_events = events.loc[events.season == season, "event_id"]
    if test_events.empty:
        raise ValueError(f"No events for qualifying forecast season {season}")
    design = build_design(2010, end_event=str(test_events.max()))
    cutoff = int(design.events.loc[design.events.season < season, "event_idx"].max())
    design = design.with_cutoff(cutoff)
    check_cutoff(design, season)
    return design


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
    design = fold_design(season)
    # Fixed retry rule shared with the qualifying validation (jobs.ATTEMPTS). If no
    # attempt converges, the fold is removed and <name>.failed.json records the attempts;
    # the validation's lfo_end<season-1> is the same fit, so its result is reused either way.
    from .jobs import copy_failure, copy_identical, fit_with_retries, lfo_design
    features.cache_clear()
    source = FITS / f"lfo_end{season - 1}.npz"
    if copy_identical(path, source, design):
        return
    if copy_failure(path, source, design, lambda: lfo_design(source.stem)[0]):
        raise RuntimeError(f"{path.name}: no attempt passed the convergence checks (as {source.stem})")
    post, info, settings, attempts = fit_with_retries(path.stem, design)
    save(path, post, info, design, job=path.stem, model_kw={}, settings=settings, attempts=attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for S in args.seasons:
        prepare(S)


if __name__ == "__main__":
    main()
