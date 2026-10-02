"""Validation and sensitivity fits, run as independent processes.

    python -m f1rank.jobs list                 # job names, with up-to-date status
    python -m f1rank.jobs synth-source         # freeze the main fit as the synthetic source
    python -m f1rank.jobs run NAME [NAME..]    # run jobs in this process
    python -m f1rank.jobs all --parallel 4     # run every job that is missing or stale

lfo_*        leave-future-out: fit on events up to a cutoff, forecast the rest
lfo2006_*    some of the same cutoffs with the data window starting in 2006
lfospell_*,  some of the same cutoffs with one team-specific effect per spell or per
lfoera_*     regulation era (instead of per team lineage)
synth_<scenario>  synthetic recovery on the real design, one shared truth (clean +
             misspecified scenarios)
synth_seed<k>     clean synthetic recovery, each with its own truth, noise and
             hyperparameters (a different posterior draw of the synthetic source)
sens_*       sensitivity variants of the main model on all data
placebo      the main model plus a placebo effect per half of a multi-season team stint

Every job uses the fixed retry rule ATTEMPTS: a fit that fails the publication
convergence checks (artifacts.diagnostics: R-hat < 1.05 for every parameter, at most one
divergence per 1,000 draws) is repeated with longer chains and a new seed. A job whose
attempts all fail keeps no fit (a stale one is removed) and writes <name>.failed.json;
the evaluation lists it as excluded. Each leave-future-out design ends with its test
season: later seasons have no data in the fit and are not forecast, and keeping them only
added unobserved car paths that NUTS mixed poorly (car R-hat up to 2.4 before 2026-10).

A fit is up to date when its metadata matches the design it would be fitted on now
(see fit.py), so new data or a changed design makes the affected fits stale. Synthetic
fits are up to date when they were made from the current synthetic source (a frozen copy
of the main fit, made explicitly with `synth-source`, and by `all` if it is missing) on the
design the current code builds from that source's data. Any failed job makes the command
exit with status 1.
"""

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
import traceback
from functools import lru_cache, partial
from pathlib import Path

import jax
import numpy as np

from .design import build_design
from .fit import FITS, IncompatibleFit, check, fingerprint, fit, load, load_meta, meta_path, save
from .model import model

# Fixed retry rule, shared with the racing folds (qualifying.prepare). The last attempt
# raises target_accept against divergences (added 2026-10-01 after the 2014 racing fold
# failed the first three on divergences).
ATTEMPTS = (*(dict(warmup=700 * (k + 1), samples=400 * 2 ** k, chains=4, target_accept=0.9, seed=k)
              for k in range(3)),
            dict(warmup=2100, samples=1600, chains=4, target_accept=0.98, seed=3))
MAIN = FITS / "main.npz"
# hyperparameters for synthetic truth: a frozen copy of the main fit, so synthetic jobs
# never read a main fit that is being rewritten
SYNTH_SOURCE = FITS / "synth_source.npz"
DRIFT = (0.0108, 0.0437)  # posterior medians of driver drift (per race, per season)


def lfo_cutoffs(design) -> dict[str, int]:
    ev = design.events
    out = {}
    for season in range(max(2013, design.events.season.min() + 3), 2026):  # end of season
        out[f"lfo_end{season}"] = int(ev[ev.season == season].event_idx.max())
    for season in range(2015, 2027):  # after round 8 -> forecast the rest of the season
        r8 = ev[(ev.season == season) & (ev["round"] == 8)]
        if len(r8):
            out[f"lfo_mid{season}"] = int(r8.event_idx.iloc[0])
    return out


SENS = {
    "sens_start2006": dict(start=2006),
    "sens_notrack": dict(model_kw=dict(car_track=False)),
    "sens_gausscar": dict(model_kw=dict(car_step_df=1000.0)),
    "sens_notransient": dict(model_kw=dict(car_transient=False)),
    "sens_noform": dict(model_kw=dict(driver_form=False)),
    "sens_nocompat": dict(model_kw=dict(compat=False)),
    "sens_splitt": dict(model_kw=dict(likelihood="split_t")),
    "sens_drift_x2": dict(model_kw=dict(skill_drift=(2 * DRIFT[0], 2 * DRIFT[1]))),
    "sens_drift_half": dict(model_kw=dict(skill_drift=(0.5 * DRIFT[0], 0.5 * DRIFT[1]))),
}
SENS["sens_team_spell"] = dict(model_kw=dict(compat_unit="spell"))
SENS["sens_team_era"] = dict(model_kw=dict(compat_unit="era"))
DIAGNOSTIC = {"placebo": dict(model_kw=dict(placebo=True))}
# forecasting variants, compared with the main model on identical targets at some
# cutoffs (compare.py): the 2006 data window, and the team-specific effect per spell or
# per regulation era. The team cutoffs include the ends of 2016, 2021 and 2025, which
# forecast the first season after a regulation reset.
WINDOW_CUTS = ("end2016", "end2019", "end2022", "end2024", "end2025", "mid2025")
TEAM_CUTS = ("end2016", "end2019", "end2021", "end2024", "end2025", "mid2025")
# The pre-registered sprint qualifying test (docs/sprint_qualifying.md): every cutoff
# after the first sprint qualifying session (2023-04).
SPRINT_CUTS = ("mid2023", "end2023", "mid2024", "end2024", "mid2025", "end2025", "mid2026")
LFO_VARIANTS = {
    "lfo2006": dict(start=2006, cuts=WINDOW_CUTS),
    "lfospell": dict(model_kw=dict(compat_unit="spell"), cuts=TEAM_CUTS),
    "lfoera": dict(model_kw=dict(compat_unit="era"), cuts=TEAM_CUTS),
    "lfosprint": dict(sprint_quali=True, cuts=SPRINT_CUTS),
}
N_SYNTH_SEEDS = 8


@lru_cache(maxsize=2)
def _base_design(start: int):
    return build_design(start)


@lru_cache(maxsize=2)
def _synth_design(data_as_of: str):
    return build_design(2010, end_event=data_as_of)


def job_names() -> list[str]:
    from .simulate import SCENARIOS
    return [*lfo_cutoffs(_base_design(2010)),
            *(f"{v}_{c}" for v, spec in LFO_VARIANTS.items() for c in spec["cuts"]),
            *(f"synth_{s}" for s in SCENARIOS), *(f"synth_seed{k}" for k in range(1, N_SYNTH_SEEDS + 1)),
            *SENS, *DIAGNOSTIC]


def job_spec(name: str) -> dict:
    """Data window, model options and cutoff of a (non-synthetic) job."""
    prefix, _, cut = name.partition("_")
    if prefix == "lfo":
        return dict(start=2010, model_kw={}, cutoff=name)
    if prefix in LFO_VARIANTS and cut in LFO_VARIANTS[prefix]["cuts"]:
        v = LFO_VARIANTS[prefix]
        return dict(start=v.get("start", 2010), model_kw=v.get("model_kw", {}), cutoff="lfo_" + cut,
                    sprint_quali=v.get("sprint_quali", False))
    spec = SENS.get(name) or DIAGNOSTIC.get(name)
    if spec is None:
        raise ValueError(f"unknown job {name!r}")
    return dict(start=spec.get("start", 2010), model_kw=spec.get("model_kw", {}), cutoff=None)


def job_design(name: str):
    """The design a (non-synthetic) job is fitted on, from the current data."""
    return lfo_design(name)[0]


def lfo_design(name: str):
    """(design, index of the last training event) of a job; the index is None without a
    cutoff. A cutoff design ends with its test season: the rest of the cutoff's season
    (lfo_mid*) or the next one (lfo_end*)."""
    spec = job_spec(name)
    design = _base_design(spec["start"])
    if spec["cutoff"] is None:
        return design, None
    cut = lfo_cutoffs(design)[spec["cutoff"]]
    ev = design.events
    test_season = ev.season[cut] + (1 if spec["cutoff"].startswith("lfo_end") else 0)
    last = str(ev.event_id.iloc[cut])
    design = build_design(spec["start"], end_event=str(ev[ev.season == test_season].event_id.max()),
                          sprint_quali_until=last if spec.get("sprint_quali") else None)
    return design.with_cutoff(cut), cut


def up_to_date(name: str) -> bool:
    path = FITS / f"{name}.npz"
    meta = load_meta(path)
    if not path.exists() or meta is None:
        return False
    if name.startswith("synth_"):
        # same frozen source, and the design built from its data by the current code
        source = load_meta(SYNTH_SOURCE)
        return (source is not None and meta.get("source_fingerprint") == source["fingerprint"]
                and meta.get("real_data_fingerprint") == fingerprint(_synth_design(source["data_as_of"])))
    try:
        check(path, job_design(name))
    except (IncompatibleFit, ValueError):
        return False
    return True


def make_synth_source() -> None:
    """Freeze the current main fit as the source of synthetic truth."""
    if not MAIN.exists():
        raise FileNotFoundError(f"{MAIN} not found: run `python -m f1rank.fit` first")
    check(MAIN, build_design(2010))  # the main fit must match the current data
    shutil.copyfile(MAIN, SYNTH_SOURCE)
    shutil.copyfile(meta_path(MAIN), meta_path(SYNTH_SOURCE))
    print(f"synthetic source: frozen copy of {MAIN.name} (data as of {load_meta(MAIN)['data_as_of']})")


def synth_setup(name: str, design, main: dict) -> tuple[str, dict, np.ndarray, np.random.Generator,
                                                     np.random.Generator, dict]:
    """Scenario, hyperparameters, segment intercepts, truth and noise generators for a
    synthetic job. synth_<scenario>: the posterior medians and one shared truth (seed
    1234). synth_seed<k>: clean data with its own truth and noise, and hyperparameters
    from posterior draw number `draw` of the synthetic source."""
    from .simulate import SCENARIOS
    tag = name[len("synth_"):]
    if tag in SCENARIOS:
        hyper = {k: float(np.median(v)) for k, v in main.items() if v.ndim == 2}
        return (tag, hyper, main["mu"].mean((0, 1)), np.random.default_rng(1234),
                np.random.default_rng(100 + SCENARIOS.index(tag)), {})
    if not tag.startswith("seed") or not tag[4:].isdigit():
        raise ValueError(f"unknown job {name!r}")
    k = int(tag[4:])
    n_draws = main["sigma0"].size
    draw = int(np.random.default_rng([k, 0]).integers(n_draws))
    hyper = {key: float(v.reshape(-1)[draw]) for key, v in main.items() if v.ndim == 2}
    mu = main["mu"].reshape(n_draws, -1)[draw]
    return ("clean", hyper, mu, np.random.default_rng([k, 1]), np.random.default_rng([k, 2]),
            dict(seed=k, draw=draw, hyper=hyper))


def copy_identical(target: Path, source: Path, design) -> bool:
    """Copy an existing converged fit of the same design made under ATTEMPTS instead of
    refitting: the validation's lfo_end<Y> and the racing fold quali_fold<Y+1> (both the
    main model, trained through Y, design ending with Y+1) are the same fit."""
    if not source.exists():
        return False
    try:
        meta = check(source, design)
    except IncompatibleFit:
        return False
    tried = meta.get("attempts") or []
    if (meta.get("model_kw") != {} or not tried or not tried[-1]["diagnostics"]["converged"]
            or [a["settings"] for a in tried] != list(ATTEMPTS[:len(tried)])):
        return False
    shutil.copyfile(source, target)
    meta_path(target).write_text(json.dumps({**meta, "job": target.stem, "copied_from": source.name}, indent=1))
    print(f"{target.name}: copied from {source.name} (same design and retry rule)", flush=True)
    return True


def fit_with_retries(name: str, design, **kw) -> tuple[dict, dict, dict, list[dict]]:
    """Fit under ATTEMPTS; returns the first converged fit, its settings and every attempt.
    Raises RuntimeError (after recording <name>.failed.json) if none converges."""
    from .artifacts import diagnostics
    tried = []
    for settings in ATTEMPTS:
        post, info = fit(design, progress=False, **settings, **kw)
        checked = diagnostics(post, info["divergences"])
        tried.append({"settings": settings, "diagnostics": checked, "elapsed_s": info["elapsed_s"]})
        print(f"{name}, attempt {len(tried)}: {checked}", flush=True)
        if checked["converged"]:
            (FITS / f"{name}.failed.json").unlink(missing_ok=True)
            return post, info, settings, tried
        # release the failed draws and the attempt's compiled programs before the next,
        # longer attempt (results are unchanged: each attempt's seed and settings are fixed)
        post = None
        gc.collect()
        jax.clear_caches()
    for path in (FITS / f"{name}.npz", meta_path(FITS / f"{name}.npz")):
        path.unlink(missing_ok=True)  # never leave a stale fit in its place
    (FITS / f"{name}.failed.json").write_text(json.dumps({"job": name, "attempts": tried}, indent=1))
    raise RuntimeError(f"{name}: no attempt passed the convergence checks")


def run(name: str) -> None:
    out = FITS / f"{name}.npz"
    if name.startswith("synth_"):
        from .simulate import draw_truth, synthetic_design
        source = load_meta(SYNTH_SOURCE)
        if source is None:
            raise FileNotFoundError(f"{SYNTH_SOURCE.name} missing: run `python -m f1rank.jobs synth-source`")
        design = _synth_design(source["data_as_of"])
        main, _ = load(SYNTH_SOURCE, design)
        scenario, hyper, mu, truth_rng, noise_rng, setting = synth_setup(name, design, main)
        truth = draw_truth(design, hyper, mu, truth_rng)
        synth = synthetic_design(design, truth, scenario, hyper["nu"], noise_rng)
        post, info, settings, tried = fit_with_retries(name, synth)
        np.savez_compressed(FITS / f"{name}_truth.npz", **truth)
        save(out, post, info, synth, job=name, scenario=scenario, settings=settings, attempts=tried, **setting,
             source_fingerprint=source["fingerprint"], real_data_fingerprint=fingerprint(design))
    else:
        design = job_design(name)
        model_kw = job_spec(name)["model_kw"]
        if name.startswith("lfo_end") and copy_identical(out, FITS / f"quali_fold{int(name[7:]) + 1}.npz", design):
            return
        post, info, settings, tried = fit_with_retries(name, design, model_fn=partial(model, **model_kw))
        save(out, post, info, design, job=name, model_kw={k: repr(v) for k, v in model_kw.items()},
             settings=settings, attempts=tried)
    print(name, info, flush=True)


def run_many(names: list[str]) -> list[str]:
    """Run jobs one after another; a failure is reported and the rest still run."""
    failed = []
    for name in names:
        try:
            run(name)
        except Exception:  # noqa: BLE001 - report every failure, then exit non-zero
            traceback.print_exc()
            print(f"FAILED {name}", file=sys.stderr, flush=True)
            failed.append(name)
    return failed


def run_pool(names: list[str], parallel: int) -> list[str]:
    """Run each job in its own process, `parallel` at a time, taking the next job as soon
    as a worker is free. Returns the jobs that failed."""
    queue, running, failed = list(names), {}, []
    while queue or running:
        while queue and len(running) < parallel:
            name = queue.pop(0)
            running[name] = subprocess.Popen([sys.executable, "-m", "f1rank.jobs", "run", name])
        for name, proc in list(running.items()):
            if proc.poll() is not None:
                if proc.returncode != 0:
                    failed.append(name)
                del running[name]
        time.sleep(2)
    return failed


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("cmd", choices=["list", "synth-source", "run", "all"])
    p.add_argument("names", nargs="*")
    p.add_argument("--parallel", type=int, default=4)
    args = p.parse_args(argv)

    if args.cmd == "list":
        for name in job_names():
            print(f"{name:24s} {'up to date' if up_to_date(name) else 'to run'}")
    elif args.cmd == "synth-source":
        make_synth_source()
    elif args.cmd == "run":
        failed = run_many(args.names)
        if failed:
            sys.exit(1)
    else:
        todo = [n for n in (args.names or job_names()) if not up_to_date(n)]
        if any(n.startswith("synth_") for n in todo) and load_meta(SYNTH_SOURCE) is None:
            make_synth_source()
        print(f"{len(todo)} jobs to run", flush=True)
        bad = run_pool(todo, args.parallel)
        missing = [n for n in todo if not up_to_date(n)]
        if bad or missing:
            print(f"FAILED: {', '.join(bad) or 'none'}; not up to date after the run: "
                  f"{', '.join(missing) or 'none'}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
