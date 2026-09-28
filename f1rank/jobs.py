"""Validation and sensitivity fits, run as independent processes.

    python -m f1rank.jobs list                 # job names, with up-to-date status
    python -m f1rank.jobs synth-source         # freeze the main fit as the synthetic source
    python -m f1rank.jobs run NAME [NAME..]    # run jobs in this process
    python -m f1rank.jobs all --parallel 4     # run every job that is missing or stale

lfo_*     leave-future-out: fit on events up to a cutoff, forecast the rest
lfo2006_* the same cutoffs with the data window starting in 2006
synth_*   synthetic recovery on the real design (clean + misspecified)
sens_*    sensitivity variants of the main model on all data
placebo   the main model plus a placebo effect per half of a multi-season team stint

A fit is up to date when its metadata matches the design it would be fitted on now
(see fit.py), so new data or a changed design makes the affected fits stale. Synthetic
fits are up to date when they were made from the current synthetic source, which is a
frozen copy of the main fit made explicitly with `synth-source` (and by `all` if it is
missing). Any failed job makes the command exit with status 1.
"""

import argparse
import shutil
import subprocess
import sys
import traceback
from functools import lru_cache, partial

import numpy as np

from .design import build_design
from .fit import FITS, IncompatibleFit, check, fingerprint, fit, load, load_meta, meta_path, save
from .model import model

VALID = dict(warmup=700, samples=400, chains=4, target_accept=0.9)
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
DIAGNOSTIC = {"placebo": dict(model_kw=dict(placebo=True))}
# cutoffs at which the 2006 data window is compared with 2010 (window_compare.py)
WINDOW_CUTS = ("end2016", "end2019", "end2022", "end2024", "end2025", "mid2025")


@lru_cache(maxsize=2)
def _base_design(start: int):
    return build_design(start)


def job_names() -> list[str]:
    from .simulate import SCENARIOS
    return [*lfo_cutoffs(_base_design(2010)), *(f"lfo2006_{c}" for c in WINDOW_CUTS),
            *(f"synth_{s}" for s in SCENARIOS), *SENS, *DIAGNOSTIC]


def job_design(name: str):
    """The design a (non-synthetic) job is fitted on, from the current data."""
    if name.startswith("lfo2006_"):
        design = _base_design(2006)
        return design.with_cutoff(lfo_cutoffs(design)["lfo_" + name[len("lfo2006_"):]])
    if name.startswith("lfo_"):
        design = _base_design(2010)
        return design.with_cutoff(lfo_cutoffs(design)[name])
    spec = SENS.get(name) or DIAGNOSTIC.get(name)
    if spec is None:
        raise ValueError(f"unknown job {name!r}")
    return _base_design(spec.get("start", 2010))


def up_to_date(name: str) -> bool:
    path = FITS / f"{name}.npz"
    meta = load_meta(path)
    if not path.exists() or meta is None:
        return False
    if name.startswith("synth_"):
        source = load_meta(SYNTH_SOURCE)
        return source is not None and meta.get("source_fingerprint") == source["fingerprint"]
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


def run(name: str) -> None:
    out = FITS / f"{name}.npz"
    if name.startswith("synth_"):
        from .simulate import SCENARIOS, draw_truth, synthetic_design
        scenario = name[len("synth_"):]
        source = load_meta(SYNTH_SOURCE)
        if source is None:
            raise FileNotFoundError(f"{SYNTH_SOURCE.name} missing: run `python -m f1rank.jobs synth-source`")
        design = build_design(2010, end_event=source["data_as_of"])
        main, _ = load(SYNTH_SOURCE, design)
        hyper = {k: float(np.median(v)) for k, v in main.items() if v.ndim == 2}
        rng = np.random.default_rng(1234)  # same truth for every scenario
        truth = draw_truth(design, hyper, main["mu"].mean((0, 1)), rng)
        synth = synthetic_design(design, truth, scenario, hyper["nu"],
                                 np.random.default_rng(100 + SCENARIOS.index(scenario)))
        post, info = fit(synth, progress=False, **VALID)
        np.savez_compressed(FITS / f"{name}_truth.npz", **truth)
        save(out, post, info, synth, job=name, scenario=scenario, settings=VALID,
             source_fingerprint=source["fingerprint"], real_data_fingerprint=fingerprint(design))
    else:
        design = job_design(name)
        model_kw = (SENS.get(name) or DIAGNOSTIC.get(name) or {}).get("model_kw", {})
        post, info = fit(design, progress=False, model_fn=partial(model, **model_kw), **VALID)
        save(out, post, info, design, job=name, model_kw={k: repr(v) for k, v in model_kw.items()},
             settings=VALID)
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
        # round-robin into `parallel` worker processes
        groups = [todo[i::args.parallel] for i in range(args.parallel) if todo[i::args.parallel]]
        procs = [(g, subprocess.Popen([sys.executable, "-m", "f1rank.jobs", "run", *g])) for g in groups]
        bad = [g for g, proc in procs if proc.wait() != 0]
        missing = [n for n in todo if not up_to_date(n)]
        if bad or missing:
            print(f"FAILED: {len(bad)} worker(s) exited with an error; not up to date after the run: "
                  f"{', '.join(missing) or 'none'}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
