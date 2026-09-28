"""Validation and sensitivity fits, run as independent processes.

    python -m f1rank.jobs list               # show job names
    python -m f1rank.jobs run NAME [NAME..]  # run jobs in this process
    python -m f1rank.jobs all --parallel 3   # run every job not yet done

lfo_*   leave-future-out: fit on events up to a cutoff, forecast the rest
synth_* synthetic recovery on the real design (clean + misspecified)
sens_*  sensitivity variants of the main model on all data
"""

import argparse
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np

from .design import build_design
from .fit import FITS, fit, load, save
from .model import model

VALID = dict(warmup=700, samples=400, chains=4, target_accept=0.9)
MAIN = FITS / "main.npz"
# hyperparameters for synthetic truth: a fixed copy, so synthetic jobs never read a
# main fit that is being rewritten
SYNTH_SOURCE = FITS / "synth_source.npz"
DRIFT = (0.0108, 0.0437)  # posterior medians of driver drift (per race, per season)


def lfo_cutoffs(design) -> dict[str, int]:
    ev = design.events
    out = {}
    for season in range(max(2013, design.events.season.min() + 3), 2026):  # end of season
        out[f"lfo_end{season}"] = int(ev[ev.season == season].event_idx.max())
    for season in range(2015, 2027):  # after round 8 -> forecast the rest of the season
        out[f"lfo_mid{season}"] = int(ev[(ev.season == season) & (ev["round"] == 8)].event_idx.iloc[0])
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


def job_names() -> list[str]:
    from .simulate import SCENARIOS
    return [*lfo_cutoffs(build_design(2010)), *(f"synth_{s}" for s in SCENARIOS), *SENS]


def run(name: str) -> None:
    out = FITS / f"{name}.npz"
    if name.startswith("lfo2006_"):  # same cutoffs, data window starting 2006
        design = build_design(2006)
        design = design.with_cutoff(lfo_cutoffs(design)["lfo_" + name[len("lfo2006_"):]])
        post, info = fit(design, progress=False, **VALID)
    elif name.startswith("lfo_"):
        design = build_design(2010)
        design = design.with_cutoff(lfo_cutoffs(design)[name])
        post, info = fit(design, progress=False, **VALID)
    elif name.startswith("synth_"):
        from .simulate import SCENARIOS, draw_truth, synthetic_design
        scenario = name[len("synth_"):]
        design = build_design(2010)
        main, _ = load(SYNTH_SOURCE)
        hyper = {k: float(np.median(v)) for k, v in main.items() if v.ndim == 2}
        rng = np.random.default_rng(1234)  # same truth for every scenario
        truth = draw_truth(design, hyper, main["mu"].mean((0, 1)), rng)
        synth = synthetic_design(design, truth, scenario, hyper["nu"],
                                 np.random.default_rng(100 + SCENARIOS.index(scenario)))
        post, info = fit(synth, progress=False, **VALID)
        np.savez_compressed(FITS / f"{name}_truth.npz", **truth)
    elif name.startswith("sens_"):
        spec = SENS[name]
        design = build_design(spec.get("start", 2010))
        post, info = fit(design, progress=False,
                         model_fn=partial(model, **spec.get("model_kw", {})), **VALID)
    else:
        raise ValueError(name)
    save(out, post, info)
    print(name, info, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("cmd", choices=["list", "run", "all"])
    p.add_argument("names", nargs="*")
    p.add_argument("--parallel", type=int, default=4)
    args = p.parse_args()

    if args.cmd == "list":
        print("\n".join(job_names()))
    elif args.cmd == "run":
        for name in args.names:
            run(name)
    else:
        todo = [n for n in (args.names or job_names()) if not (FITS / f"{n}.npz").exists()]
        # round-robin into `parallel` worker processes
        procs = [subprocess.Popen([sys.executable, "-m", "f1rank.jobs", "run", *todo[i::args.parallel]])
                 for i in range(args.parallel) if todo[i::args.parallel]]
        for proc in procs:
            proc.wait()


if __name__ == "__main__":
    main()
