"""Registered synthetic recovery and computational profiling, never driver ranks.

    python -m f1rank.overall.recovery --registration docs/overall_profile_protocol.json

Run as a capped f1.slice user service. Observations are generated from known
latent states. The first registration profiles computation; it cannot establish
simulation-based calibration from its small inventory.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import argparse
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro import handlers

from ..artifacts import atomic_json, digest
from .design import build
from .fitting import arrays_digest, committed_registration, run
from .model import model
from .provenance import ROOT, snapshot, require_snapshot

N_DRIVERS = 6
EVENTS_PER_YEAR = 4
YEARS = (2021, 2022, 2023)
LAPS_PER_STINT = 8
SYNTHETIC_CASES = ("connected", "confounded", "sparse_rookie", "dominant_teammate", "era_boundary", "real_network")
REFERENCE_CUTOFF = "2026-15"
TRUTH_SITES = ("skill", "overall_race_skill", "compat", "overall_team", "car", "overall_car",
               "overall_grid", "overall_pace", "overall_hazard")


def synthetic_design(case):
    if case not in SYNTHETIC_CASES:
        raise ValueError("Unregistered synthetic case")
    if case == "real_network":
        design = build(REFERENCE_CUTOFF)
        # Treat network, observation availability and controls as fixed design.
        # No observed outcome values enter the generated likelihood responses.
        design.qualifying.circuit_factor = np.linspace(-1., 1., len(design.catalog["circuits"]))
        return design
    years = (2021, 2022, 2026) if case == "era_boundary" else YEARS
    drivers = [f"synthetic_{i}" for i in range(N_DRIVERS)]
    events, entries, times, races, laps = [], [], [], [], []
    for yi, year in enumerate(years):
        for ri in range(EVENTS_PER_YEAR):
            event = f"{year}-{ri + 1:02d}"
            event_number = yi * EVENTS_PER_YEAR + ri
            events.append(dict(event_id=event, season=year, circuit_id=f"track_{ri % 3}",
                               date=f"{year}-{ri + 1:02d}-01"))
            pairing = np.arange(N_DRIVERS) if case == "confounded" else np.roll(np.arange(N_DRIVERS), yi)
            for i, driver in enumerate(drivers):
                if case == "sparse_rookie" and i == N_DRIVERS - 1 and event_number < len(years) * EVENTS_PER_YEAR - 2:
                    continue
                team = f"team_{pairing[i] // 2}"
                row = dict(event_id=event, driver_id=driver, team=team, constructor_id=team,
                           constructor_name=team, quali_position=i + 1)
                entries.append(row)
                for segment in ("Q1", "Q2", "Q3"):
                    if not (case == "sparse_rookie" and i == N_DRIVERS - 1 and ri == EVENTS_PER_YEAR - 1):
                        times.append(dict(event_id=event, driver_id=driver, segment=segment, time_s=90.))
                races.append(dict(**{k: v for k, v in row.items() if k != "quali_position"},
                                  grid=i + 1, position=i + 1, status="Finished", laps=60))
                for stint in range(2):
                    for within in range(LAPS_PER_STINT):
                        lap = stint * LAPS_PER_STINT + within + 2
                        laps.append(dict(event_id=event, season=year, driver_id=driver, team=team,
                            compound="HARD" if stint else "MEDIUM", stint_key=f"{driver}|{stint}",
                            lap_number=lap, tyre_life=within + 2, y=0., close=float(within % 4 == 0),
                            near=float(within % 4 == 1), unpressured=float(within % 4 == 2)))
    event_frame = pd.DataFrame(events)
    lap_frame = pd.DataFrame(laps)
    t = dict(events=event_frame, entries=pd.DataFrame(entries), quali_times=pd.DataFrame(times),
        race=pd.DataFrame(races), race_laps=lap_frame,
        drivers=pd.DataFrame(dict(driver_id=drivers, dob=[f"{1988 + i}-01-01" for i in range(N_DRIVERS)],
                                 debut_season=[2010 + i for i in range(N_DRIVERS)])),
        race_weather=pd.DataFrame(dict(event_id=event_frame.event_id, rainfall=False)),
        timeline=pd.DataFrame(columns=["event_id", "kind"]))
    design = build(str(event_frame.event_id.max()), tables=t, clean=lap_frame)
    design.qualifying.circuit_factor = np.linspace(-1., 1., len(design.catalog["circuits"]))
    return design


def generate(case, seed):
    design = synthetic_design(case)
    blocks = dict(qualifying=design.qualifying.arrays(), laps=design.laps, outcomes=design.outcomes)
    as_jax = lambda b: {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in b.items()}
    fn = model
    if case == "dominant_teammate":
        fn = handlers.substitute(fn, data={"sd_level": jnp.asarray(.5),
            "level_z": jnp.array([2., -2., 0., 0., 0., 0.])})
    trace = handlers.trace(handlers.seed(fn, jax.random.PRNGKey(seed))).get_trace(
        *[as_jax(blocks[name]) for name in ("qualifying", "laps", "outcomes")], generate=True)
    truth = {name: np.asarray(trace[name]["value"]) for name in TRUTH_SITES}
    blocks = copy.deepcopy(blocks)
    blocks["qualifying"]["y"] = np.asarray(trace["y"]["value"])
    blocks["laps"]["y"] = np.asarray(trace["generated_laps"]["value"])
    order = np.asarray(trace["generated_order"]["value"])
    if not np.array_equal(np.sort(order, axis=-1),
            np.broadcast_to(np.arange(order.shape[-1]), order.shape)):
        raise ValueError("Invalid generated order; preserve the failure instead of indexing it")
    for name in ("entry", "log_grid", "mask"):
        blocks["outcomes"][name] = np.take_along_axis(np.asarray(blocks["outcomes"][name]), order, axis=1)
    blocks["outcomes"]["completed"] = np.asarray(trace["generated_completed"]["value"])
    blocks["outcomes"]["failed"] = np.asarray(trace["generated_failed"]["value"])
    arrays_digest(blocks)  # reject nonfinite generation, without silently redrawing
    catalog = dict(design.catalog, synthetic_case=case, generation_seed=seed,
                   experiment="conditional_observation_design", real_driver_ranking=False)
    return blocks, catalog, truth


def recovery_metrics(post, truth, catalog):
    """Current-state pairwise contrasts; no favorable driver or pair selection."""
    rows = []
    indices = catalog["field_entries"]
    for site in ("skill", "overall_race_skill"):
        draws = np.asarray(post[site]).reshape(-1, post[site].shape[-1])[:, indices]
        actual = truth[site][indices]
        for i in range(len(indices)):
            for j in range(i):
                difference = draws[:, i] - draws[:, j]
                target = float(actual[i] - actual[j])
                low, median, high = np.quantile(difference, [.05, .5, .95])
                rows.append(dict(site=site, driver_i=catalog["field"][i], driver_j=catalog["field"][j],
                    truth=target, q05=float(low), q50=float(median), q95=float(high),
                    covered=bool(low <= target <= high),
                    rank=int((difference < target).sum()), n_draws=len(difference)))
    return rows


def execute(registration_path):
    registration_path = Path(registration_path).resolve()
    registration = committed_registration(registration_path)
    if registration.get("purpose") != "numerical_profile":
        raise ValueError("This entry point requires the synthetic numerical-profile registration")
    inputs = snapshot()
    root = ROOT / "outputs" / "championship" / "development" / registration["id"]
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for study in registration["studies"]:
        spec = registration["synthetic"][study]
        try:
            require_snapshot(inputs)
            blocks, catalog, truth = generate(spec["case"], spec["seed"])
            truth_file = root / f"{study}.truth.json"
            truth_record = dict(case=spec["case"], seed=spec["seed"], arrays_key=arrays_digest(blocks),
                                truth={k: a.tolist() for k, a in truth.items()})
            if truth_file.exists() and json.loads(truth_file.read_text()) != truth_record:
                raise ValueError("Synthetic truth changed; do not overwrite a previous experiment")
            if not truth_file.exists():
                atomic_json(truth_file, truth_record)
            cache = ROOT / "outputs" / "championship" / "cache" / registration["id"] / study
            post, receipt = run(blocks, catalog, cache, registration_path, study=study, synthetic=True)
            # Compact, portable evidence excludes the potentially large posterior.
            result = dict(study=study, state="converged", receipt=receipt,
                          truth_sha256=digest(truth_file), recovery=recovery_metrics(post, truth, catalog))
            del post
        except Exception as exc:
            cache = ROOT / "outputs" / "championship" / "cache" / registration["id"] / study
            saved = cache / "receipt.json"
            result = dict(study=study, state="failed", error=type(exc).__name__, message=str(exc),
                          receipt=json.loads(saved.read_text()) if saved.exists() else None)
        result_file = root / f"{study}.json"
        if result_file.exists() and json.loads(result_file.read_text()) != result:
            raise ValueError("Study result already recorded; inspect it instead of replacing it")
        if not result_file.exists():
            atomic_json(result_file, result)
        results.append(dict(study=study, state=result["state"]))
        print(json.dumps(results[-1]), flush=True)
    require_snapshot(inputs)
    summary = dict(registration=str(registration_path.relative_to(ROOT)), studies=results,
                   all_processed=len(results) == len(registration["studies"]),
                   all_converged=all(x["state"] == "converged" for x in results),
                   synthetic_recovery_complete=False, sbc_complete=False, publication_allowed=False,
                   source=inputs, purpose=registration["purpose"])
    atomic_json(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    result = execute(args.registration)
    raise SystemExit(0 if result["all_converged"] else 1)
