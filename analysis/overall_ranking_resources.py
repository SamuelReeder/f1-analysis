"""Capped local measurement of one synthetic ranking-functional evaluation.

Field size and calendar shape come from committed artifacts. Parameter values
are arbitrary synthetic constants; no fitted driver values or table are saved.
This measures local simulation cost, not fitting, a full SBC study or CI runtime.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.overall_ranking_recovery import from_states
from f1rank.overall.simulation import MAX_EXPECTED_POINTS_MCSE, POINTS, QUALIFYING_FIELDS


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def scope_limits():
    line = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/")
    memory, cpu = ((group / name).read_text().strip() for name in ("memory.max", "cpu.max"))
    quota, period = cpu.split()
    if memory == "max" or int(memory) > 3 * 1024**3 or quota == "max" or int(quota) > 2 * int(period):
        raise ValueError("Use the required capped systemd scope")
    return dict(cgroup=str(group), memory_max=memory, cpu_max=cpu)


def run():
    limits = scope_limits()
    names = ["analysis/overall_ranking_resources.py", "analysis/overall_ranking_recovery.py",
        "f1rank/overall/simulation.py", "f1rank/overall/order.py", "f1rank/overall/scoring.py",
        "docs/overall_reference_calendar.json", "outputs/championship/development/interface_preflight.json",
        "docs/overall_rotation_profile_proposal.json", "data/processed/race.parquet"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {name: sha(ROOT / name) for name in names}
    for name in names:
        saved = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(saved).hexdigest() != hashes[name]:
            raise ValueError("Commit the resource workload and its inputs before measuring")
    reference = json.loads((ROOT / "docs/overall_reference_calendar.json").read_text())
    preflight = json.loads((ROOT / "outputs/championship/development/interface_preflight.json").read_text())
    if preflight["source"]["inputs"]["data/processed/race.parquet"] != hashes["data/processed/race.parquet"]:
        raise ValueError("The field-size reference no longer describes the saved race data")
    n, contexts = preflight["counts"]["field"], len(reference["events"])
    # Smallest even workload satisfying the pre-result bounded-MCSE policy.
    seasons = 2 * math.ceil((float(np.max(POINTS)) / (2 * MAX_EXPECTED_POINTS_MCSE))**2 / 2)
    policy = dict(scenario_id="synthetic-resource-" + reference["scenario_id"],
        target="finite_simulation_equal_machinery_and_team", seasons=seasons, seed=20261012,
        max_points_mcse=MAX_EXPECTED_POINTS_MCSE, precision_policy="bounded_points_mcse_fixed_workload_no_retry",
        advancing=list(QUALIFYING_FIELDS[n]),
        scenarios=[dict(event_id=row["event_id"], context=i, laps=row["reference_distance_laps"],
            consecutive_racing_laps=row["reference_distance_laps"]) for i, row in enumerate(reference["events"])])
    states = dict(skill=np.zeros(n), overall_race_skill=np.zeros(n), sd_driver_form=np.array(.2),
        overall_sd_day=np.array(.3), overall_shared_weekend=np.array(1.), sigma0=np.array(.5),
        sigma_tau=np.array(.5), nu=np.array(4.), overall_grid=np.ones(contexts),
        overall_pace=np.ones(contexts), overall_hazard=np.zeros(contexts))
    catalog = dict(field=[f"synthetic_{i}" for i in range(n)], field_entries=list(range(n)))
    begun = time.monotonic()
    quantities, numerics = from_states(states, catalog, policy)
    seconds = time.monotonic() - begun
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError("Resource inputs changed while evaluating")
    if not numerics["points_precision_passed"]:
        raise ValueError("The fixed synthetic workload missed its precision check")
    return dict(purpose="synthetic_ranking_functional_resource_check", source_commit=commit,
        source_hashes=hashes, limits=limits, recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        field_size=n, reference_cutoff=preflight["cutoff"], parameter_values_evaluated=1,
        policy=policy, seasons=seasons, reference_events=contexts,
        synthetic_parameter_values={k:v.tolist() for k, v in states.items()},
        output_quantities=len(quantities), quantities_sha256=hashlib.sha256(
            json.dumps(quantities, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        every_output_finite=bool(np.isfinite(list(quantities.values())).all()),
        numerical_summary={k:v for k, v in numerics.items() if k != "points_mcse"},
        seconds=seconds, process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        runtime=dict(python=sys.version, numpy=np.__version__),
        limitations=["Arbitrary synthetic states, not fitted parameters or a recovery experiment",
            "Explicit full-racing-distance and zero-retirement resource scenario, not a calibrated incident model",
            "One parameter value; full-posterior and complete-study runtime are unmeasured",
            "Local capped scope only; intended CI runner remains unverified"],
        mcmc_run=False, real_driver_table_saved=False, intended_ci_runner_verified=False,
        publication_allowed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Preserve the existing resource record")
    result = run()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:result[k] for k in ("field_size", "seasons", "reference_events", "seconds", "process_max_rss_kib")}))
