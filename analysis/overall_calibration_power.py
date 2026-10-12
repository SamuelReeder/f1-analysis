"""Pre-fit sensitivity planning for the complete current SBC quantity family.

Exact binomial probabilities evaluate the existing CDF tests under declared rank
distributions. No posterior, empirical SBC ranks, real scores or fitted truths
are read. Planning constants are not a frozen full-study acceptance protocol.
"""
import argparse
import datetime as dt
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import binom, binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fixed planning grid, committed before this calculation is run. These do not
# authorize fits or choose the eventual full-study inventory.
REPLICATE_GRID = (25, 50, 100, 200, 400, 800, 1600, 3200, 6400)
MIXTURE_WEIGHTS = (.1, .25)
PLANNING_FAMILY_ERROR = .05
PLANNING_POWER = .8


@lru_cache(maxsize=32)
def rejection_regions(replicates, chains, family_size, alpha):
    if (type(replicates) is not int or replicates < 1 or type(chains) is not int or chains < 2
            or type(family_size) is not int or family_size < chains
            or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1):
        raise ValueError("Invalid power calculation design")
    regions = []
    for cutoff in range(chains):
        p0 = (cutoff + 1) / (chains + 1)
        # Match summarize's two-sided exact test and STRICT adjusted threshold.
        regions.append(tuple(k for k in range(replicates + 1)
            if min(1., binomtest(k, replicates, p0).pvalue * family_size) < alpha))
    return tuple(regions)


def power_bounds(rank_probabilities, *, replicates, family_size, alpha):
    probabilities = np.asarray(rank_probabilities, float)
    if (probabilities.ndim != 1 or len(probabilities) < 3 or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("A complete rank probability distribution is required")
    probabilities = probabilities / probabilities.sum()
    chains = len(probabilities) - 1
    rows = []
    for cutoff, rejected in enumerate(rejection_regions(replicates, chains, family_size, alpha)):
        p0, p1 = (cutoff + 1) / (chains + 1), float(probabilities[:cutoff + 1].sum())
        accepted = np.ones(replicates + 1, dtype=bool)
        accepted[list(rejected)] = False
        remaining = np.flatnonzero(accepted)
        if not len(remaining) or not accepted[remaining[0]:remaining[-1] + 1].all():
            raise ValueError("Unexpected noncontiguous acceptance region")
        def probability(p):
            # Sum the two disjoint tails using CDF/SF, avoiding a long PMF sum.
            value = float(binom.cdf(remaining[0] - 1, replicates, p)
                          + binom.sf(remaining[-1], replicates, p))
            tolerance = np.finfo(float).eps * (replicates + 1)
            if not np.isfinite(value) or not -tolerance <= value <= 1 + tolerance:
                raise ValueError("Binomial tail probability is outside rounding tolerance")
            return min(1., max(0., value))
        power, size = probability(p1), probability(p0)
        rows.append(dict(cutoff=cutoff, null_probability=p0, alternative_probability=p1,
            rejection_counts=list(rejected), null_rejection_probability=size, rejection_probability=power))
    # Cutoffs are dependent, as are driver quantities. Do not multiply their
    # non-rejection probabilities or call this an exact whole-family power.
    return dict(replicates=replicates, rank_probabilities=probabilities.tolist(), cutoffs=rows,
        affected_quantity_power_lower_bound=max(r["rejection_probability"] for r in rows),
        affected_quantity_power_upper_bound=min(1., sum(r["rejection_probability"] for r in rows)))


def binary_likelihood_ranks(chains, *, posterior_match_probability):
    """Exact ranks from the data-ignoring binary example, including random ties."""
    truth_match = .9
    result = np.zeros(chains + 1)
    for matching in range(chains + 1):
        mass = binom.pmf(matching, chains, posterior_match_probability)
        result[chains - matching:] += truth_match * mass / (matching + 1)
        result[:chains - matching + 1] += (1 - truth_match) * mass / (chains - matching + 1)
    return result


def alternatives(chains):
    uniform = np.full(chains + 1, 1 / (chains + 1))
    result = dict(uniform_reference=uniform)
    for weight in MIXTURE_WEIGHTS:
        for name, positions in (("lower_end", [0]), ("both_ends", [0, chains]), ("middle", [chains // 2])):
            distribution = (1 - weight) * uniform
            distribution[positions] += weight / len(positions)
            result[f"{name}_mixture_{weight}"] = distribution
    result["binary_prior_only_inference"] = binary_likelihood_ranks(chains, posterior_match_probability=.5)
    return result


def current_inventory():
    from analysis.overall_recovery_checks import ROLES, quantity_specs
    from analysis.overall_ranking_recovery import specs as ranking_specs
    from f1rank.overall.recovery import synthetic_design
    result = []
    for case, role in ROLES.items():
        design = synthetic_design(case)
        blocks = dict(qualifying=design.qualifying.arrays(), laps=design.laps, outcomes=design.outcomes)
        base = quantity_specs(blocks, design.catalog)
        ranking = ranking_specs(design.catalog["field"])
        identities = sorted(s["id"] for s in [*base, *ranking])
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate planning quantity")
        result.append(dict(case=case, role=role, field_size=len(design.catalog["field"]),
            base_quantities=len(base), ranking_quantities=len(ranking), quantities=len(identities),
            quantity_inventory_sha256=hashlib.sha256(json.dumps(identities).encode()).hexdigest()))
    return result


def run():
    from f1rank.overall.fitting import FIT_SETTINGS, RETRY, require_clean_inputs
    from f1rank.overall.provenance import snapshot, require_snapshot
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = ["analysis/overall_calibration_power.py", "analysis/overall_recovery_checks.py",
             "analysis/overall_ranking_recovery.py"]
    hashes = {}
    for name in paths:
        content = (ROOT / name).read_bytes()
        if subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT) != content:
            raise ValueError("Commit the planning grid and evaluator before calculation")
        hashes[name] = hashlib.sha256(content).hexdigest()
    inputs = snapshot()
    require_clean_inputs(inputs)
    inventory = current_inventory()
    chains = FIT_SETTINGS["chains"]
    family = chains * sum(row["quantities"] for row in inventory if row["role"] == "prior_predictive")
    rows = []
    for name, probabilities in alternatives(chains).items():
        values = [power_bounds(probabilities, replicates=n, family_size=family, alpha=PLANNING_FAMILY_ERROR)
                  for n in REPLICATE_GRID]
        # Keep rejection bounds compact; exact sets remain reproducible in code.
        for value in values:
            for cutoff in value["cutoffs"]:
                counts = cutoff.pop("rejection_counts")
                cutoff["rejection_count_intervals"] = []
                for k in counts:
                    intervals = cutoff["rejection_count_intervals"]
                    if intervals and k == intervals[-1][-1] + 1: intervals[-1][-1] = k
                    else: intervals.append([k, k])
        eligible = [v["replicates"] for v in values if v["affected_quantity_power_lower_bound"] >= PLANNING_POWER]
        rows.append(dict(alternative=name, rank_probabilities=probabilities.tolist(), calculations=values,
                         first_grid_size_meeting_illustrative_power=eligible[0] if eligible else None))
    require_snapshot(inputs)
    return dict(purpose="pre_fit_calibration_power_planning", recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=source, source_hashes=hashes, model_inputs_key=inputs["key"], inventory=inventory,
        chains=chains, family_size=family, assumed_family_error_probability=PLANNING_FAMILY_ERROR,
        illustrative_power_target=PLANNING_POWER, replicate_grid=list(REPLICATE_GRID), alternatives=rows,
        balanced_workload=[dict(replicates_per_case=n, total_studies=n * len(inventory),
            prior_predictive_studies=n * sum(r["role"] == "prior_predictive" for r in inventory),
            maximum_fit_attempts=n * len(inventory) * len(RETRY),
            functional_evaluations_if_all_fits_converge=n * len(inventory) * (chains + 1)) for n in REPLICATE_GRID],
        scope="Power bounds for rejection within one affected quantity, with the complete-family Bonferroni threshold. Rank-CDF cutoffs are dependent; other quantity dependence is unspecified.",
        assumptions=["Independent complete replicates and the stated rank distribution",
            "The current fixed-design quantity inventory, including finite-simulation ranking functions",
            "No missing or failed study and no seed reuse"],
        limitations=["Synthetic rank alternatives are not measured inference errors in the F1 model",
            "Bounds do not specify the joint whole-family power or detect every possible error",
            "The first grid size meeting an illustrative target is not a selected or approved study size",
            "No fit runtime, full-study cost or intended-runner feasibility is established",
            "Original numerical failure and pending sampling approval remain unresolved"],
        selected_study_size=None, full_registration_frozen=False, posterior_ranks_read=False,
        model_fits_run=False, establishes_model_calibration=False, publication_allowed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    if path.exists():
        raise FileExistsError("Preserve the existing planning record")
    result = run()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"family_size":result["family_size"], "inventory":result["inventory"],
        "planning_sizes":{r["alternative"]:r["first_grid_size_meeting_illustrative_power"] for r in result["alternatives"]}}))
