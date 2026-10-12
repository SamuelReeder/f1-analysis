"""Known-truth quantities and complete-inventory SBC checks; no fitting entrypoint.

The original numerical profile is not a full calibration study. A future study
must freeze its inventory, error budget and sources before generating outcomes.
These checks cannot grant publication or turn missing/failed fits into successes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from f1rank.overall.fitting import arrays_digest, committed_registration, sampling_seeds, study_settings
from f1rank.overall.recovery import SYNTHETIC_CASES, TRUTH_SITES, generate


ROLES = {case: "stress" if case == "dominant_teammate" else "prior_predictive"
         for case in SYNTHETIC_CASES}
OBSERVATIONS = ("y", "overall_laps", "overall_finish_order", "overall_completion")
DRIVER_SITES = ("skill", "overall_race_skill", "compat", "overall_team")
CAR_SITES = ("car", "overall_car")
CONTEXT_SITES = ("overall_grid", "overall_pace", "overall_hazard")
# The existing recovery_metrics reports the q05--q95 interval.
RECOVERY_INTERVAL_PROBABILITY = .9


def _finite(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"Nonfinite or nonnumeric {name}; preserve this failed replicate")
    return a


def generate_with_parameters(case, seed):
    """Capture the original generator's latents without adding any random site.

    The outer trace observes its existing execution. Generation failures are
    propagated, never retried with another truth. The fixed dominant-teammate
    scenario is a stress case, not a draw from the fitted prior.
    """
    from numpyro import handlers
    if case not in ROLES or type(seed) is not int or seed < 0:
        raise ValueError("Unknown case or invalid generation seed")
    with handlers.trace() as trace:
        blocks, catalog, truth = generate(case, seed)
    parameters = {name: _finite(site["value"], name) for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
        and name != "y" and not name.startswith("generated_")}
    return dict(case=case, seed=seed, role=ROLES[case], blocks=blocks, catalog=catalog,
                truth=truth, parameters=parameters, arrays_key=arrays_digest(blocks),
                parameters_key=arrays_digest({"truth_parameters": parameters}),
                truth_key=arrays_digest({"truth": truth}))


def quantity_specs(blocks, catalog, *, ranking_protocol=None):
    """All declared current-field contrasts and contexts, selected without values."""
    field, indices = catalog["field"], catalog["field_entries"]
    if len(field) != len(indices) or len(field) < 2 or len(set(field)) != len(field):
        raise ValueError("Recovery needs a complete, uniquely identified current field")
    q, o = blocks["qualifying"], blocks["outcomes"]
    if (len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < q["n_entries"] for i in indices)):
        raise ValueError("Invalid current-entry references")
    result = []
    def contrast(sites, identities, positions, kind):
        for site in sites:
            for i in range(len(positions)):
                for j in range(i):
                    identity = json.dumps([site, kind, identities[i], identities[j]], separators=(",", ":"))
                    result.append(dict(id=identity, site=site, indices=[positions[i], positions[j]]))
    contrast(DRIVER_SITES, field, indices, "driver_difference")
    cars = sorted({int(o["entry_car"][i]) for i in indices})
    contrast(CAR_SITES, cars, cars, "car_difference")
    for site in CONTEXT_SITES:
        for index in range(len(o["context_circuit"])):
            result.append(dict(id=json.dumps([site, "context", index], separators=(",", ":")),
                               site=site, indices=[index]))
    result.extend(dict(id="likelihood:" + site, site=site, likelihood=True) for site in (*OBSERVATIONS, "joint"))
    if ranking_protocol is not None:
        from analysis.overall_ranking_recovery import specs, validate_protocol
        validate_protocol(ranking_protocol, field_size=len(field), n_contexts=len(o["context_circuit"]))
        result.extend(specs(field))
    return result


def quantity_values(blocks, catalog, parameters, *, ranking_protocol=None, with_numerics=False):
    """Evaluate the SAME generated responses at truth or a posterior parameter.

    The joint-likelihood quantity sums observation factors, not priors. Supplying
    an observed site as a parameter would replace the data and is rejected.
    All stochastic parameter sites must be supplied; no latent is redrawn.
    """
    import jax.numpy as jnp
    from numpyro import handlers
    from f1rank.overall.model import model
    if any(name in OBSERVATIONS or name.startswith("generated_") for name in parameters):
        raise ValueError("Observed responses cannot be supplied as latent parameters")
    # Use the same input conversion as the fitter and synthetic generator.
    convert = lambda block: {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in block.items()}
    trace = handlers.trace(handlers.substitute(model,
        data={k: jnp.asarray(v) for k, v in parameters.items()})).get_trace(
        *[convert(blocks[name]) for name in ("qualifying", "laps", "outcomes")])
    stochastic = {name for name, site in trace.items() if site["type"] == "sample" and not site["is_observed"]}
    if stochastic != set(parameters):
        raise ValueError("Supplied parameters do not match the model's stochastic sites")
    likelihood = {name: float(_finite(trace[name]["fn"].log_prob(trace[name]["value"]), name).sum())
                  for name in OBSERVATIONS}
    likelihood["joint"] = sum(likelihood.values())
    values = {}
    for spec in quantity_specs(blocks, catalog):
        if spec.get("likelihood"):
            value = likelihood[spec["site"]]
        else:
            value = np.asarray(trace[spec["site"]]["value"])[spec["indices"]]
            value = value[0] - value[1] if len(value) == 2 else value[0]
        values[spec["id"]] = float(_finite(value, spec["id"]))
    numerics = None
    if ranking_protocol is not None:
        from analysis.overall_ranking_recovery import SITES, from_states
        extra, numerics = from_states({s: trace[s]["value"] for s in SITES.values()}, catalog, ranking_protocol)
        values.update(extra)
    return (values, numerics) if with_numerics else values


def rank_statistic(draws, truth, *, tie_uniform):
    """Randomized discrete rank, including the generating truth in a tied group."""
    values, true = _finite(draws, "rank draws"), _finite(truth, "rank truth")
    if values.ndim != 1 or not len(values) or true.ndim != 0 or not 0 <= tie_uniform < 1:
        raise ValueError("Expected scalar truth, nonempty draws and tie uniform in [0, 1)")
    below, ties = int((values < true).sum()), int((values == true).sum())
    rank = below + int(np.floor(tie_uniform * (ties + 1)))
    return dict(rank=rank, below=below, ties=ties, posterior_draws=len(values))


def _rng(seed, identity):
    if type(seed) is not int or seed < 0:
        raise ValueError("Randomization seed must be an explicit nonnegative integer")
    data = json.dumps(["overall_recovery_checks", seed, identity], separators=(",", ":")).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(data).digest(), "big"))


def recovery_intervals(generated, posterior):
    """Describe current-state recovery using all draws, separately from SBC ranks.

    All-pair contrasts are formed within joint draws. Fixed-truth stress coverage
    has no uniform-rank or nominal average-coverage interpretation. Likelihood
    quantities are used by SBC; this interval report covers the state quantities.
    """
    if not set(TRUTH_SITES) <= set(posterior):
        raise ValueError("Missing posterior state sites for recovery")
    shape = np.shape(posterior[TRUTH_SITES[0]])[:2]
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Grouped recovery state axes are required")
    for site in TRUTH_SITES:
        values = np.asarray(posterior[site])
        if values.shape != shape + np.shape(generated["truth"][site]):
            raise ValueError("Recovery states do not share their declared axes")
        for chain in range(shape[0]):
            for start in range(0, shape[1], 128):
                _finite(values[chain, start:start + 128], site)
    tail = (1 - RECOVERY_INTERVAL_PROBABILITY) / 2
    rows = []
    for spec in quantity_specs(generated["blocks"], generated["catalog"]):
        if spec.get("likelihood"):
            continue
        indices, site = spec["indices"], spec["site"]
        values = np.asarray(posterior[site])[..., indices[0]]
        truth = np.asarray(generated["truth"][site])[indices[0]]
        if len(indices) == 2:
            values = values - np.asarray(posterior[site])[..., indices[1]]
            truth = truth - np.asarray(generated["truth"][site])[indices[1]]
        flat = _finite(values, spec["id"]).reshape(-1)
        truth = float(_finite(truth, spec["id"]))
        low, median, high = np.quantile(flat, [tail, .5, 1 - tail])
        rows.append(dict(quantity=spec["id"], truth=truth, mean=float(flat.mean()),
            median=float(median), mean_error=float(flat.mean() - truth), interval=[float(low), float(high)],
            interval_probability=RECOVERY_INTERVAL_PROBABILITY, covered=bool(low <= truth <= high),
            posterior_draws=len(flat)))
    return rows


def evaluate(generated, posterior, *, study_id, randomization_seed, ranking_protocol=None):
    """One value-independent draw per chain, shared across every test quantity.

    This avoids using correlated iterations within a chain as independent SBC
    ranks. Independent chains and adequate inference still need verified fit
    evidence. Coarse ranks require a prespecified study-size/power justification.
    Direct calls provide arithmetic evidence only; use evaluate_checked for fits.
    """
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("A study identity is required")
    truth_parameters = generated["parameters"]
    if (generated["case"] not in ROLES or generated["role"] != ROLES[generated["case"]]
            or arrays_digest(generated["blocks"]) != generated["arrays_key"]
            or arrays_digest({"truth_parameters": truth_parameters}) != generated["parameters_key"]
            or arrays_digest({"truth": generated["truth"]}) != generated["truth_key"]):
        raise ValueError("Generated responses, truths or case role changed")
    if not truth_parameters or not set(truth_parameters) <= set(posterior):
        raise ValueError("Missing posterior parameters")
    shape = np.shape(posterior[next(iter(truth_parameters))])[:2]
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Grouped chain/draw axes are required")
    if any(np.shape(posterior.get(name))[:2] != shape for name in TRUTH_SITES):
        raise ValueError("State and parameter draws do not share chain/draw axes")
    selected = _rng(randomization_seed, [study_id, "chain_draws"]).integers(shape[1], size=shape[0])
    for name, truth in truth_parameters.items():
        values = np.asarray(posterior[name])
        if values.shape != shape + np.shape(truth):
            raise ValueError(f"Mismatched posterior axes for {name}")
        for chain in range(shape[0]):
            for start in range(0, shape[1], 128):
                _finite(values[chain, start:start + 128], name)
    blocks, catalog = generated["blocks"], generated["catalog"]
    quantity_specs(blocks, catalog, ranking_protocol=ranking_protocol)
    truth, true_numerics = quantity_values(blocks, catalog, truth_parameters,
        ranking_protocol=ranking_protocol, with_numerics=True)
    evaluated = [quantity_values(blocks, catalog,
        {name: np.asarray(posterior[name])[chain, draw] for name in truth_parameters},
        ranking_protocol=ranking_protocol, with_numerics=True)
        for chain, draw in enumerate(selected)]
    samples = [value for value, _ in evaluated]
    numerics = ([dict(role="truth", numerics=true_numerics)] +
        [dict(role="posterior", chain=chain, draw=int(selected[chain]), numerics=report)
         for chain, (_, report) in enumerate(evaluated)]) if ranking_protocol is not None else []
    precise = all(row["numerics"]["points_precision_passed"] for row in numerics)
    ranks = {}
    for name, value in truth.items():
        random = _rng(randomization_seed, [study_id, "tie", name]).random()
        ranks[name] = rank_statistic([sample[name] for sample in samples], value, tie_uniform=random)
        ranks[name]["truth"] = value
    return dict(study_id=study_id, case=generated["case"], generation_seed=generated["seed"],
        role=generated["role"], state="evaluated" if precise else "failed_evaluation", arrays_key=generated["arrays_key"],
        evaluation_failure=None if precise else "ranking_functional_simulation_precision",
        ranking_numerics=numerics,
        quantities=ranks, selection_policy="one_draw_per_chain", randomization_seed=randomization_seed,
        recovery_intervals=recovery_intervals(generated, posterior),
        selected_indices=[[chain, int(draw)] for chain, draw in enumerate(selected)],
        posterior_shape=list(shape), fit_verified=False, publication_allowed=False)


def validate_inventory(registration):
    """A complete, fixed inventory includes stress cases and every prior design."""
    if registration.get("purpose") != "recovery_calibration":
        raise ValueError("The numerical profile is not a recovery/calibration registration")
    sampling_seeds(registration)
    studies, synthetic = registration.get("studies"), registration.get("synthetic")
    if (not isinstance(studies, list) or not studies or len(set(studies)) != len(studies)
            or not all(isinstance(s, str) and s for s in studies)
            or not isinstance(synthetic, dict) or set(synthetic) != set(studies)):
        raise ValueError("Missing or duplicate study inventory")
    counts, seeds = {case: 0 for case in ROLES}, set()
    for name in studies:
        spec = synthetic[name]
        case, seed = spec.get("case"), spec.get("seed")
        if (case not in ROLES or spec.get("role") != ROLES[case]
                or type(seed) is not int or seed < 0 or seed in seeds):
            raise ValueError("Invalid case role or reused generation seed")
        counts[case] += 1; seeds.add(seed)
    declared = registration.get("replicates_per_case")
    if (not isinstance(declared, dict) or set(declared) != set(ROLES)
            or any(type(n) is not int or n < 1 for n in declared.values())
            or any(n == 0 for n in counts.values()) or counts != declared):
        raise ValueError("Every required case and its complete replicate count must be registered")
    expected = registration.get("quantities_by_case", {})
    if set(expected) != set(ROLES) or any(not isinstance(ids, list) or not ids
            or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids) for ids in expected.values()):
        raise ValueError("Every case needs a fixed quantity inventory")
    if any("likelihood:joint" not in ids for ids in expected.values()):
        raise ValueError("Data-dependent joint likelihood must be a required quantity")
    ranking = registration.get("ranking_functionals")
    if ranking is not None and (not isinstance(ranking, dict) or set(ranking) != {"source_sha256", "by_case"}
            or not isinstance(ranking["by_case"], dict) or set(ranking["by_case"]) != set(ROLES)
            or not isinstance(ranking["source_sha256"], str) or len(ranking["source_sha256"]) != 64):
        raise ValueError("Ranking functionals require a fixed source and every case protocol")
    if (registration.get("selection_policy") != "one_draw_per_chain"
            or type(registration.get("randomization_seed")) is not int or registration["randomization_seed"] < 0):
        raise ValueError("Unregistered draw-selection policy")
    if registration.get("recovery_interval_probability") != RECOVERY_INTERVAL_PROBABILITY:
        raise ValueError("Registration changes the inherited recovery interval")
    alpha = registration.get("family_error_probability")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ValueError("A fixed family error probability must be declared")
    chains = registration.get("fit_settings", {}).get("chains")
    if type(chains) is not int or chains < 2:
        raise ValueError("Independent chain count must be declared")
    return counts


def evaluate_checked(generated, run_dir, registration_path, *, study_id):
    """Consume a checked, source-matched archive under a frozen new registration."""
    from analysis.overall_posterior_archive import mapped_checked_posterior
    from f1rank.artifacts import digest
    registration = committed_registration(registration_path)
    validate_inventory(registration)
    ranking = registration.get("ranking_functionals")
    ranking_protocol, ranking_path = None, None
    if ranking is not None:
        from analysis import overall_ranking_recovery
        ranking_path = Path(overall_ranking_recovery.__file__)
        if digest(ranking_path) != ranking["source_sha256"]:
            raise ValueError("Ranking functionals use unregistered source")
        ranking_protocol = ranking["by_case"][generated["case"]]
    spec = registration["synthetic"].get(study_id)
    if spec != dict(case=generated["case"], seed=generated["seed"], role=generated["role"]):
        raise ValueError("Generated case is not the registered replicate")
    if set(registration["quantities_by_case"][generated["case"]]) != {
            s["id"] for s in quantity_specs(generated["blocks"], generated["catalog"], ranking_protocol=ranking_protocol)}:
        raise ValueError("Registration omits or changes the required model quantities")
    evaluator_sha = digest(Path(__file__))
    if registration.get("evaluator_sha256") != evaluator_sha:
        raise ValueError("The recovery evaluator is not the registered source")
    registration_sha = digest(registration_path)
    with mapped_checked_posterior(run_dir, names=sorted(set(generated["parameters"]) | set(TRUTH_SITES)),
            expected_arrays_key=generated["arrays_key"]) as (posterior, receipt):
        saved = receipt["specification"]
        expected = dict(source_key=receipt["source"]["key"], arrays_key=generated["arrays_key"],
            catalog=generated["catalog"], registration_sha256=registration_sha,
            study=study_id, synthetic=True, settings=study_settings(registration, study_id))
        expected_key = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        if (saved != expected or receipt["run_key"] != expected_key
                or receipt["catalog"] != generated["catalog"]
                or receipt["settings"] != expected["settings"]):
            raise ValueError("Posterior belongs to a different experiment")
        result = evaluate(generated, posterior, study_id=study_id,
                          randomization_seed=registration["randomization_seed"], ranking_protocol=ranking_protocol)
        result.update(fit_verified=True, run_key=receipt["run_key"],
                      registration_sha256=registration_sha, evaluator_sha256=evaluator_sha)
    # The archive context rechecks immutability before this result can escape.
    if digest(registration_path) != registration_sha or digest(Path(__file__)) != evaluator_sha:
        raise ValueError("Recovery registration or evaluator changed during evaluation")
    if ranking_path is not None and digest(ranking_path) != ranking["source_sha256"]:
        raise ValueError("Ranking functional source changed during evaluation")
    return result


def summarize(records, registration, *, registration_sha256):
    """Exact binomial CDF checks with a fixed Bonferroni family.

    Replicate is the sampling unit. Never pool dependent driver pairs as trials.
    Stress cases remain accounted for but have no uniform-rank reference. Missing
    or failed studies prevent aggregate calibration tests on a selected subset.
    Non-rejection is not a proof of correct inference or adequate test power.
    """
    validate_inventory(registration)
    if (not isinstance(registration_sha256, str) or len(registration_sha256) != 64
            or any(c not in "0123456789abcdef" for c in registration_sha256)):
        raise ValueError("The frozen registration's file hash is required")
    planned = registration["studies"]
    indexed = {row["study_id"]: row for row in records}
    if len(indexed) != len(records) or set(indexed) - set(planned):
        raise ValueError("Duplicate or unregistered study result")
    problems = []
    chains = registration["fit_settings"]["chains"]
    for identity in planned:
        row, spec = indexed.get(identity), registration["synthetic"][identity]
        if row is None:
            problems.append(dict(study=identity, reason="missing")); continue
        if row.get("registration_sha256") != registration_sha256:
            raise ValueError("Result belongs to a different registration")
        if row.get("case") != spec["case"] or row.get("generation_seed") != spec["seed"] or row.get("role") != spec["role"]:
            raise ValueError("Result differs from its registered case, seed or role")
        if row.get("state") != "evaluated" or row.get("fit_verified") is not True:
            problems.append(dict(study=identity, reason="unverified_fit" if row.get("state") == "evaluated"
                                 else row.get("state", "unverified"))); continue
        quantities = row.get("quantities", {})
        if (set(quantities) != set(registration["quantities_by_case"][spec["case"]])
                or row.get("selection_policy") != registration["selection_policy"]
                or row.get("randomization_seed") != registration["randomization_seed"]):
            raise ValueError("Result uses unregistered quantities or randomization")
        for result in quantities.values():
            if (result.get("posterior_draws") != chains or type(result.get("rank")) is not int
                    or not 0 <= result["rank"] <= chains):
                raise ValueError("Invalid rank support")
        ranking = registration.get("ranking_functionals")
        if ranking is not None:
            policy = ranking["by_case"][spec["case"]]
            protocol_sha = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
            reports = row.get("ranking_numerics", [])
            if (len(reports) != chains + 1 or reports[0].get("role") != "truth"
                    or any(r.get("role") != "posterior" or r.get("chain") != c
                           for c, r in enumerate(reports[1:]))):
                raise ValueError("Missing truth or posterior ranking precision reports")
            for r in reports:
                info = r["numerics"]
                if info.get("protocol_sha256") != protocol_sha:
                    raise ValueError("Ranking precision uses a different protocol")
                if info.get("points_precision_passed") is not True:
                    problems.append(dict(study=identity, reason="ranking_functional_simulation_precision"))
                    break
    complete = not problems
    terminal = {"evaluated", "failed_generation", "failed_fit", "failed_evaluation"}
    result = dict(all_required_processed=all(s in indexed and indexed[s].get("state") in terminal for s in planned),
        evaluable=complete,
        problems=problems, tests=[], stress_studies=[s for s in planned if registration["synthetic"][s]["role"] == "stress"],
        no_detected_miscalibration=None, establishes_adequate_power=False,
        establishes_full_recovery=False, publication_allowed=False)
    if not complete:
        return result
    cases = [case for case in ROLES if ROLES[case] == "prior_predictive"]
    family_size = chains * sum(len(registration["quantities_by_case"][case]) for case in cases)
    for case in cases:
        rows = [indexed[s] for s in planned if registration["synthetic"][s]["case"] == case]
        for quantity in registration["quantities_by_case"][case]:
            ranks = np.array([row["quantities"][quantity]["rank"] for row in rows])
            for cutoff in range(chains):
                probability = (cutoff + 1) / (chains + 1)
                count = int((ranks <= cutoff).sum())
                raw = float(binomtest(count, len(rows), probability).pvalue)
                adjusted = min(1., raw * family_size)
                result["tests"].append(dict(case=case, quantity=quantity, cutoff=cutoff,
                    count=count, replicates=len(rows), null_probability=probability,
                    p_value=raw, adjusted_p_value=adjusted,
                    flagged=adjusted < registration["family_error_probability"]))
    result.update(family_size=family_size, family_error_probability=registration["family_error_probability"],
                  no_detected_miscalibration=not any(t["flagged"] for t in result["tests"]))
    return result
