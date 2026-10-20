"""Portable Overall statistics and payload validation, without publication authority.

The producer must separately verify registrations, source/archive identities and
scientific reports. These pure checks cannot turn success flags into evidence,
read a posterior, set an acceptance decision, or advance a public pointer.
"""
import copy
import json
import math
from pathlib import Path
import re

CONTRACT = json.loads((Path(__file__).resolve().parents[2] / "contracts/overall.json").read_text())
IDENTITY = ("fit_id", "posterior_key", "scenario_key", "simulation_key")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value):
    return type(value) is int and 0 < value <= 2**53 - 1


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bounded(value, low, high):
    return _number(value) and low <= value <= high


def _close(a, b):
    return abs(a - b) <= 32 * math.ulp(1.) * max(1., abs(a), abs(b))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _context(evidence):
    _require(isinstance(evidence, dict), "Missing Overall evidence context")
    _require(all(_text(evidence.get(k)) for k in ("fit_id", "recorded_at", "data_as_of")),
             "Incomplete Overall fit/cutoff context")
    _require(all(isinstance(evidence.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", evidence[k])
                 for k in IDENTITY[1:]), "Missing posterior, scenario or simulation identity")
    scenario, field = evidence.get("scenario"), evidence.get("field")
    _require(isinstance(scenario, dict) and _text(scenario.get("id")) and _text(scenario.get("label"))
             and _integer(scenario.get("races")), "Invalid common scenario")
    _require(isinstance(field, list) and len(field) >= 2 and all(_text(d) for d in field)
             and len(set(field)) == len(field), "Invalid current field")
    _require(_integer(evidence.get("posterior_draws")) and _integer(evidence.get("replicates_per_draw")),
             "Invalid posterior or replicate inventory")
    return evidence


def _moments(value, *, difference, replicates):
    maximum, limit = CONTRACT["points"][0], CONTRACT["max_conditional_points_mcse"]
    low = -maximum if difference else 0
    _require(isinstance(value, dict), "Missing paired point summary")
    _require(_bounded(value.get("mean"), low, maximum) and _bounded(value.get("q05"), low, maximum)
             and _bounded(value.get("q95"), value["q05"], maximum), "Invalid expected-point interval")
    _require(_number(value.get("posterior_sd")) and value["posterior_sd"] >= 0,
             "Invalid posterior uncertainty")
    bound = maximum / math.sqrt(replicates) / (1 if difference else 2)
    _require(_bounded(value.get("mcse"), 0, limit)
             and _bounded(value.get("max_conditional_mcse"), value["mcse"], limit)
             and _bounded(value.get("conditional_mcse_bound"), 0, limit)
             and _close(value["conditional_mcse_bound"], bound)
             and value.get("conditional_precision_passed") is True,
             "Incomplete or failed conditional simulation precision")


def _identity_matches(row, evidence):
    return (all(row.get(k) == evidence[k] for k in IDENTITY)
            and row.get("scenario_id") == evidence["scenario"]["id"])


def check_components(components):
    """Check paired statistics and inventories; this supplies no scientific gate."""
    _require(isinstance(components, dict), "Missing Overall components")
    e = _context(components.get("evidence_context"))
    field, reps = set(e["field"]), e["replicates_per_draw"]
    standings = components.get("standings")
    _require(isinstance(standings, list) and len(standings) == len(field)
             and all(isinstance(row, dict) and _text(row.get("driver_id")) for row in standings),
             "Incomplete standings inventory")
    _require({r["driver_id"] for r in standings} == field, "Standings do not match the current field")
    for row in standings:
        _require(_text(row.get("name")), "Missing current driver name")
        _moments(row.get("expected_points"), difference=False, replicates=reps)
        rank = row.get("season_rank")
        _require(_bounded(row.get("p_title"), 0, 1) and isinstance(rank, dict)
                 and _bounded(rank.get("q05"), 1, len(field))
                 and _bounded(rank.get("q95"), rank["q05"], len(field))
                 and _bounded(rank.get("median"), rank["q05"], rank["q95"]),
                 "Invalid season outcome uncertainty")
    _require(abs(sum(row["p_title"] for row in standings) - 1) <= CONTRACT["title_probability_sum_tolerance"],
             "Title probabilities do not sum to unity")
    explanations = components.get("contributions")
    _require(isinstance(explanations, list) and len(explanations) == len(field)
             and all(isinstance(row, dict) and _text(row.get("driver_id")) for row in explanations)
             and {r["driver_id"] for r in explanations} == field, "Incomplete explanation inventory")
    for row in explanations:
        _require(_identity_matches(row, e), "Explanation uses a different posterior or scenario")
        losses = row.get("losses")
        _require(isinstance(losses, dict) and set(losses) == set(CONTRACT["required_explanations"]),
                 "Explanation components do not match the contract")
        for value in losses.values():
            _moments(value, difference=True, replicates=reps)
    pairs = components.get("pairwise")
    _require(isinstance(pairs, list) and len(pairs) == len(field) * (len(field) - 1) // 2,
             "Incomplete paired comparison inventory")
    seen = set()
    means = {r["driver_id"]: r["expected_points"]["mean"] for r in standings}
    for pair in pairs:
        _require(isinstance(pair, dict) and _text(pair.get("driver_a")) and _text(pair.get("driver_b")),
                 "Invalid paired driver identities")
        a, b = pair["driver_a"], pair["driver_b"]
        key = tuple(sorted((a, b)))
        _require(a in field and b in field and a != b and key not in seen,
                 "Duplicate, self or unknown driver comparison")
        seen.add(key)
        _require(_identity_matches(pair, e), "Comparison uses a different posterior or scenario")
        value = pair.get("points_difference")
        _moments(value, difference=True, replicates=reps)
        _require(_close(value["mean"], means[a] - means[b]), "Paired mean disagrees with the standings")
    return components


def prepare_components(summary, *, names, scenario, data_as_of, recorded_at):
    """Translate a complete paired summary without inferring scientific acceptance.

    All intervals and numerical errors come directly from the joint summary;
    marginal intervals are never subtracted to manufacture a paired interval.
    The returned components have no public schema/status or acceptance decision.
    """
    _require(isinstance(summary, dict) and summary.get("model_version") == CONTRACT["model_version"]
             and summary.get("target") == CONTRACT["target"], "Unsupported Overall summary")
    _require(all(summary.get(k) is True for k in ("complete_draw_and_replicate_inventory",
             "expected_points_precision_passed", "explanation_precision_passed", "pairwise_points_precision_passed")),
             "Incomplete summary or failed conditional precision")
    axes = summary.get("posterior_shape")
    _require(isinstance(axes, list) and len(axes) == 2 and all(_integer(n) for n in axes),
             "Missing grouped posterior inventory")
    identity = summary.get("identity")
    _require(isinstance(identity, dict) and set(identity) == set(IDENTITY), "Missing joint summary identity")
    _require(isinstance(scenario, dict) and scenario.get("races") == summary.get("n_races"),
             "Scenario and simulated race inventory disagree")
    context = dict(**identity, recorded_at=recorded_at, data_as_of=data_as_of,
                   field=copy.deepcopy(summary["field"]), scenario=copy.deepcopy(scenario),
                   posterior_draws=math.prod(axes), replicates_per_draw=summary["replicates_per_draw"])
    _context(context)
    _require(isinstance(names, dict) and all(_text(names.get(d)) for d in context["field"]),
             "Current field lacks driver names")
    rows = copy.deepcopy(summary["standings"])
    for row in rows:
        _require(row.get("driver_id") in names, "Unknown driver in summary")
        row["name"] = names[row["driver_id"]]
    explanations = copy.deepcopy(summary["contributions"])
    for row in explanations:
        _require(all(row.get(k) == identity[k] for k in IDENTITY), "Mixed explanation identity in summary")
        row["scenario_id"] = scenario["id"]
    pairs = []
    for pair in summary["pairwise"]:
        # Finite-simulation ordering frequencies are not validated posterior
        # rank probabilities. Preserve only the supported paired point summary.
        pairs.append(dict(driver_a=pair["driver_a"], driver_b=pair["driver_b"], **identity,
                          scenario_id=scenario["id"], points_difference=copy.deepcopy(pair["points_difference"])))
    result = dict(purpose="overall_export_components", model_version=CONTRACT["model_version"],
                  target=CONTRACT["target"], evidence_context=context, standings=rows,
                  contributions=explanations, pairwise=pairs,
                  scientific_evidence_verified=False, publication_allowed=False)
    return check_components(result)


def check_payload(payload):
    """Validate an established payload's shape, after independent evidence checks.

    Returning successfully establishes internal consistency only. The canonical
    producer still owns authenticated scientific acceptance and source checking.
    """
    _require(isinstance(payload, dict) and type(payload.get("schema_version")) is int
             and payload["schema_version"] == CONTRACT["schema_version"]
             and payload.get("model_version") == CONTRACT["model_version"]
             and payload.get("target") == CONTRACT["target"] and payload.get("status") == "established",
             "Unsupported or unestablished Overall payload")
    _require(_text(payload.get("reason")) and isinstance(payload.get("limitations"), list)
             and all(_text(x) for x in payload["limitations"]), "Invalid publication explanation")
    evidence = _context(payload.get("evidence"))
    checks = evidence.get("checks")
    _require(isinstance(checks, dict) and all(isinstance(checks.get(k), dict)
             and checks[k].get("passed") is True and _text(checks[k].get("source"))
             and _text(checks[k].get("reason")) for k in CONTRACT["required_checks"]),
             "Missing or failed Overall acceptance check")
    check_components(dict(evidence_context=evidence, standings=payload.get("standings"),
                          contributions=payload.get("contributions"), pairwise=payload.get("pairwise")))
    return payload
