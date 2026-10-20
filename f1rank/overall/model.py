"""Joint successor likelihood. No legacy posterior means or independent draws.

Specification and limitations: docs/overall_candidate.md. Development only until
recovery, chronological validation and new prospective evidence are complete.
"""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from ..model import centre, linear_recurrence
from .order import plackett_luce, perturbed_order
from .qualifying import qualifying
from .studentt import StableStudentT


def normal(name, size, scale=1.):
    return numpyro.sample("overall_" + name, dist.Normal(0, scale).expand([size]))


def scale(name, prior):
    return numpyro.sample("overall_sd_" + name, dist.HalfNormal(prior))


def model(q, laps, outcomes, *, generate=False):
    # The trace observes sites; it does not block them from the enclosing MCMC.
    # Thus race and qualifying update the same skill, car and weekend variables.
    trace = handlers.trace(qualifying).get_trace(q, generate=generate)
    value = lambda key: trace[key]["value"]
    link = numpyro.sample("overall_qualifying_link", dist.Normal(1, .5))
    level = scale("level", .5) * normal("level_z", q["n_drivers"])
    drift_race, drift_season = scale("drift_race", .03), scale("drift_season", .2)
    step = jnp.where(q["entry_same_season"], drift_race * jnp.sqrt(q["entry_event_gap"]),
                     drift_season * jnp.sqrt(jnp.maximum(q["entry_season_gap"], 1)))
    increment = jnp.where(q["entry_first"], level[q["entry_driver"]],
                          step * normal("walk_z", q["n_entries"]))
    residual = linear_recurrence(jnp.where(q["entry_first"], 0., 1.), increment)
    ability = numpyro.deterministic("overall_race_skill", centre(
        link * value("skill") + residual, q["entry_event"], q["n_events"]))
    # The total current race state above is the sole driver pace entering outcomes.
    # No additional qualifying coefficient or old career race residual is added.
    team_link = numpyro.sample("overall_team_link", dist.Normal(1, .5))
    team_extra = scale("team", .1) * normal("team_z", q["n_era_stints"])
    team = numpyro.deterministic("overall_team", centre(
        team_link * value("compat") + team_extra[q["entry_era_stint"]], q["entry_event"], q["n_events"]))
    car_link = numpyro.sample("overall_car_link", dist.Normal(1, .5))
    basis = outcomes["car_basis"]
    car_extra = basis @ normal("car_contrast", basis.shape[1], 1.5)
    car_day = scale("car_day", .4) * normal("car_day_z", q["n_cars"])
    car = numpyro.deterministic("overall_car", centre(
        car_link * (value("car") + value("car_track"))
        + car_extra[q["car_team_season"]] + car_day, q["car_event"], q["n_events"]))
    shared = numpyro.sample("overall_shared_weekend", dist.Normal(1, .5))
    day = numpyro.deterministic("overall_day", centre(
        shared * value("driver_form") + scale("day", .3) * normal("day_z", q["n_entries"]),
        q["entry_event"], q["n_events"]))
    total = ability + team + day + car[outcomes["entry_car"]]
    r = laps
    mean = (total[r["entry"]] + normal("intercept", r["n_race"], 3)[r["race"]]
            + normal("trend", r["n_race"], 2)[r["race"]] * r["lap"]
            + jnp.where(r["nonref"], normal("compound", r["n_rc"], 3)[r["rc"]], 0))
    wear = centre(scale("wear", .05) * normal("wear_z", r["n_dr"]), r["dr_race"], r["n_race"])
    mean += (normal("tyre_slope", r["n_rc"], .3)[r["rc"]] + wear[r["dr"]]) * r["age"]
    traffic = numpyro.sample("overall_traffic", dist.Normal(0, 1).expand([r["n_race"], 3]))
    mean += (traffic[r["race"]] * r["traffic"]).sum(-1)
    rho = numpyro.sample("overall_lap_rho", dist.Uniform(-.5, .95))
    sigma = scale("lap", 1)
    nu = numpyro.sample("overall_lap_nu", dist.Gamma(2, .1))
    residual_lap = r["y"] - mean
    previous = jnp.concatenate([jnp.zeros(1), residual_lap[:-1]])
    location = jnp.where(r["prev"], rho * previous, 0)
    noise_scale = jnp.where(r["prev"], sigma, sigma / jnp.sqrt(1 - rho ** 2))
    if generate:
        innovation = numpyro.sample("generated_lap_innovation", StableStudentT(nu, 0., noise_scale))
        simulated = mean + linear_recurrence(jnp.where(r["prev"], rho, 0.), innovation)
        numpyro.deterministic("generated_laps", simulated)
    else:
        numpyro.factor("overall_laps", StableStudentT(nu, location, noise_scale).log_prob(residual_lap).sum())

    # Partial pooling keeps a rare circuit/condition from getting its own free fit.
    o = outcomes
    def context_coefficient(name, prior_location):
        base = numpyro.sample("overall_" + name + "_base", dist.Normal(prior_location, .7))
        circ = centre(scale(name + "_circuit", .3) * normal(name + "_circuit_z", o["n_circuits"]),
                      jnp.zeros(o["n_circuits"], dtype=int), 1)
        wet = numpyro.sample("overall_" + name + "_wet", dist.Normal(0, .3))
        sc = numpyro.sample("overall_" + name + "_sc", dist.Normal(0, .3))
        return base + circ[o["context_circuit"]] + wet * o["context_wet"] + sc * o["context_sc"]
    grid = numpyro.deterministic("overall_grid", jnp.exp(context_coefficient("grid", 0)))
    pace = numpyro.deterministic("overall_pace", jnp.exp(context_coefficient("pace", 0)))
    strength = pace[o["context"], None] * total[o["entry"]] - grid[o["context"], None] * o["log_grid"]
    import jax
    # Explicit empty suffixes and relative normalizers exclude padding without
    # losing probability or gradients at large common strength offsets.
    if generate:
        gumbel = numpyro.sample("generated_order_noise", dist.Gumbel(0., 1.).expand(strength.shape))
        order = perturbed_order(strength, gumbel, o["mask"])
        numpyro.deterministic("generated_order", order)
    else:
        numpyro.factor("overall_finish_order", plackett_luce(strength, o["mask"]).sum())
    hazard = numpyro.deterministic("overall_hazard", jax.nn.sigmoid(context_coefficient("hazard", -6)))
    # Observed completed laps are censored exposure for finishers. Retirement
    # contributes the next-lap failure. This is a common all-cause hazard, not a
    # validated driver-specific incident model or a mechanical-cause attribution.
    h = hazard[o["risk_context"]]
    if generate:
        failure = numpyro.sample("generated_failure_lap", dist.Geometric(probs=h))
        # Exposure windows are fixed design variables; status/completion are new
        # draws. They do not choose the independent conditional-order observation
        # panel used by this likelihood-level calibration experiment.
        numpyro.deterministic("generated_completed", jnp.minimum(failure, o["risk_distance"]))
        numpyro.deterministic("generated_failed", failure < o["risk_distance"])
    else:
        numpyro.factor("overall_completion", (o["completed"] * jnp.log1p(-h)
                        + o["failed"] * jnp.log(h)).sum())
