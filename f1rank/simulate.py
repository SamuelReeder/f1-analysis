"""Synthetic data on the real F1 design, for recovery and misspecification tests.

Truth is drawn from the model's full generative process (including the
one-event car, car-segment, driver-form and compatibility effects) with hyperparameters
set to the main fit's posterior medians, on the real entries, teams and
segments (so the teammate/transfer network is exactly F1's). Misspecification
scenarios then add effects the model does not contain; the recovery target is
always the clean skill and car truth.
"""

import numpy as np
import pandas as pd

from .design import Design, circuit_factors
from .model import CAR_FIRST_SD, CAR_STEP_DF

SCENARIOS = ["clean", "compat", "upgrades", "form", "transfer_luck"]


def _centre(x, group):
    return x - pd.Series(x).groupby(group).transform("mean").to_numpy()


def draw_truth(design: Design, hyper: dict, mu: np.ndarray, rng) -> dict:
    a = design.arrays()
    e, c = design.entries, design.cars

    level = rng.normal(0, hyper["sd_level"], a["n_drivers"])
    step = np.where(a["entry_same_season"], hyper["sd_skill_race"] * np.sqrt(a["entry_event_gap"]),
                    hyper["sd_skill_season"] * np.sqrt(np.maximum(a["entry_season_gap"], 1)))
    inc = np.where(a["entry_first"], level[a["entry_driver"]], step * rng.standard_normal(len(e)))
    walk = pd.Series(inc).groupby(e.driver_idx.to_numpy()).cumsum().to_numpy()
    trend = (hyper["exp_gain"] * (1 - np.exp(-a["entry_experience"] / hyper["exp_scale"]))
             + hyper["age_slope"] * a["entry_age_over"])
    skill = _centre(walk + trend, a["entry_event"])

    car = np.zeros(len(c))
    for j in range(len(c)):
        if a["car_first"][j]:
            car[j] = rng.normal(0, CAR_FIRST_SD)
        elif a["car_season_start"][j]:
            reset = a["car_reset"][j]
            rho = hyper["rho_reset"] if reset else hyper["rho"]
            sd = hyper["sd_car_reset"] if reset else hyper["sd_car_season"]
            car[j] = rho * car[j - 1] + rng.normal(0, sd)
        else:
            car[j] = car[j - 1] + hyper["sd_car_race"] * rng.standard_t(CAR_STEP_DF)
    car = _centre(car, a["car_event"])
    event = _centre(rng.normal(0, hyper["sd_car_event"], len(c)), a["car_event"])
    load = rng.normal(0, hyper["sd_track_load"], a["n_team_seasons"])
    track = _centre(load[a["car_team_season"]] * a["circuit_factor"][a["car_circuit"]], a["car_event"])

    form = _centre(rng.normal(0, hyper["sd_driver_form"], len(e)), a["entry_event"])
    compat = _centre(rng.normal(0, hyper["sd_compat"], a["n_stints"])[a["entry_stint"]],
                     a["entry_event"])
    segment = _centre(rng.normal(0, hyper["sd_car_segment"], a["n_car_segments"]),
                      a["car_segment_session"])
    sigma = hyper["sigma0"] * np.exp(hyper["sigma_tau"] * rng.standard_normal(a["n_sessions"]))
    return {"skill": skill, "car": car, "car_event": event, "car_track": track,
            "driver_form": form, "compat": compat, "car_segment": segment,
            "mu": mu, "sigma": sigma}


def violations(design: Design, scenario: str, rng) -> np.ndarray:
    """Extra pace per observation (percent) that the model cannot represent."""
    o = design.obs
    e = design.entries.set_index("entry_idx")
    oe = e.loc[o.entry_idx]
    extra = np.zeros(len(o))
    if scenario == "compat":
        # extra driver-car compatibility on top of the modelled amount (sd 0.10%)
        key = oe.driver_id.to_numpy() + "|" + oe.team.to_numpy()
        codes, inv = np.unique(key, return_inverse=True)
        extra = rng.normal(0, 0.10, len(codes))[inv]
    elif scenario == "upgrades":
        # unequal parts: in a quarter of team-seasons one driver gets new parts
        # six races before the teammate (+0.15%)
        ent = design.entries
        for (team, season), g in ent.groupby(["team", "season"]):
            if rng.random() > 0.25:
                continue
            driver = rng.choice(g.driver_id.unique())
            events = np.sort(g.event_idx.unique())
            start = rng.integers(0, max(1, len(events) - 6))
            window = set(events[start:start + 6])
            hit = (oe.team.to_numpy() == team) & (oe.season.to_numpy() == season) & \
                  (oe.driver_id.to_numpy() == driver) & np.isin(oe.event_idx.to_numpy(), list(window))
            extra[hit] += 0.15
    elif scenario == "form":
        # form that persists for several events (AR(1), stationary sd 0.1%), unlike
        # the model's one-event form term
        phi, sd = 0.85, 0.10
        form = np.zeros(len(design.entries))
        for _, idx in design.entries.groupby("driver_idx").indices.items():
            f = rng.normal(0, sd)
            for j in idx:
                f = phi * f + np.sqrt(1 - phi ** 2) * sd * rng.standard_normal()
                form[j] = f
        extra = form[o.entry_idx.to_numpy()]
    elif scenario == "transfer_luck":
        # endogenous mobility: drivers who change team had a lucky (+0.12%) season before
        ent = design.entries
        by = ent.groupby(["driver_id", "season"]).team.last().reset_index()
        by["next_team"] = by.groupby("driver_id").team.shift(-1)
        lucky = by[by.next_team.notna() & (by.next_team != by.team)]
        lucky_keys = set(zip(lucky.driver_id, lucky.season))
        hit = np.array([(d, s) in lucky_keys for d, s in zip(oe.driver_id, oe.season)])
        extra[hit] += 0.12
    elif scenario != "clean":
        raise ValueError(scenario)
    return extra


def synthetic_design(design: Design, truth: dict, scenario: str, nu: float, rng) -> Design:
    o = design.obs
    loc = (truth["mu"][o.session_idx] + truth["car"][o.car_idx] + truth["car_event"][o.car_idx]
           + truth["car_track"][o.car_idx] + truth["car_segment"][o.car_segment_idx]
           + truth["skill"][o.entry_idx] + truth["driver_form"][o.entry_idx]
           + truth["compat"][o.entry_idx])
    noise = truth["sigma"][o.session_idx] * rng.standard_t(nu, len(o))
    y = loc + noise + violations(design, scenario, rng)
    out = Design(**{k: getattr(design, k) for k in design.__dataclass_fields__})
    out.obs = o.assign(y=y)
    out.circuit_factor = circuit_factors(out)
    return out
