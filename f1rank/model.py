"""Dynamic Bayesian model separating driver skill from car-package pace.

    y[i] = mu[segment] + car[team, event] + skill[driver, event] + noise

Units are percent of lap time, positive = faster.

car    persistent development: random walk within a season with heavy-tailed
       steps (upgrades are lumpy), partial carryover rho between seasons, weaker
       carryover and larger innovations at regulation resets. On top of it, a
       one-event effect shared by both teammates that does not persist, and a
       car x circuit term (team-season loading x fixed circuit factor). The
       persistent part is the reported car rating.
skill  driver level + slow random walk (per race within a season, per season
       across gaps) + shared experience curve + shared decline after age 32.
noise  Student-t with a per-segment scale, so wet or chaotic sessions are
       down-weighted automatically.

Only differences between cars at the same event are identified, so car and
skill are centred on the field mean at each event before entering the
likelihood. The centred values are the reported ratings.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

CAR_STEP_DF = 4.0      # tail weight of within-season car steps
CAR_FIRST_SD = 1.5     # prior sd of a team's first car state in the window
COMPROMISED_DF = 4.0   # tail weight of the compromised-lap component (mixture likelihood)


def linear_recurrence(a, b):
    """x[j] = a[j] * x[j-1] + b[j] over a flat array; a[j] = 0 starts a new sequence."""
    def combine(left, right):
        a1, b1 = left
        a2, b2 = right
        return a1 * a2, a2 * b1 + b2
    return jax.lax.associative_scan(combine, (a, b))[1]


def centre(x, group, n_groups):
    total = jax.ops.segment_sum(x, group, n_groups)
    count = jax.ops.segment_sum(jnp.ones_like(x), group, n_groups)
    return x - (total / count)[group]


def model(d, car_track=True, car_step_df=CAR_STEP_DF, skill_drift=None, car_transient=True,
          likelihood="t", car_segment=True, driver_form=True, compat=True,
          placebo=False):
    """car_track: include car x circuit term. car_step_df: tail weight of car steps
    (large = Gaussian). skill_drift: None to estimate driver drift, or a fixed
    (per-race sd, per-season sd) pair for sensitivity runs. car_transient: add a
    one-event car effect (shared by both teammates) that does not persist.
    likelihood: "t" (symmetric Student-t), "split_t" (Student-t with a wider
    scale on the slow side: a lap can be far slower than potential, rarely
    faster) or "mixture" (clean laps ~ Normal, plus a slower, wider
    compromised-lap component). car_segment: add an effect shared by both
    teammates within one qualifying segment (same run window, track state and
    setup), which otherwise inflates the per-lap noise. driver_form: add a
    one-event driver effect (weekend form) that does not persist, so the skill
    walk only tracks lasting changes. compat: add a driver x team effect
    (car compatibility) that applies for as long as the driver stays in that
    team; the reported skill then excludes it (portable ability). placebo:
    diagnostic only - an extra effect for each half of a multi-season stint
    in one team, to test whether "compatibility" is team-specific."""
    # ---------------- drivers
    sd_level = numpyro.sample("sd_level", dist.HalfNormal(0.5))
    level = sd_level * numpyro.sample("level_z", dist.Normal(0, 1).expand([d["n_drivers"]]))
    if skill_drift is None:
        sd_race = numpyro.sample("sd_skill_race", dist.HalfNormal(0.03))
        sd_season = numpyro.sample("sd_skill_season", dist.HalfNormal(0.2))
    else:
        sd_race, sd_season = skill_drift
    z_skill = numpyro.sample("skill_z", dist.Normal(0, 1).expand([d["n_entries"]]))

    first = d["entry_first"]
    step = jnp.where(d["entry_same_season"],
                     sd_race * jnp.sqrt(d["entry_event_gap"]),
                     sd_season * jnp.sqrt(jnp.maximum(d["entry_season_gap"], 1.0)))
    b = jnp.where(first, level[d["entry_driver"]], step * z_skill)
    walk = linear_recurrence(jnp.where(first, 0.0, 1.0), b)

    exp_gain = numpyro.sample("exp_gain", dist.Normal(0, 1))
    exp_scale = numpyro.sample("exp_scale", dist.LogNormal(jnp.log(20.0), 0.7))
    age_slope = numpyro.sample("age_slope", dist.Normal(0, 0.2))
    trend = (exp_gain * (1 - jnp.exp(-d["entry_experience"] / exp_scale))
             + age_slope * d["entry_age_over"])
    skill = numpyro.deterministic(
        "skill", centre(walk + trend, d["entry_event"], d["n_events"]))

    # ---------------- cars
    rho = numpyro.sample("rho", dist.Beta(8, 2))
    rho_reset = numpyro.sample("rho_reset", dist.Beta(3, 3))
    sd_car_race = numpyro.sample("sd_car_race", dist.HalfNormal(0.15))
    sd_car_season = numpyro.sample("sd_car_season", dist.HalfNormal(0.5))
    sd_car_reset = numpyro.sample("sd_car_reset", dist.HalfNormal(1.0))

    cfirst, cstart, creset = d["car_first"], d["car_season_start"], d["car_reset"]
    carry = jnp.where(cstart, jnp.where(creset, rho_reset, rho), 1.0)
    jump_sd = jnp.where(cfirst, CAR_FIRST_SD, jnp.where(creset, sd_car_reset, sd_car_season))

    if car_transient:
        # Local-level model: a persistent walk (development) plus a one-event
        # effect shared by both teammates that does not carry over. The event
        # total is well informed by data, so it is sampled directly; the
        # persistent walk is weakly informed per event, so it is non-centred.
        z_t = numpyro.sample("car_step_z", dist.StudentT(car_step_df).expand([d["n_cars"]]))
        z_n = numpyro.sample("car_jump_z", dist.Normal(0, 1).expand([d["n_cars"]]))
        a = jnp.where(cfirst, 0.0, carry)
        b = jnp.where(cfirst | cstart, jump_sd * z_n, sd_car_race * z_t)
        persistent = linear_recurrence(a, b)
        sd_evt = numpyro.sample("sd_car_event", dist.HalfNormal(0.2))
        total = numpyro.sample("car_total", dist.ImproperUniform(
            dist.constraints.real, (), (d["n_cars"],)))
        numpyro.factor("car_event_prior", jnp.sum(dist.Normal(persistent, sd_evt).log_prob(total)))
        car = centre(persistent, d["car_event"], d["n_events"])
        numpyro.deterministic("car", car)
        total = centre(total, d["car_event"], d["n_events"])
        numpyro.deterministic("car_event", total - car)
        car = total
    else:
        # Car states are well informed by data (two drivers, several segments
        # per event), so they are sampled directly with the random walk as
        # their prior (centred parameterisation).
        state = numpyro.sample("car_state", dist.ImproperUniform(
            dist.constraints.real, (), (d["n_cars"],)))
        prev = jnp.concatenate([jnp.zeros(1), state[:-1]])
        innov = state - jnp.where(cfirst, 0.0, carry * prev)
        numpyro.factor("car_prior", jnp.sum(jnp.where(
            cfirst | cstart,
            dist.Normal(0, jump_sd).log_prob(innov),
            dist.StudentT(car_step_df, 0, sd_car_race).log_prob(innov))))
        car = centre(state, d["car_event"], d["n_events"])
        numpyro.deterministic("car", car)

    # car x circuit: team-season loading times a fixed circuit factor
    # (high-speed circuits positive; estimated beforehand, see design.circuit_factors)
    if car_track:
        sd_load = numpyro.sample("sd_track_load", dist.HalfNormal(0.5))
        load = sd_load * numpyro.sample(
            "track_load_z", dist.Normal(0, 1).expand([d["n_team_seasons"]]))
        numpyro.deterministic("track_load", load)
        fit_track = centre(load[d["car_team_season"]] * d["circuit_factor"][d["car_circuit"]],
                           d["car_event"], d["n_events"])
        numpyro.deterministic("car_track", fit_track)
        car = car + fit_track

    # ---------------- segments and noise
    mu = numpyro.sample("mu", dist.Normal(0, 2).expand([d["n_sessions"]]))
    sigma0 = numpyro.sample("sigma0", dist.HalfNormal(0.5))
    tau = numpyro.sample("sigma_tau", dist.HalfNormal(0.5))
    u = numpyro.sample("sigma_u", dist.Normal(0, 1).expand([d["n_sessions"]]))
    sigma = sigma0 * jnp.exp(tau * u)
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))

    loc = mu[d["obs_session"]] + car[d["obs_car"]] + skill[d["obs_entry"]]
    if compat:
        sd_compat = numpyro.sample("sd_compat", dist.HalfNormal(0.1))
        fit_ = sd_compat * numpyro.sample("compat_z", dist.Normal(0, 1).expand([d["n_stints"]]))
        fit_ = numpyro.deterministic(
            "compat", centre(fit_[d["entry_stint"]], d["entry_event"], d["n_events"]))
        loc = loc + fit_[d["obs_entry"]]
    if placebo:
        sd_placebo = numpyro.sample("sd_placebo", dist.HalfNormal(0.1))
        pz = sd_placebo * numpyro.sample("placebo_z", dist.Normal(0, 1).expand([d["n_placebo"]]))
        pe = jnp.where(d["entry_placebo"] >= 0, pz[jnp.maximum(d["entry_placebo"], 0)], 0.0)
        loc = loc + centre(pe, d["entry_event"], d["n_events"])[d["obs_entry"]]
    if driver_form:
        sd_form = numpyro.sample("sd_driver_form", dist.HalfNormal(0.2))
        form = sd_form * numpyro.sample("driver_form_z", dist.Normal(0, 1).expand([d["n_entries"]]))
        form = numpyro.deterministic("driver_form", centre(form, d["entry_event"], d["n_events"]))
        loc = loc + form[d["obs_entry"]]
    if car_segment:
        sd_seg = numpyro.sample("sd_car_segment", dist.HalfNormal(0.2))
        seg = sd_seg * numpyro.sample("car_segment_z",
                                      dist.Normal(0, 1).expand([d["n_car_segments"]]))
        seg = numpyro.deterministic("car_segment",
                                    centre(seg, d["car_segment_session"], d["n_sessions"]))
        loc = loc + seg[d["obs_car_segment"]]
    scale = sigma[d["obs_session"]]
    if likelihood == "split_t":
        skew = numpyro.sample("slow_scale_ratio", dist.LogNormal(0.0, 0.5))
        noise = SplitStudentT(nu, loc, scale, scale * skew)
    elif likelihood == "mixture":
        # clean laps: Normal(loc, scale); compromised laps (traffic, mistakes,
        # yellow flags): slower on average and wider, heavy-tailed
        p_bad = numpyro.sample("p_compromised", dist.Beta(2, 8))
        wide = 1.0 + numpyro.sample("compromised_extra_scale", dist.LogNormal(jnp.log(2.0), 0.5))
        shift = numpyro.sample("compromised_shift", dist.Normal(1.0, 1.0))
        noise = dist.MixtureGeneral(
            dist.Categorical(probs=jnp.array([1 - p_bad, p_bad])),
            [dist.Normal(loc, scale),
             dist.StudentT(COMPROMISED_DF, loc - shift * scale, wide * scale)])
    else:
        noise = dist.StudentT(nu, loc, scale)
    with numpyro.handlers.mask(mask=d["train"]):
        numpyro.sample("y", noise, obs=d["y"])


class SplitStudentT(dist.Distribution):
    """Student-t with scale `fast` above the mode and `slow` below it (two-piece)."""
    support = dist.constraints.real

    def __init__(self, df, loc, fast, slow, validate_args=None):
        self.df, self.loc, self.fast, self.slow = df, loc, fast, slow
        shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(fast), jnp.shape(slow))
        super().__init__(batch_shape=shape, validate_args=validate_args)

    def log_prob(self, value):
        r = value - self.loc
        side = jnp.where(r < 0, self.slow, self.fast)
        std = dist.StudentT(self.df).log_prob(r / side)
        return std + jnp.log(2.0) - jnp.log(self.fast + self.slow)

    def sample(self, key, sample_shape=()):
        k1, k2 = jax.random.split(key)
        shape = sample_shape + self.batch_shape
        t = jnp.abs(dist.StudentT(self.df).sample(k1, shape))
        slow_side = jax.random.uniform(k2, shape) < self.slow / (self.fast + self.slow)
        return self.loc + jnp.where(slow_side, -self.slow * t, self.fast * t)
