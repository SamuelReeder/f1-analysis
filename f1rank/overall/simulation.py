"""Equal-opportunity simulation from one joint posterior, not marginal exports.

Scoring follows the full-distance GP scenario in the candidate contract. The
completion/line-order model is an approximation, not a physical lap simulator.
"""
from dataclasses import dataclass

import numpy as np

from .scoring import POINTS, gp_points_table, score_grand_prix
from .order import numpy_perturbed_order

MAX_EXPECTED_POINTS_MCSE = .05
QUALIFYING_FIELDS = {20: (15, 10), 22: (16, 10), 24: (17, 10)}
POSTERIOR_SITES = dict(qualifying="skill", race="overall_race_skill", q_weekend_sd="sd_driver_form",
    r_weekend_sd="overall_sd_day", shared_weekend="overall_shared_weekend",
    q_scale="sigma0", q_log_scale_sd="sigma_tau", q_df="nu", grid_coefficient="overall_grid",
    pace_coefficient="overall_pace", retirement_hazard="overall_hazard")


def _finite(name, value):
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class JointDraws:
    """Draw axis is shared by every state and coefficient; no resampling here.

    Construct with from_posterior, using a single checked MCMC result. Current
    driver states exclude fitted team, car and transient weekend effects.
    """
    drivers: tuple[str, ...]
    qualifying: np.ndarray
    race: np.ndarray
    q_weekend_sd: np.ndarray
    r_weekend_sd: np.ndarray
    shared_weekend: np.ndarray
    q_scale: np.ndarray
    q_log_scale_sd: np.ndarray
    q_df: np.ndarray
    grid_coefficient: np.ndarray
    pace_coefficient: np.ndarray
    retirement_hazard: np.ndarray

    def __post_init__(self):
        if len(set(self.drivers)) != len(self.drivers) or len(self.drivers) < 2:
            raise ValueError("Need unique driver identities")
        if self.qualifying.ndim != 2 or self.qualifying.shape[1] != len(self.drivers):
            raise ValueError("Qualifying must have shape (draw, driver)")
        n, _ = self.qualifying.shape
        if n < 2 or self.race.shape != self.qualifying.shape:
            raise ValueError("Race and qualifying must share draw and driver axes")
        for name in self.__dataclass_fields__:
            if name == "drivers":
                continue
            value = getattr(self, name)
            _finite(name, value)
            if value.shape[0] != n:
                raise ValueError(f"{name} has a different draw axis")
        for name in ("q_weekend_sd", "r_weekend_sd", "q_scale", "q_log_scale_sd"):
            value = getattr(self, name)
            if value.shape != (n,) or (value < 0).any():
                raise ValueError(f"Invalid {name}")
        for name in ("shared_weekend", "q_df"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"Invalid {name}")
        if (self.q_df <= 0).any():
            raise ValueError("Student-t degrees of freedom must be positive")
        shape = self.grid_coefficient.shape
        if len(shape) != 2 or shape[1] == 0:
            raise ValueError("Coefficients need a context axis")
        for name in ("grid_coefficient", "pace_coefficient", "retirement_hazard"):
            value = getattr(self, name)
            if value.shape != shape or (value < 0).any():
                raise ValueError(f"Invalid {name}")
        if (self.retirement_hazard >= 1).any():
            raise ValueError("Per-lap retirement hazard must be below one")

    @classmethod
    def from_posterior(cls, post, entries, drivers, *, checked):
        """Flatten chain/draw ONCE for the entire joint result.

        Diagnostics are supplied by the new model's checker, never legacy gates.
        No means, independent fit files or missing-driver zero fills are accepted.
        """
        if checked.get("converged") is not True:
            raise ValueError("Joint posterior has not passed its convergence checks")
        mapping = dict(qualifying="skill", race="overall_race_skill",
                       q_weekend_sd="sd_driver_form", r_weekend_sd="overall_sd_day",
                       shared_weekend="overall_shared_weekend", q_scale="sigma0",
                       q_log_scale_sd="sigma_tau", q_df="nu",
                       grid_coefficient="overall_grid", pace_coefficient="overall_pace",
                       retirement_hazard="overall_hazard")
        shape = np.asarray(post["skill"]).shape[:2]
        if len(shape) != 2 or shape[0] < 2:
            raise ValueError("Grouped chains and draws are required")
        index = np.asarray(entries)
        if (index.ndim != 1 or len(index) != len(drivers) or not np.issubdtype(index.dtype, np.integer)
                or len(np.unique(index)) != len(index) or (index < 0).any()
                or (index >= np.asarray(post["skill"]).shape[-1]).any()):
            raise ValueError("Field must map to unique valid driver-event states")
        values = {}
        for dest, src in mapping.items():
            a = np.asarray(post[src])
            if a.shape[:2] != shape:
                raise ValueError(f"{src} has mismatched chains/draws")
            a = a.reshape((-1,) + a.shape[2:])
            if dest in ("qualifying", "race"):
                a = a[:, index]
            values[dest] = a
        return cls(tuple(drivers), **values)


@dataclass(frozen=True)
class Scenario:
    """One GP, with explicit race-control assumptions for scoring.

    consecutive_racing_laps is a scenario input until a calibrated model generates
    the race-control sequence. A neutralisation indicator alone cannot supply it.
    """
    event_id: str
    context: int
    laps: int
    consecutive_racing_laps: int


def qualifying_grid(ability, weekend, scale, log_scale_sd, df, rng, advancing):
    """Fresh Student-t segment measurements; elimination keeps earlier positions.

    Segment intercept and equal car/segment effects cancel in all comparisons.
    Session scale is shared across drivers and newly drawn for each segment.
    """
    n = ability.shape[-1]
    if any(a <= 0 or a >= b for a, b in zip(advancing, (n, *advancing))):
        raise ValueError("Qualifying fields must shrink at each elimination")
    active = np.broadcast_to(np.arange(n), ability.shape).copy()
    order = np.empty_like(active)
    for keep in (*advancing, 0):
        session_scale = scale * np.exp(log_scale_sd * rng.normal(size=len(ability)))
        noise = rng.standard_t(df[:, None], size=ability.shape) * session_scale[:, None]
        pace = ability + weekend + noise
        active_pace = np.take_along_axis(pace, active, axis=-1)
        ranked = np.take_along_axis(active, np.argsort(-active_pace, axis=-1), axis=-1)
        order[:, keep:active.shape[-1]] = ranked[:, keep:]
        active = ranked[:, :keep]
    return np.argsort(order, axis=-1) + 1


def championship_countback(points, finishes, qualifying):
    """Points, race countback, then qualifying countback (current FIA rule).

    Inputs: points (..., driver), results (..., race, driver), 0 = unclassified.
    If every criterion is tied, expose a rank interval and split title mass.
    No driver identity or array position breaks a sporting tie.
    """
    points, finishes, qualifying = map(np.asarray, (points, finishes, qualifying))
    n = points.shape[-1]
    if finishes.shape != qualifying.shape or finishes.shape[:-2] != points.shape[:-1] or finishes.shape[-1] != n:
        raise ValueError("Championship axes do not match")
    _finite("points", points)
    for result in (finishes, qualifying):
        if not np.isfinite(result).all() or ((result < 0) | (result > n) | (result != np.floor(result))).any():
            raise ValueError("Invalid championship positions")
    criteria = [points]
    for result in (finishes, qualifying):
        criteria.extend((result == p).sum(axis=-2) for p in range(1, n + 1))
    tied = np.ones(points.shape + (n,), dtype=bool)
    better = np.zeros_like(tied)
    for criterion in criteria:
        left, right = criterion[..., :, None], criterion[..., None, :]
        better |= tied & (left > right)
        tied &= left == right
    # Row i counts competitors that it beats; column i counts those beating it.
    low = 1 + better.sum(axis=-2)
    high = low + tied.sum(axis=-1) - 1
    title = np.where(low == 1, 1 / tied.sum(axis=-1), 0)
    return dict(rank_low=low, rank_high=high, title_share=title)


def simulate(draws, scenarios, *, replicates, seed=0, advancing=None):
    """Nested replicates preserve posterior dependence and separate uncertainty.

    Ability is held at its current state across the reference calendar. All drivers
    share context, machinery, operations distribution and retirement hazard.
    Randomness attaches to sorted identities, making input permutation harmless.
    """
    if replicates < 2 or not scenarios:
        raise ValueError("Need repeated seasons and a nonempty reference calendar")
    if len({s.event_id for s in scenarios}) != len(scenarios):
        raise ValueError("Reference event identities must be unique")
    for s in scenarios:
        if s.context < 0 or s.context >= draws.grid_coefficient.shape[1] or s.laps < 2 or int(s.laps) != s.laps:
            raise ValueError("Invalid reference scenario")
        if not np.isscalar(s.consecutive_racing_laps):
            raise ValueError("Reference race-control assumption must be scalar")
        gp_points_table(scheduled_laps=s.laps, leader_laps=s.laps,
                        consecutive_racing_laps=s.consecutive_racing_laps)
    ix = np.argsort(draws.drivers)
    reverse = np.argsort(ix)
    q, r = draws.qualifying[:, ix], draws.race[:, ix]
    nd, n = q.shape
    if advancing is None:
        if n not in QUALIFYING_FIELDS:
            raise ValueError("An explicit qualifying format is required for this field")
        advancing = QUALIFYING_FIELDS[n]
    rng = np.random.default_rng(seed)
    season_points = np.zeros((nd, replicates, n))
    titles = np.zeros_like(season_points)
    rank_low, rank_high = np.empty_like(season_points), np.empty_like(season_points)
    for rep in range(replicates):
        finishes, grids = [], []
        for s in scenarios:
            qday = rng.normal(size=q.shape) * draws.q_weekend_sd[:, None]
            rday = (draws.shared_weekend[:, None] * qday
                    + rng.normal(size=q.shape) * draws.r_weekend_sd[:, None])
            grid = qualifying_grid(q, qday, draws.q_scale, draws.q_log_scale_sd,
                                   draws.q_df, rng, advancing)
            strength = (draws.pace_coefficient[:, s.context, None] * (r + rday)
                        - draws.grid_coefficient[:, s.context, None] * np.log(grid))
            line = np.argsort(numpy_perturbed_order(strength, rng.gumbel(size=q.shape)), axis=-1) + 1
            # Number of completed laps before a first failure; zero hazard is allowed.
            h = draws.retirement_hazard[:, s.context, None]
            uniform = rng.uniform(size=q.shape)
            failed_after = np.floor(np.divide(np.log1p(-uniform), np.log1p(-h),
                                    out=np.full(q.shape, np.inf), where=h > 0))
            laps = np.minimum(failed_after, s.laps).astype(int)
            # Scenario conditions on a completed GP. All-retired outcomes must not
            # silently award full points as if the scheduled distance was covered.
            if (laps.max(axis=-1) < s.laps).any():
                raise ValueError("No full-distance finisher: shortened-race model required")
            finish, points = score_grand_prix(
                laps, line, scheduled_laps=s.laps, leader_laps=s.laps,
                consecutive_racing_laps=s.consecutive_racing_laps)
            season_points[:, rep] += points
            finishes.append(finish)
            grids.append(grid)
        counted = championship_countback(season_points[:, rep], np.stack(finishes, axis=1), np.stack(grids, axis=1))
        titles[:, rep] = counted["title_share"]
        rank_low[:, rep], rank_high[:, rep] = counted["rank_low"], counted["rank_high"]
    return {"drivers": list(draws.drivers), "points": season_points[:, :, reverse],
            "title_share": titles[:, :, reverse], "rank_low": rank_low[:, :, reverse],
            "rank_high": rank_high[:, :, reverse], "n_races": len(scenarios)}


def uncertainty(result):
    """Keep within-draw season variability separate from between-draw ability.

    MCSE here covers independent simulation replicates conditional on the saved
    posterior, not MCMC integration error. Chain ESS is checked separately.
    Posterior intervals of replicate means are emitted only at declared precision.
    """
    points = np.asarray(result["points"], float) / result["n_races"]
    if points.ndim != 3 or min(points.shape[:2]) < 2:
        raise ValueError("Need draw, replicate, driver axes")
    _finite("simulation points", points)
    means = points.mean(axis=1)
    variance = points.var(axis=1, ddof=1)
    mcse = np.sqrt(variance / points.shape[1])
    precise = (mcse.max(axis=0) <= MAX_EXPECTED_POINTS_MCSE)
    posterior_sd = np.sqrt(np.maximum(0, means.var(axis=0, ddof=1) - (variance / points.shape[1]).mean(axis=0)))
    interval = np.quantile(means, [.05, .95], axis=0).T
    return dict(expected_points=means.mean(axis=0), posterior_sd=posterior_sd,
                expected_points_interval=np.where(precise[:, None], interval, np.nan),
                season_points_interval=np.quantile(points * result["n_races"], [.05, .95], axis=(0, 1)).T,
                expected_points_mcse=np.sqrt(variance.sum(axis=0) / (points.shape[0] ** 2 * points.shape[1])),
                max_conditional_mcse=mcse.max(axis=0), precision_passed=precise)
