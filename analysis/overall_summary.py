"""Bounded paired summaries for the unfinished canonical Overall producer.

Identity consistency and numerical summaries are not scientific acceptance.
This module has no publication entry point and never permits publication.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from f1rank.overall.simulation import JointDraws, MAX_EXPECTED_POINTS_MCSE, POINTS, simulate

ROOT = Path(__file__).resolve().parents[1]
DEFINITION = ROOT / "docs/overall_summary.md"
CONTRACT = json.loads((ROOT / "contracts/overall.json").read_text())
PROBES = tuple(CONTRACT["required_explanations"])
MAX_BLOCK_DRAWS = 128
MAX_BLOCK_REPLICATES = 128
IDENTITY_KEYS = {"fit_id", "posterior_key", "scenario_key", "simulation_key"}
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SIMULATOR_SOURCES = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
    "f1rank/overall/simulation.py", "f1rank/overall/scoring.py", "f1rank/overall/order.py")}


def _source_fingerprints():
    contract_bytes = (ROOT / "contracts/overall.json").read_bytes()
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if implementation != IMPLEMENTATION_SHA256 or json.loads(contract_bytes) != CONTRACT:
        raise ValueError("Summary implementation or contract changed after import")
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value
           for name, value in SIMULATOR_SOURCES.items()):
        raise ValueError("Summary simulator dependency changed after import")
    return dict(definition_sha256=hashlib.sha256(DEFINITION.read_bytes()).hexdigest(),
        implementation_sha256=implementation, contract_sha256=hashlib.sha256(contract_bytes).hexdigest(),
        simulator_sources=SIMULATOR_SOURCES.copy())


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= np.iinfo(np.int64).max


def _numeric(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name} axes or numeric values")
    return result.astype(float, copy=False)


def _histogram_quantile(counts, probability, *, offset=1, scale=1):
    """NumPy's linear empirical quantile without storing season observations."""
    size = int(counts.sum())
    if size < 1:
        raise ValueError("Empty season-rank inventory")
    position = (size - 1) * probability
    low, high = int(np.floor(position)), int(np.ceil(position))
    cdf = np.cumsum(counts)
    a, b = np.searchsorted(cdf, [low, high], side="right")
    return float((a + offset + (b - a) * (position - low)) * scale)


def _precision_summary(means, variances, replicates, bound):
    count = len(means)
    conditional = np.sqrt(variances / replicates)
    passed = bool(np.max(conditional) <= MAX_EXPECTED_POINTS_MCSE
                  and bound <= MAX_EXPECTED_POINTS_MCSE)
    tail = (1 - CONTRACT["interval_probability"]) / 2
    interval = np.quantile(means, [tail, 1 - tail]) if passed else [None, None]
    return dict(mean=float(means.mean()), q05=None if interval[0] is None else float(interval[0]),
        q95=None if interval[1] is None else float(interval[1]),
        posterior_sd=float(np.sqrt(max(0., means.var(ddof=1) - (variances / replicates).mean()))),
        mcse=float(np.sqrt(variances.sum() / (count ** 2 * replicates))),
        max_conditional_mcse=float(conditional.max()), conditional_mcse_bound=float(bound),
        conditional_precision_passed=passed)


def paired_simulation_block(draws, scenarios, *, replicates, seed, chain, draw_start,
                            replicate_start, advancing):
    """Generate paired baseline/probe seasons for a fixed bounded block layout.

    The stream address includes the declared seed and block coordinates. The
    same address is reused ONLY within this block's explanatory interventions.
    A future registered controller must bind the layout and seed in its identity;
    changing layout is not a way to rerun a failed precision measurement.
    """
    if (not isinstance(draws, JointDraws) or not _integer(replicates, 2)
            or replicates > MAX_BLOCK_REPLICATES or len(draws.qualifying) > MAX_BLOCK_DRAWS
            or not all(_integer(value) for value in (seed, chain, draw_start, replicate_start))):
        raise ValueError("Need a bounded joint-draw block and complete nonnegative stream address")
    source = _source_fingerprints()
    address = [seed, chain, draw_start, replicate_start]
    baseline = simulate(draws, scenarios, replicates=replicates, seed=address, advancing=advancing)
    probes = np.empty(baseline["points"].shape + (len(PROBES),))
    field_order = np.argsort(draws.drivers)
    q_mean = draws.qualifying[:, field_order].mean(axis=-1)
    r_mean = draws.race[:, field_order].mean(axis=-1)
    for driver in range(len(draws.drivers)):
        for k, name in enumerate(PROBES):
            values = {}
            if name in ("qualifying_pace", "all"):
                values["qualifying"] = draws.qualifying.copy()
                values["qualifying"][:, driver] = q_mean
            if name in ("total_race_pace", "all"):
                values["race"] = draws.race.copy()
                values["race"][:, driver] = r_mean
            intervention = simulate(replace(draws, **values), scenarios, replicates=replicates,
                                    seed=address, advancing=advancing)
            probes[..., driver, k] = intervention["points"][..., driver]
    if source != _source_fingerprints():
        raise ValueError("Paired simulation source changed during generation")
    return baseline, probes


class OverallSummary:
    """Keep posterior moments, paired covariance and discrete rank histograms.

    Blocks contain common-randomness baseline/probe seasons. Their authenticity
    belongs to the future producer; this consumer checks identities and coverage.
    Memory is proportional to posterior draws and squared field size, independent
    of the total declared replicate count.
    """

    def __init__(self, *, drivers, posterior_shape, replicates, n_races, identity):
        if (not isinstance(drivers, (list, tuple)) or len(drivers) < 2
                or any(not isinstance(d, str) or not d.strip() for d in drivers)
                or len(set(drivers)) != len(drivers)):
            raise ValueError("A complete unique field is required")
        if (not isinstance(posterior_shape, (tuple, list)) or len(posterior_shape) != 2
                or not all(_integer(n, 2) for n in posterior_shape)
                or not _integer(replicates, 2) or not _integer(n_races, 1)):
            raise ValueError("Declare grouped posterior axes, replicates and race count")
        if (not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS
                or not isinstance(identity["fit_id"], str) or not identity["fit_id"].strip()
                or any(not isinstance(identity[k], str) or not re.fullmatch(r"[0-9a-f]{64}", identity[k])
                       for k in IDENTITY_KEYS - {"fit_id"})):
            raise ValueError("Complete fit, posterior, scenario and simulation identities are required")
        if (tuple(PROBES) != ("qualifying_pace", "total_race_pace", "all")
                or CONTRACT["max_conditional_points_mcse"] != MAX_EXPECTED_POINTS_MCSE
                or list(POINTS) != CONTRACT["points"]):
            raise ValueError("Summary contract and simulator disagree")
        self.drivers, self.shape = tuple(drivers), tuple(posterior_shape)
        self.replicates, self.n_races = replicates, n_races
        self.identity = copy.deepcopy(identity)
        self._sources = _source_fingerprints()
        self._counts = np.zeros(self.shape, dtype=np.int64)
        n = len(drivers)
        self._means = np.zeros(self.shape + (n,))
        self._covariance_sum = np.zeros(self.shape + (n, n))
        self._loss_means = np.zeros(self.shape + (n, len(PROBES)))
        self._loss_variance_sum = np.zeros_like(self._loss_means)
        self._title_sum = np.zeros(n)
        self._rank_low = np.zeros((n, n), dtype=np.int64)
        self._rank_high = np.zeros((n, n), dtype=np.int64)
        self._rank_mid = np.zeros((n, 2 * n + 1), dtype=np.int64)
        self._closed = False

    @property
    def storage_bytes(self):
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))

    def add(self, result, probe_points, *, chain, draw_start, replicate_start, identity):
        """Add a bounded draw × replicate block after validating it completely.

        probe_points has a final probe axis in PROBES order, retaining only each
        intervened driver's own outcome. Every value is a season points total.
        Invalid blocks leave the accumulator unchanged.
        """
        if self._closed:
            raise ValueError("Summary is already finalized")
        if (identity != self.identity or not isinstance(result, dict)
                or result.get("drivers") != list(self.drivers)
                or type(result.get("n_races")) is not int or result["n_races"] != self.n_races):
            raise ValueError("Simulation identities, field or reference calendar disagree")
        if (not _integer(chain) or chain >= self.shape[0] or not _integer(draw_start)
                or not _integer(replicate_start)):
            raise ValueError("Invalid posterior or replicate indices")
        shape = np.shape(result.get("points"))
        n = len(self.drivers)
        if (len(shape) != 3 or shape[2] != n
                or not 1 <= shape[0] <= MAX_BLOCK_DRAWS or not 1 <= shape[1] <= MAX_BLOCK_REPLICATES
                or draw_start + shape[0] > self.shape[1] or replicate_start + shape[1] > self.replicates):
            raise ValueError("Block exceeds declared posterior, replicate or memory bounds")
        index = (chain, slice(draw_start, draw_start + shape[0]))
        if not np.all(self._counts[index] == replicate_start):
            raise ValueError("Missing, duplicate or out-of-order replicate block")
        points = _numeric(result.get("points"), shape, "season points")
        probes = _numeric(probe_points, shape + (len(PROBES),), "paired probe points")
        bound = float(np.max(POINTS)) * self.n_races
        if (not np.isfinite(bound) or (points < 0).any() or (points > bound).any()
                or (probes < 0).any() or (probes > bound).any()):
            raise ValueError("Baseline and probe points exceed the common scoring bounds")
        field_bound = float(np.sum(POINTS[:n])) * self.n_races
        if (points.sum(axis=-1) > field_bound).any():
            raise ValueError("Baseline field points exceed the common scoring allocation")
        low = _numeric(result.get("rank_low"), shape, "lower season ranks")
        high = _numeric(result.get("rank_high"), shape, "upper season ranks")
        title = _numeric(result.get("title_share"), shape, "title shares")
        if ((low < 1).any() or (high > n).any() or (low > high).any()
                or (low != np.floor(low)).any() or (high != np.floor(high)).any()):
            raise ValueError("Invalid sporting rank intervals")
        for rank in range(1, n + 1):
            same = low == rank
            if (same & (high != rank + same.sum(axis=-1, keepdims=True) - 1)).any():
                raise ValueError("Tied sporting ranks must describe complete groups")
        order = np.argsort(low, axis=-1)
        positions = np.arange(1, n + 1)
        if ((np.take_along_axis(low, order, axis=-1) > positions).any()
                or (np.take_along_axis(high, order, axis=-1) < positions).any()):
            raise ValueError("Sporting rank groups overlap or omit positions")
        ordered_points = np.take_along_axis(points, order, axis=-1)
        ordered_low = np.take_along_axis(low, order, axis=-1)
        point_difference = np.diff(ordered_points, axis=-1)
        if ((point_difference > 0).any()
                or ((np.diff(ordered_low, axis=-1) == 0) & (point_difference != 0)).any()):
            raise ValueError("Sporting ranks contradict championship points")
        expected_title = np.where(low == 1, 1 / (high - low + 1), 0.)
        if not np.array_equal(title, expected_title):
            raise ValueError("Title mass must follow the complete sporting ties")

        # Prepare moments before mutating state, keeping rejected input atomic.
        points = points / self.n_races
        losses = points[..., None] - probes / self.n_races
        means, loss_means = points.mean(axis=1), losses.mean(axis=1)
        residual, loss_residual = points - means[:, None], losses - loss_means[:, None]
        covariance = np.einsum("brn,brm->bnm", residual, residual)
        loss_variance = (loss_residual ** 2).sum(axis=1)
        count = shape[1]
        total = replicate_start + count
        correction = replicate_start * count / total
        delta = means - self._means[index]
        loss_delta = loss_means - self._loss_means[index]
        self._covariance_sum[index] += covariance + delta[..., :, None] * delta[..., None, :] * correction
        self._loss_variance_sum[index] += loss_variance + loss_delta ** 2 * correction
        self._means[index] += delta * (count / total)
        self._loss_means[index] += loss_delta * (count / total)
        self._counts[index] = total
        self._title_sum += title.sum(axis=(0, 1))
        for driver in range(n):
            self._rank_low[driver] += np.bincount(low[..., driver].astype(int).ravel(), minlength=n + 1)[1:]
            self._rank_high[driver] += np.bincount(high[..., driver].astype(int).ravel(), minlength=n + 1)[1:]
            self._rank_mid[driver] += np.bincount((low[..., driver] + high[..., driver]).astype(int).ravel(), minlength=2 * n + 1)

    def finalize(self):
        if self._closed:
            raise ValueError("Summary is already finalized")
        if not np.all(self._counts == self.replicates):
            raise ValueError("Every declared posterior draw must have its complete replicate inventory")
        if self._sources != _source_fingerprints():
            raise ValueError("Summary definition or source changed during accumulation")
        n, count = len(self.drivers), int(np.prod(self.shape))
        means = self._means.reshape(count, n)
        covariance = self._covariance_sum.reshape(count, n, n) / (self.replicates - 1)
        variance = np.diagonal(covariance, axis1=-2, axis2=-1)
        losses = self._loss_means.reshape(count, n, len(PROBES))
        loss_variance = self._loss_variance_sum.reshape(count, n, len(PROBES)) / (self.replicates - 1)
        maximum = float(np.max(POINTS))
        points_bound = maximum / (2 * np.sqrt(self.replicates))
        difference_bound = maximum / np.sqrt(self.replicates)
        tail = (1 - CONTRACT["interval_probability"]) / 2
        rows, explanations, pairs = [], [], []
        for i, driver in enumerate(self.drivers):
            rows.append(dict(driver_id=driver,
                expected_points=_precision_summary(means[:, i], variance[:, i], self.replicates, points_bound),
                p_title=float(self._title_sum[i] / (count * self.replicates)),
                season_rank=dict(q05=_histogram_quantile(self._rank_low[i], tail),
                    median=_histogram_quantile(self._rank_mid[i], .5, offset=0, scale=.5),
                    q95=_histogram_quantile(self._rank_high[i], 1 - tail))))
            explanations.append(dict(driver_id=driver, **self.identity,
                losses={name: _precision_summary(losses[:, i, k], loss_variance[:, i, k],
                    self.replicates, difference_bound) for k, name in enumerate(PROBES)}))
        indices = sorted(range(n), key=lambda i: self.drivers[i])
        for position, i in enumerate(indices):
            for j in indices[position + 1:]:
                difference = means[:, i] - means[:, j]
                paired_variance = variance[:, i] + variance[:, j] - 2 * covariance[:, i, j]
                tolerance = 32 * np.finfo(float).eps * np.maximum(1., variance[:, i] + variance[:, j])
                if (paired_variance < -tolerance).any():
                    raise ValueError("Accumulated paired variance is inconsistent")
                summary = _precision_summary(difference, np.maximum(0, paired_variance), self.replicates, difference_bound)
                pairs.append(dict(driver_a=self.drivers[i], driver_b=self.drivers[j], points_difference=summary,
                    finite_mean_probability_greater=float((difference > 0).mean()),
                    finite_mean_probability_equal=float((difference == 0).mean()),
                    rank_precision_established=False))
        rows.sort(key=lambda row: (-row["expected_points"]["mean"], row["driver_id"]))
        explanations.sort(key=lambda row: row["driver_id"])
        if self._sources != _source_fingerprints():
            raise ValueError("Summary definition or source changed during accumulation")
        self._closed = True
        return dict(purpose="development_canonical_overall_summary", model_version=CONTRACT["model_version"],
            target=CONTRACT["target"], identity=copy.deepcopy(self.identity), field=list(self.drivers),
            posterior_shape=list(self.shape), replicates_per_draw=self.replicates, n_races=self.n_races,
            complete_draw_and_replicate_inventory=True, standings=rows, contributions=explanations,
            pairwise=pairs, season_tie_policy="full_sporting_intervals_with_midpoint_median_and_shared_title_mass",
            expected_points_precision_passed=all(row["expected_points"]["conditional_precision_passed"] for row in rows),
            explanation_precision_passed=all(loss["conditional_precision_passed"] for row in explanations for loss in row["losses"].values()),
            pairwise_points_precision_passed=all(row["points_difference"]["conditional_precision_passed"] for row in pairs),
            **self._sources,
            conditional_simulation_error_only=True, mcmc_precision_verified=False,
            rank_precision_established=False, scientific_evidence_verified=False, publication_allowed=False)
