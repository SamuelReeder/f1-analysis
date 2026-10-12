"""Development race-stage baselines; no fitter, cutoff selection or publication.

The primary arm preserves the legacy fixed qualifying features and coefficient
priors. Its posterior must come from a separately checked qualifying-only fit.
These arithmetic interfaces cannot certify source vintage, convergence or a
common training inventory. The evaluation controller must establish those facts.
"""
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from analysis.overall_observation_kernels import order_log_probability
from f1rank.overall.order import plackett_luce


COEFFICIENTS = {
    "legacy_grid_qualifying": ("c_grid", "a_car", "b_driver"),
    "grid_only": ("c_grid",),
}
# Inherited from f1rank.championship.race_model, not chosen from comparison scores.
COEFFICIENT_SCALE = 5.


def _arm(arm):
    if arm not in COEFFICIENTS:
        raise ValueError("Unsupported legacy baseline; current_race_only uses its separate model")
    return COEFFICIENTS[arm]


def _finite(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and numeric")
    return array


def legacy_feature_medians(posterior):
    """Match qualifying.features: median of paired sums, not sums of medians.

    Arrays have chain, draw, entry/car axes. This is deliberately the legacy
    plug-in feature policy; only race-mapping coefficient uncertainty is then
    integrated. Passing joint race-updated states would change the baseline.
    The qualifying fit must use the legacy lineage compatibility definition.
    """
    names = ("skill", "compat", "car", "car_track")
    if any(name not in posterior for name in names):
        raise ValueError("Missing legacy qualifying feature sites")
    values = {name: _finite(posterior[name], name) for name in names}
    shape = values["skill"].shape
    if (len(shape) != 3 or min(shape) < 1 or min(shape[:2]) < 2
            or any(a.ndim != 3 or a.shape[:2] != shape[:2] or a.shape[-1] < 1
                   for a in values.values())
            or values["skill"].shape != values["compat"].shape
            or values["car"].shape != values["car_track"].shape):
        raise ValueError("Legacy features require paired chain/draw and entry/car axes")
    result = {}
    for name, left, right in (("driver", "skill", "compat"), ("car", "car", "car_track")):
        with np.errstate(over="ignore", invalid="ignore"):
            combined = values[left] + values[right]
        _finite(combined, name + " paired sum")
        result[name] = np.median(combined, axis=(0, 1))
    return result


def ordered_arrays(events, *, arm):
    """Build best-first order rows without removing a required event or entrant.

    Each event supplies event_id, drivers, grid_slot and running_position, plus
    fixed car/driver features for the primary arm. Position zero denotes outside
    the declared running panel. Grid slots must already use the complete-field
    pit-lane policy; zero is rejected rather than repaired after panel selection.
    Events and field identities remain in caller order in the returned metadata.
    No implicit finish-status rule, origin policy or historical exclusion applies.
    """
    _arm(arm)
    if not isinstance(events, (list, tuple)) or not events:
        raise ValueError("A nonempty complete event inventory is required")
    checked, identities = [], set()
    for event in events:
        identity = event.get("event_id")
        drivers = event.get("drivers")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Missing or duplicate event identity")
        identities.add(identity)
        if (not isinstance(drivers, (list, tuple)) or not drivers
                or not all(isinstance(d, str) and d for d in drivers)
                or len(set(drivers)) != len(drivers)):
            raise ValueError("Missing or duplicate driver identity")
        n = len(drivers)
        grid = _finite(event.get("grid_slot"), "grid slots")
        rank = _finite(event.get("running_position"), "running positions")
        if (grid.shape != (n,) or (grid < 1).any() or (grid > n).any()
                or (grid != np.floor(grid)).any()):
            raise ValueError("Grid slots must refer to the complete declared field")
        if (rank.shape != (n,) or (rank < 0).any() or (rank != np.floor(rank)).any()
                or not np.array_equal(np.sort(rank[rank > 0]), np.arange(1, (rank > 0).sum() + 1))):
            raise ValueError("Running positions must be unique contiguous positive ranks or zero")
        features = {}
        for name in ("car", "driver"):
            value = _finite(event.get(name), name) if arm == "legacy_grid_qualifying" else np.zeros(n)
            if value.shape != (n,):
                raise ValueError("Qualifying features must cover the full declared field")
            features[name] = value
        order = np.argsort(np.where(rank > 0, rank, np.inf), kind="stable")
        checked.append((event, grid, rank, features, order))
    width = max(len(e[0]["drivers"]) for e in checked)
    shape = (len(events), width)
    result = {name: np.zeros(shape) for name in ("car", "driver", "log_grid")}
    result["mask"] = np.zeros(shape, bool)
    fields, ordered_fields = [], []
    for row, (event, grid, rank, features, order) in enumerate(checked):
        n = len(order)
        result["mask"][row, :n] = rank[order] > 0
        result["log_grid"][row, :n] = np.log(grid[order])
        for name in features:
            result[name][row, :n] = features[name][order]
        fields.append(list(event["drivers"]))
        ordered_fields.append([event["drivers"][i] for i in order])
    return dict(arrays=result, event_ids=[e["event_id"] for e in events],
                fields=fields, ordered_fields=ordered_fields, arm=arm)


def race_model(arrays, *, arm):
    """Legacy statistical model, evaluated with the corrected PL arithmetic."""
    names = _arm(arm)
    c_grid = numpyro.sample("c_grid", dist.Normal(0., COEFFICIENT_SCALE))
    strength = -c_grid * jnp.asarray(arrays["log_grid"])
    if "a_car" in names:
        a_car = numpyro.sample("a_car", dist.Normal(0., COEFFICIENT_SCALE))
        b_driver = numpyro.sample("b_driver", dist.Normal(0., COEFFICIENT_SCALE))
        strength = strength + a_car * jnp.asarray(arrays["car"]) + b_driver * jnp.asarray(arrays["driver"])
    ll = plackett_luce(strength, jnp.asarray(arrays["mask"]))
    numpyro.factor("ll", ll.sum())
    numpyro.deterministic("ll_race", ll)


def log_probabilities(posterior, panel):
    """Conditional order log masses with chain, draw, event axes retained.

    Integrate probabilities with overall_predictive_scores.log_predictive after
    attaching verified posterior/draw/event identities. The baseline is directly
    conditional on the supplied running panel, with no completion model. It
    supplies no retirement, full-classification or pre-qualifying forecast.
    """
    names = _arm(panel["arm"])
    if any(name not in posterior for name in names):
        raise ValueError("Missing baseline coefficient draws")
    values = {name: _finite(posterior[name], name) for name in names}
    shape = values["c_grid"].shape
    if len(shape) != 2 or min(shape) < 2 or any(a.shape != shape for a in values.values()):
        raise ValueError("Baseline coefficients must retain paired chain/draw axes")
    arrays = panel["arrays"]
    result = []
    for row in range(len(panel["event_ids"])):
        with np.errstate(over="ignore", invalid="ignore"):
            strength = -values["c_grid"][..., None] * arrays["log_grid"][row]
            if "a_car" in names:
                strength = (strength + values["a_car"][..., None] * arrays["car"][row]
                            + values["b_driver"][..., None] * arrays["driver"][row])
        mask = arrays["mask"][row]
        position = np.where(mask, np.cumsum(mask), 0)
        result.append(order_log_probability(strength, position))
    return np.stack(result, axis=-1)
