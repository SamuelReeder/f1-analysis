"""Production copy of the verified overall_control_rotation.py.

The original research implementation and its evidence remain unchanged.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


def control_design(laps, race):
    """All fixed-scale Gaussian lap controls in the committed joint likelihood.

    Prior-whiten the columns. Zero columns remain in the likelihood model, but
    have no design information; the report distinguishes them from linear
    dependence among the active columns. No column is dropped from any fit.
    """
    mask = np.asarray(laps["race"]) == race
    columns, names, scales = [], [], []

    def append(name, value, scale):
        names.append(name)
        columns.append(np.asarray(value, dtype=float)[mask])
        scales.append(scale)

    append(f"overall_intercept[{race}]", np.ones(len(mask)), 3.)
    append(f"overall_trend[{race}]", laps["lap"], 2.)
    compounds = sorted(set(np.asarray(laps["rc"])[mask].tolist()))
    for compound in compounds:
        selected = np.asarray(laps["rc"]) == compound
        append(f"overall_compound[{compound}]", selected & np.asarray(laps["nonref"]), 3.)
        append(f"overall_tyre_slope[{compound}]", selected * np.asarray(laps["age"]), .3)
    for i in range(3):
        append(f"overall_traffic[{race},{i}]", np.asarray(laps["traffic"])[:, i], 1.)
    return np.column_stack(columns), names, np.asarray(scales), mask



ROTATED_SITE = "overall_lap_controls_rotated"


@dataclass(frozen=True)
class ControlRotation:
    blocks: tuple
    shapes: dict
    size: int

    @property
    def log_abs_det(self):
        return float(sum(np.log(block["scales"]).sum() for block in self.blocks))

    def original(self, rotated):
        """Map one latent coordinate vector back to the original model sites."""
        import jax.numpy as jnp
        if rotated.shape != (self.size,):
            raise ValueError("Rotated control vector has incorrect dimensions")
        values = {name: jnp.zeros(shape) for name, shape in self.shapes.items()}
        for block in self.blocks:
            z = rotated[block["start"]:block["stop"]]
            coefficients = (jnp.asarray(block["basis"]) @ z) * jnp.asarray(block["scales"])
            for i, (name, coordinate) in enumerate(block["coordinates"]):
                values[name] = values[name].at[coordinate].set(coefficients[i])
        return values

    def rotated(self, original):
        """Inverse for density/equivalence checks; this does not sample or fit."""
        if set(original) != set(self.shapes):
            raise ValueError("Original control sites do not match the coordinate map")
        if any(np.shape(original[name]) != shape for name, shape in self.shapes.items()):
            raise ValueError("Original control dimensions do not match the coordinate map")
        return np.concatenate([block["basis"].T @ (
            np.array([np.asarray(original[name])[coord] for name, coord in block["coordinates"]])
            / block["scales"]) for block in self.blocks])

    def fingerprint(self):
        h = hashlib.sha256()
        h.update(json.dumps(self.shapes, sort_keys=True).encode())
        for block in self.blocks:
            h.update(json.dumps(block["coordinates"]).encode())
            for key in ("basis", "scales"):
                a = block[key]
                h.update(f"{key}:{a.dtype.str}:{a.shape}".encode())
                h.update(a.tobytes())
        return h.hexdigest()


def build_rotation(laps):
    """Orthogonal eigenbasis of each prior-whitened design's Gram matrix.

    Keep every direction, including unused prior-only and collinear directions.
    No rank cutoff, fitted scale, response, or parameter-selection rule enters.
    This is a square rotation, so deficient designs do not truncate the prior.
    """
    nr, nc = int(laps["n_race"]), int(laps["n_rc"])
    shapes = {"overall_intercept": (nr,), "overall_trend": (nr,),
              "overall_compound": (nc,), "overall_tyre_slope": (nc,),
              "overall_traffic": (nr, 3)}
    blocks, seen, offset = [], set(), 0
    for race in range(nr):
        matrix, names, scales, _ = control_design(laps, race)
        if len(matrix) == 0 or not np.isfinite(matrix).all():
            raise ValueError("Control design must have finite observations in every race")
        coordinates = []
        for label in names:
            site, index = label.rstrip("]").split("[")
            key = site, tuple(int(x) for x in index.split(","))
            if key in seen:
                raise ValueError("A control is assigned to more than one race")
            seen.add(key)
            coordinates.append(key)
        whitened = matrix * scales
        _, basis = np.linalg.eigh(whitened.T @ whitened)
        # Fix eigenvector signs for repeatability; preserve degenerate directions.
        anchor = np.argmax(np.abs(basis), axis=0)
        basis *= np.where(basis[anchor, np.arange(len(basis))] < 0, -1, 1)
        np.testing.assert_allclose(basis.T @ basis, np.eye(len(basis)), rtol=0, atol=1e-12)
        blocks.append(dict(coordinates=coordinates, basis=basis, scales=scales,
                           start=offset, stop=offset + len(names)))
        offset += len(names)
    expected = {(site, index) for site, shape in shapes.items() for index in np.ndindex(shape)}
    if seen != expected:
        raise ValueError("Coordinate map must include every original control exactly once")
    return ControlRotation(tuple(blocks), shapes, offset)


def rotated_model(rotation):
    """Density-preserving wrapper selected by a registered production runner.

    A standard Normal is rotation invariant. Original fixed Normal scales are
    restored after rotating, then NumPyro exposes the original coefficients as
    deterministic sites. No second copy of their prior enters the density.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro import handlers
    from numpyro.infer.reparam import Reparam
    from .model import model

    class SuppliedControl(Reparam):
        def __init__(self, value):
            self.value = value

        def __call__(self, name, fn, obs):
            base, _, _ = self._unwrap(fn)
            if obs is not None or not isinstance(base, dist.Normal) or fn.shape() != self.value.shape:
                raise ValueError(f"Unexpected control prior at {name}")
            return None, self.value

    def wrapped(q, laps, outcomes, *, generate=False):
        z = numpyro.sample(ROTATED_SITE, dist.Normal(0., 1.).expand([rotation.size]).to_event(1))
        controls = rotation.original(jnp.asarray(z))
        with handlers.reparam(config={name: SuppliedControl(value) for name, value in controls.items()}):
            return model(q, laps, outcomes, generate=generate)

    return wrapped
