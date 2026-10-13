"""Prepare a batched evaluation of the unchanged, complete control rotation.

This is not used by the running frozen profile. Full equivalence and resource
verification must precede its integration in a subsequent registered runner.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class _Group:
    basis: np.ndarray
    scales: np.ndarray
    latent_indices: np.ndarray


@dataclass(frozen=True)
class BatchedControlRotation:
    size: int
    shapes: object
    groups: tuple
    gathers: object
    inverse_order: tuple
    source_fingerprint: str
    log_abs_det: float

    def fingerprint(self):
        return self.source_fingerprint

    def original(self, rotated):
        """Same small matrix products and complete output assignments, batched."""
        import jax.numpy as jnp
        if rotated.shape != (self.size,):
            raise ValueError('Rotated control vector has incorrect dimensions')
        pieces = []
        for group in self.groups:
            z = rotated[jnp.asarray(group.latent_indices)]
            coefficients = jnp.einsum('bij,bj->bi', jnp.asarray(group.basis), z)
            pieces.append((coefficients * jnp.asarray(group.scales)).reshape(-1))
        joined = jnp.concatenate(pieces)
        # The original implementation assigns into default-dtype jnp.zeros.
        dtype = jnp.zeros(()).dtype
        return {name: joined[jnp.asarray(self.gathers[name])].astype(dtype).reshape(shape)
                for name, shape in self.shapes.items()}

    def rotated(self, original):
        """Keep the original per-block NumPy inverse, outside the fitted kernel."""
        if set(original) != set(self.shapes):
            raise ValueError('Original control sites do not match the coordinate map')
        if any(np.shape(original[name]) != shape for name, shape in self.shapes.items()):
            raise ValueError('Original control dimensions do not match the coordinate map')
        blocks = []
        for group_index, row, coordinates in self.inverse_order:
            group = self.groups[group_index]
            values = np.array([np.asarray(original[name])[coord] for name, coord in coordinates])
            blocks.append(group.basis[row].T @ (values / group.scales[row]))
        return np.concatenate(blocks)


def prepare(rotation):
    """Copy the full existing map; no responses, fits or new eigenbasis enter."""
    if type(rotation.size) is not int or not 0 < rotation.size <= np.iinfo(np.int32).max:
        raise ValueError('A complete control map with supported indices is required')
    shapes = {name: tuple(shape) for name, shape in rotation.shapes.items()}
    if (not shapes or any(not isinstance(name, str) or not name for name in shapes)
            or any(type(d) is not int or d < 0 for shape in shapes.values() for d in shape)
            or sum(math.prod(shape) for shape in shapes.values()) != rotation.size):
        raise ValueError('Original site shapes must cover the complete control vector')
    inventory, seen, offset = [], set(), 0
    for block in rotation.blocks:
        coordinates = tuple((name, tuple(coordinate)) for name, coordinate in block['coordinates'])
        k = len(coordinates)
        basis, scales = np.asarray(block['basis']), np.asarray(block['scales'])
        if (not k or block['start'] != offset or block['stop'] != offset+k
                or basis.shape != (k, k) or scales.shape != (k,)
                or basis.dtype.kind != 'f' or scales.dtype.kind != 'f'
                or not np.isfinite(basis).all() or not np.isfinite(scales).all()
                or (scales <= 0).any()):
            raise ValueError('Invalid original control block or latent assignment')
        for name, coordinate in coordinates:
            if (name not in shapes or len(coordinate) != len(shapes[name])
                    or any(type(i) is not int or not 0 <= i < d for i, d in zip(coordinate, shapes[name]))
                    or (name, coordinate) in seen):
                raise ValueError('Every original output coordinate must appear exactly once')
            seen.add((name, coordinate))
        np.testing.assert_allclose(basis.T @ basis, np.eye(k), rtol=0, atol=1e-12)
        inventory.append(dict(coordinates=coordinates, basis=basis, scales=scales,
                              start=offset, stop=offset+k, key=(k, basis.dtype.str, scales.dtype.str)))
        offset += k
    expected = {(name, coordinate) for name, shape in shapes.items() for coordinate in np.ndindex(shape)}
    if offset != rotation.size or seen != expected:
        raise ValueError('The control map omitted an original direction')
    keys = sorted({block['key'] for block in inventory})
    groups, locations, inverse_order, coefficient_index = [], {}, {}, 0
    for group_index, key in enumerate(keys):
        entries = [(i, block) for i, block in enumerate(inventory) if block['key'] == key]
        groups.append(_Group(
            basis=_readonly(np.stack([b['basis'] for _, b in entries])),
            scales=_readonly(np.stack([b['scales'] for _, b in entries])),
            latent_indices=_readonly(np.array([np.arange(b['start'], b['stop']) for _, b in entries], dtype=np.int32))))
        for row, (i, block) in enumerate(entries):
            inverse_order[i] = (group_index, row, block['coordinates'])
            for coordinate in block['coordinates']:
                locations[coordinate] = coefficient_index
                coefficient_index += 1
    gathers = {name: _readonly(np.array([locations[(name, coordinate)] for coordinate in np.ndindex(shape)], dtype=np.int32))
               for name, shape in shapes.items()}
    return BatchedControlRotation(size=rotation.size, shapes=MappingProxyType(shapes), groups=tuple(groups),
        gathers=MappingProxyType(gathers), inverse_order=tuple(inverse_order[i] for i in range(len(inventory))),
        source_fingerprint=rotation.fingerprint(), log_abs_det=rotation.log_abs_det)
