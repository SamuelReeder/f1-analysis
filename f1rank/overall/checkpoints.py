"""Atomic, same-target sampler checkpoints; never accepted posterior caches.

Only completed generations count. A process killed before the directory rename
leaves an ignored partial generation; a process killed afterwards has committed
both the samples and the state that follows them. Files contain JSON and numeric
arrays, not pickle or importable class names supplied by a checkpoint.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

SCHEMA = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def _arrays(path, values):
    import numpy as np
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())


def _numeric(value):
    import numpy as np
    value = np.asarray(value)
    if (value.dtype.fields or value.dtype.kind not in "biufc"
            or not np.isfinite(value).all()):
        raise ValueError("Checkpoint arrays must contain finite plain numeric values")
    return value


def _types():
    from numpyro.infer.hmc import HMCState
    from numpyro.infer.hmc_util import HMCAdaptState
    return {"HMCState": HMCState, "HMCAdaptState": HMCAdaptState}


def encode_state(state):
    """Preserve NumPyro containers, array types and typed random keys explicitly."""
    import jax
    import numpy as np
    leaves, classes = {}, _types()

    def encode(value):
        if value is None or type(value) in (str, bool, int):
            return {"kind": "literal", "value": value}
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Nonfinite scalar in sampler state")
            return {"kind": "literal", "value": value}
        if type(value) in classes.values():
            return {"kind": "state", "type": type(value).__name__,
                    "fields": list(value._fields), "items": [encode(v) for v in value]}
        if type(value) in (dict, list, tuple):
            if type(value) is dict:
                return {"kind": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
            return {"kind": type(value).__name__, "items": [encode(v) for v in value]}
        if isinstance(value, (jax.Array, np.ndarray, np.generic)):
            name = f"leaf_{len(leaves):06d}"
            typed_key = isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            weak = bool(getattr(value, "weak_type", False))
            data = _numeric(jax.random.key_data(value) if typed_key else value)
            leaves[name] = data
            return {"kind": "array", "name": name, "shape": list(data.shape), "dtype": data.dtype.str,
                    "family": "jax" if isinstance(value, jax.Array) else
                              "numpy_scalar" if isinstance(value, np.generic) else "numpy",
                    "weak": weak,
                    "key_impl": str(jax.random.key_impl(value)) if typed_key else None}
        raise ValueError(f"Unsupported sampler state value: {type(value).__name__}")

    tree = encode(state)
    if tree.get("type") != "HMCState":
        raise ValueError("A checkpoint requires the complete HMCState")
    return tree, leaves


def decode_state(tree, leaves):
    import jax
    import jax.numpy as jnp
    import numpy as np
    classes, used = _types(), set()

    def decode(node):
        kind = node["kind"]
        if kind == "literal":
            value = node["value"]
            if value is not None and type(value) not in (str, bool, int, float):
                raise ValueError("Invalid state literal")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("Nonfinite state literal")
            return value
        if kind == "array":
            name = node["name"]
            if name in used or name not in leaves:
                raise ValueError("Repeated or missing state array")
            used.add(name)
            value = _numeric(leaves[name])
            if list(value.shape) != node["shape"] or value.dtype.str != node["dtype"]:
                raise ValueError("State array shape or dtype changed")
            family = node["family"]
            if family == "jax":
                if node["key_impl"] is not None:
                    return jax.random.wrap_key_data(jnp.asarray(value), impl=node["key_impl"])
                # Parallel chains can yield nonscalar weak arrays. The public
                # conversion API cannot restore this attribute; bind this small
                # compatibility dependency in runtime_identity and replay tests.
                from jax._src.lax.lax import _convert_element_type
                result = _convert_element_type(value, new_dtype=value.dtype, weak_type=node["weak"])
                if result.dtype != value.dtype or bool(result.weak_type) != node["weak"]:
                    raise ValueError("Runtime changed sampler dtype or weak typing")
                return result
            if node["weak"] or node["key_impl"] is not None:
                raise ValueError("Invalid NumPy state metadata")
            if family == "numpy_scalar" and value.shape == ():
                return value[()]
            if family == "numpy":
                return value
            raise ValueError("Unsupported state array family")
        if kind == "dict":
            pairs = [(decode(k), decode(v)) for k, v in node["items"]]
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError("Duplicate state dictionary key")
            return result
        if kind in ("list", "tuple", "state"):
            values = [decode(v) for v in node["items"]]
            if kind == "state":
                cls = classes.get(node["type"])
                if cls is None or node["fields"] != list(cls._fields):
                    raise ValueError("Sampler state layout changed")
                return cls(*values)
            return tuple(values) if kind == "tuple" else values
        raise ValueError("Unknown state descriptor")

    state = decode(tree)
    if type(state) is not classes["HMCState"] or used != set(leaves):
        raise ValueError("Incomplete state inventory")
    return state


def _load_arrays(path):
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(set(names)) != len(names) or
                any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.npy", n) for n in names)):
            raise ValueError("Invalid checkpoint array inventory")
    with np.load(path, allow_pickle=False) as archive:
        return {k: _numeric(archive[k]) for k in archive.files}


def _array_shapes(path):
    """Inspect compressed members without allocating their retained draw arrays."""
    import numpy as np
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.npy", member.filename):
                raise ValueError("Invalid checkpoint array inventory")
            name = member.filename[:-4]
            if name in result:
                raise ValueError("Duplicate checkpoint array")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported checkpoint array format")
                if (dtype.fields or dtype.kind not in "biufc"
                        or stream.tell()+math.prod(shape)*dtype.itemsize != member.file_size):
                    raise ValueError("Invalid checkpoint array header or length")
                result[name] = {"shape": list(shape), "dtype": dtype.str}
    return result


class Checkpoints:
    """Exclusive writer/reader for an immutable registered attempt identity."""

    def __init__(self, root, identity):
        self.root = Path(root)
        self.identity = json.loads(canonical(identity))
        self.identity_key = hashlib.sha256(canonical(self.identity)).hexdigest()
        self.lock = None
        self.records = []
        self.sealed_files = {}

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.root / "identity.json"
            if path.exists():
                if path.read_bytes() != canonical(self.identity):
                    raise ValueError("Checkpoint belongs to a different target, settings or runtime")
            else:
                if any(p.name.startswith("checkpoint-") for p in self.root.iterdir()):
                    raise ValueError("Checkpoint identity is missing")
                descriptor, temporary = tempfile.mkstemp(prefix=".identity-", dir=self.root)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(canonical(self.identity))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                _sync_directory(self.root)
            versions = self._file_versions()
            self.records = self._read_records()
            if versions != self._file_versions():
                raise ValueError("Checkpoint history changed during verification")
            self.sealed_files = versions
            return self
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def __exit__(self, *exc):
        self.lock.close()
        self.lock = None

    def _require_lock(self):
        if self.lock is None:
            raise RuntimeError("Open checkpoints in an exclusive context")
        if (self.root / "identity.json").read_bytes() != canonical(self.identity):
            raise ValueError("Checkpoint identity changed during use")

    def _file_versions(self):
        """Cheap change detection between full byte verification at open/read."""
        result = {}
        for directory in sorted(self.root.glob("checkpoint-*")):
            for path in (directory, *sorted(directory.iterdir())):
                stat = path.stat()
                result[str(path.relative_to(self.root))] = (
                    stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return result

    def _unchanged(self):
        # Avoid hashing the entire growing posterior at every chunk boundary.
        # The exclusive lock protects cooperating writers; filesystem versions
        # catch replacement or modification by another process.
        if self._file_versions() != self.sealed_files:
            raise ValueError("Checkpoint history changed during use")

    def _read_records(self):
        records, previous, draws, inventory = [], None, 0, None
        for index, directory in enumerate(sorted(self.root.glob("checkpoint-*"))):
            if directory.name != f"checkpoint-{index:06d}" or not directory.is_dir():
                raise ValueError("Checkpoint generations are not contiguous")
            path = directory / "manifest.json"
            record = json.loads(path.read_text())
            if (record["schema"] != SCHEMA or record["identity_key"] != self.identity_key
                    or record["generation"] != index or record["previous"] != previous):
                raise ValueError("Broken checkpoint identity or manifest chain")
            expected = {"state.npz"} if index == 0 else {"state.npz", "posterior.npz", "extras.npz"}
            if set(record["files"]) != expected or {p.name for p in directory.iterdir()} != expected | {"manifest.json"}:
                raise ValueError("Incomplete checkpoint generation")
            for name, checksum in record["files"].items():
                if digest(directory / name) != checksum:
                    raise ValueError("Checkpoint bytes changed")
            end = record["draws"]
            if (type(end) is not int or (end != 0 if index == 0 else end <= draws)
                    or record["start_draw"] != draws or end > self.identity["settings"]["samples"]):
                raise ValueError("Overlapping or missing checkpoint draws")
            if index:
                axes = [self.identity["settings"]["chains"], end-draws]
                if (not isinstance(record["inventory"], dict) or
                        set(record["inventory"]) != {"posterior", "extras"}):
                    raise ValueError("Missing checkpoint site inventory")
                for kind in ("posterior", "extras"):
                    actual = _array_shapes(directory / f"{kind}.npz")
                    expected_inventory = record["inventory"][kind]
                    if not actual or actual.keys() != expected_inventory.keys():
                        raise ValueError("Checkpoint site inventory does not match arrays")
                    for name, spec in actual.items():
                        expected_spec = expected_inventory[name]
                        if (spec["shape"][:2] != axes or
                                [spec["shape"][0], *spec["shape"][2:]] != expected_spec["shape"] or
                                spec["dtype"] != expected_spec["dtype"] or
                                (kind == "extras" and spec["shape"] != axes)):
                            raise ValueError("Checkpoint array dimensions or dtype changed")
                if inventory is not None and record["inventory"] != inventory:
                    raise ValueError("Posterior site inventory changed between chunks")
                inventory = record["inventory"]
            elif record["inventory"] is not None:
                raise ValueError("Warmup checkpoint cannot contain posterior draws")
            records.append(record)
            previous, draws = digest(path), end
        return records

    @property
    def draws(self):
        return self.records[-1]["draws"] if self.records else 0

    def state(self):
        self._require_lock()
        if not self.records:
            return None
        self._unchanged()
        directory = self.root / f"checkpoint-{len(self.records)-1:06d}"
        state = decode_state(self.records[-1]["tree"], _load_arrays(directory / "state.npz"))
        self._validate_iteration(state, self.draws)
        self._unchanged()
        return state

    def _validate_iteration(self, state, draws):
        import numpy as np
        settings = self.identity["settings"]
        indices = np.asarray(state.i)
        expected_shape = () if settings["chains"] == 1 else (settings["chains"],)
        if (indices.dtype.kind not in "iu" or indices.shape != expected_shape
                or not np.all(indices == settings["warmup"]+draws)
                or draws > settings["samples"]):
            raise ValueError("Sampler iteration and committed draw counts disagree")

    def append(self, state, *, posterior=None, extras=None):
        self._require_lock()
        self._unchanged()
        tree, leaves = encode_state(state)
        generation, start = len(self.records), self.draws
        inventory = None
        draws = start
        if generation == 0:
            if posterior is not None or extras is not None:
                raise ValueError("Save the warmup state before retained sampling")
        else:
            if not posterior or not extras:
                raise ValueError("Retain all posterior samples and sampler extras")
            posterior = {k: _numeric(v) for k, v in posterior.items()}
            extras = {k: _numeric(v) for k, v in extras.items()}
            names = list(posterior) + list(extras)
            if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) for k in names):
                raise ValueError("Invalid sampled site name")
            shape = next(iter(posterior.values())).shape
            if len(shape) < 2 or min(shape[:2]) < 1:
                raise ValueError("Posterior requires complete grouped chain/draw axes")
            axes = shape[:2]
            if any(v.shape[:2] != axes for v in posterior.values()) or any(v.shape != axes for v in extras.values()):
                raise ValueError("Posterior and extras must share complete grouped axes")
            inventory = {kind: {k: {"shape": [v.shape[0], *v.shape[2:]], "dtype": v.dtype.str}
                                for k, v in values.items()}
                         for kind, values in (("posterior", posterior), ("extras", extras))}
            if generation > 1 and inventory != self.records[-1]["inventory"]:
                raise ValueError("Posterior site inventory changed between chunks")
            draws += axes[1]
        self._validate_iteration(state, draws)
        if generation and axes[0] != self.identity["settings"]["chains"]:
            raise ValueError("Checkpoint chain count changed")
        directory = Path(tempfile.mkdtemp(prefix=".partial-", dir=self.root))
        _arrays(directory / "state.npz", leaves)
        if generation:
            _arrays(directory / "posterior.npz", posterior)
            _arrays(directory / "extras.npz", extras)
        record = dict(schema=SCHEMA, identity_key=self.identity_key, generation=generation,
                      previous=digest(self.root / f"checkpoint-{generation-1:06d}" / "manifest.json")
                      if generation else None, start_draw=start, draws=draws, inventory=inventory,
                      tree=tree, files={p.name: digest(p) for p in sorted(directory.iterdir())})
        _json(directory / "manifest.json", record)
        _sync_directory(directory)
        self._require_lock()
        self._unchanged()
        destination = self.root / f"checkpoint-{generation:06d}"
        directory.rename(destination)
        _sync_directory(self.root)
        versions = self._file_versions()
        if any(versions.get(name) != value for name, value in self.sealed_files.items()):
            raise ValueError("Checkpoint history changed during commit")
        self.records.append(record)
        self.sealed_files = versions
        return record

    def chunks(self):
        """Yield bounded grouped arrays; this does not grant fit acceptance."""
        self._require_lock()
        if self._read_records() != self.records:
            raise ValueError("Checkpoint history changed during use")
        for index, record in enumerate(self.records[1:], 1):
            directory = self.root / f"checkpoint-{index:06d}"
            posterior = _load_arrays(directory / "posterior.npz")
            extras = _load_arrays(directory / "extras.npz")
            axes = (self.identity["settings"]["chains"], record["draws"]-record["start_draw"])
            for kind, values in (("posterior", posterior), ("extras", extras)):
                if set(values) != set(record["inventory"][kind]):
                    raise ValueError("Checkpoint site inventory does not match arrays")
                for name, value in values.items():
                    expected = record["inventory"][kind][name]
                    if (value.shape[:2] != axes or
                            [value.shape[0], *value.shape[2:]] != expected["shape"] or
                            value.dtype.str != expected["dtype"] or
                            (kind == "extras" and value.shape != axes)):
                        raise ValueError("Checkpoint array dimensions or dtype changed")
            yield record, posterior, extras
        self._unchanged()
