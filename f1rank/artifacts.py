"""Fail-closed provenance for racing outputs and checks before publishing a fit.

A manifest is written last. Readers verify inputs, code and every output; an old,
partially written or mixed-generation artifact cannot enter a downstream model.
"""

import datetime as dt
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1


class StaleArtifact(RuntimeError):
    pass


def digest(path: Path) -> str:
    path = path.resolve()
    s = path.stat()
    return _digest(str(path), s.st_size, s.st_mtime_ns, s.st_ctime_ns)


@lru_cache(maxsize=256)
def _digest(path: str, size: int, mtime: int, ctime: int) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_files() -> list[Path]:
    """Conservative common snapshot: corrections or code changes invalidate racing outputs."""
    return sorted([*(ROOT / "data" / "processed").glob("*.*"),
                   *(ROOT / "data" / "overrides").glob("*.*"),
                   *(ROOT / "data" / "reference").glob("*.*"),
                   *(ROOT / "f1rank").glob("*.py"), *(ROOT / "extract").glob("*.py")])


def atomic_json(path: Path, value: dict) -> None:
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=1, allow_nan=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_once(path: Path, obj: dict) -> bool:
    """Write JSON to a new file; never overwrite. Returns False if the file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x") as f:
            json.dump(obj, f, indent=1, default=float)
    except FileExistsError:
        return False
    return True


def record(directory: Path, outputs: list[Path], *, model: str, inputs=None,
           details=None, name="manifest.json") -> dict:
    inputs = input_files() if inputs is None else list(inputs)
    def label(p):
        p = p.resolve()
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    def hashes(paths):
        return {label(p): digest(p) for p in sorted(set(paths))}
    meta = {"schema_version": SCHEMA_VERSION, "model": model,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "inputs": hashes(inputs), "outputs": hashes(outputs), "details": details or {}}
    meta["fit_id"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:20]
    events = ROOT / "data" / "processed" / "events.parquet"
    if events in inputs:
        meta["data_as_of"] = str(pd.read_parquet(events).event_id.max())
    atomic_json(directory / name, meta)
    return meta


def require(directory: Path, *, name="manifest.json", required_outputs=()) -> dict:
    path = directory / name
    try:
        meta = json.loads(path.read_text())
        if not isinstance(meta, dict) or meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported manifest")
        for group in ("inputs", "outputs"):
            if not isinstance(meta[group], dict):
                raise ValueError(f"invalid {group} manifest")
            for file, expected in meta[group].items():
                if digest(ROOT / file) != expected:
                    raise ValueError(f"changed {file}")
        for output in required_outputs:
            output = Path(output).resolve()
            label = str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)
            if label not in meta["outputs"]:
                raise ValueError(f"unrecorded output {output}")
    except (OSError, ValueError, KeyError) as e:
        raise StaleArtifact(f"{directory}: missing or stale provenance ({e}); regenerate this artifact") from e
    return meta


def diagnostics(grouped: dict, divergences: int) -> dict:
    """Check every supplied parameter, including nuisance terms; reject nonfinite draws."""
    from numpyro.diagnostics import split_gelman_rubin
    if not grouped:
        raise ValueError("no posterior parameters")
    first = np.asarray(next(iter(grouped.values())))
    chains, samples = first.shape[:2]
    worst, finite = 1.0, True
    for key, values in grouped.items():
        a = np.asarray(values)
        if a.shape[:2] != (chains, samples) or not np.isfinite(a).all():
            finite = False
            continue
        if chains < 2 or samples < 4:
            continue
        # Deterministic, structurally constant coordinates (e.g. a zero circuit factor)
        # have no R-hat. All varying coordinates must have a finite R-hat.
        varying = np.ptp(a.reshape(chains * samples, -1), axis=0) > 0
        if varying.any():
            rhat = np.asarray(split_gelman_rubin(a.reshape(chains, samples, -1)[:, :, varying]))
            finite = finite and bool(np.isfinite(rhat).all())
            worst = max(worst, float(np.max(rhat)))
    ok = (finite and chains >= 2 and samples >= 4 and worst < 1.05
          and 0 <= divergences <= chains * samples / 1000)
    return {"rhat_max": worst, "divergences": int(divergences), "n_draws": chains * samples,
            "finite": finite, "converged": bool(ok)}


def require_convergence(result: dict) -> None:
    if not result.get("converged", False):
        raise RuntimeError(f"Fit failed publication diagnostics: {result}")


# Retry ladder for the racing-quality MCMC fits (docs/racing_approach.md, amendment of
# 2026-10-04): (warm-up multiple, draws multiple, target acceptance; None = the fit's own).
# The first attempt is the fit as it was made before the amendment. The ratios are those
# of jobs.ATTEMPTS (700+400, 1,400+800, 2,100+1,600, 2,100+1,600 at 0.98).
RETRY = ((1, 1, None), (2, 2, None), (3, 4, None), (3, 4, 0.98))
_FITS: list[list[dict]] = []


def fit_until_converged(run, warmup: int, samples: int, seed: int = 0):
    """Run `run(warmup, samples, target_accept, seed) -> (mcmc, diagnostics)` under RETRY
    until the publication check passes; return the converged sampler. When every attempt
    fails, raise as require_convergence does. Each attempt is printed and recorded."""
    attempts = []
    for k, (w, s, accept) in enumerate(RETRY):
        settings = {"warmup": warmup * w, "samples": samples * s, "target_accept": accept, "seed": seed + k}
        mcmc, result = run(settings["warmup"], settings["samples"], accept, settings["seed"])
        attempts.append({**settings, **result})
        print(f"fit attempt {k + 1}: {result}", flush=True)
        if result.get("converged", False):
            _FITS.append(attempts)
            return mcmc
    _FITS.append(attempts)
    raise RuntimeError(f"Fit failed publication diagnostics after {len(attempts)} attempts: {attempts[-1]}")


def fit_record() -> dict:
    """The fits made by this process and the attempts of any that needed more than one,
    for the summary file (`fit_attempts`)."""
    return {"fits": len(_FITS), "retried": [a for a in _FITS if len(a) > 1]}
