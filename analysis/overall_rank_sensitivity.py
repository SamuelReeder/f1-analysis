"""Conditional simulation-error bounds; never functional MCMC acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.overall_simulation_moments import FILENAME, load_checked

SOURCES = ("analysis/overall_rank_sensitivity.py", "docs/overall_rank_sensitivity.md",
           "analysis/overall_simulation_moments.py", "contracts/overall.json")
CONTRACT = json.loads((ROOT / "contracts/overall.json").read_text())
MAX_POINTS = float(max(CONTRACT["points"]))
PROBES = CONTRACT["required_explanations"]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


LOADED_SOURCE_HASHES = {name: sha(ROOT / name) for name in SOURCES}


def _field(field, n):
    if (not isinstance(field, (tuple, list)) or len(field) != n or n < 2
            or any(not isinstance(v, str) or not v.strip() for v in field)
            or len(set(field)) != n):
        raise ValueError("Retain the complete unique field")


def _mean_interval(lower, upper, axis=None):
    # Conservative floating-point summation/division allowance. This is not an
    # extra sampling tolerance and never determines a scientific pass/fail flag.
    n = lower.size
    eps = np.finfo(float).eps
    scale = max(1., float(np.max(np.abs(lower))), float(np.max(np.abs(upper))))
    guard = (n + 2) * eps / (1 - (n + 2) * eps) * scale
    return np.stack([lower.mean(axis=axis) - guard, upper.mean(axis=axis) + guard], axis=-1).tolist()


def ordering_bounds(lower, upper, field):
    """Finite-trace ordering bounds for every value in the supplied boxes.

    These are expected-point ranks, not sporting ranks of simulated seasons.
    Ties remain full rank intervals; no driver identifier is a tiebreak.
    """
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (lower.ndim != 3 or lower.shape != upper.shape or min(lower.shape[:2]) < 2
            or lower.dtype.kind not in "fiu" or upper.dtype.kind not in "fiu"
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or (lower > upper).any()):
        raise ValueError("Need complete finite grouped mean intervals")
    lower, upper = lower.astype(float), upper.astype(float)
    n = lower.shape[-1]
    _field(field, n)
    best = np.ones(lower.shape, dtype=int)
    worst = np.full(lower.shape, n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                best[..., i] += lower[..., j] > upper[..., i]
                worst[..., i] -= upper[..., j] < lower[..., i]
    levels = np.arange(1, n + 1)
    rows, pairs = [], []
    for i, driver in enumerate(field):
        cdf_low = worst[..., i, None] <= levels
        cdf_high = best[..., i, None] <= levels
        rows.append(dict(driver_id=driver,
            trace_mean_interval=_mean_interval(lower[..., i], upper[..., i]),
            chain_mean_intervals=_mean_interval(lower[..., i], upper[..., i], axis=1),
            rank_cdf_bounds=dict(levels=levels.tolist(), lower=cdf_low.mean(axis=(0, 1)).tolist(),
                upper=cdf_high.mean(axis=(0, 1)).tolist(),
                lower_by_chain=cdf_low.mean(axis=1).tolist(), upper_by_chain=cdf_high.mean(axis=1).tolist())))
        for j in range(i):
            # Endpoint comparisons avoid replacing an exact equality with a
            # chosen ordering. No independence between the drivers is needed.
            predicates = {
                "a_greater": (lower[..., i] > upper[..., j], upper[..., i] > lower[..., j]),
                "b_greater": (lower[..., j] > upper[..., i], upper[..., j] > lower[..., i]),
                "equal": ((lower[..., i] == upper[..., i]) & (lower[..., j] == upper[..., j])
                          & (lower[..., i] == lower[..., j]),
                          (lower[..., i] <= upper[..., j]) & (upper[..., i] >= lower[..., j]))}
            fractions = {name: dict(lower=float(lo.mean()), upper=float(hi.mean()),
                lower_by_chain=lo.mean(axis=1).tolist(), upper_by_chain=hi.mean(axis=1).tolist())
                for name, (lo, hi) in predicates.items()}
            lo = np.nextafter(lower[..., i] - upper[..., j], -np.inf)
            hi = np.nextafter(upper[..., i] - lower[..., j], np.inf)
            pairs.append(dict(driver_a=driver, driver_b=field[j], empirical_fraction_bounds=fractions,
                trace_mean_difference_interval=_mean_interval(lo, hi),
                chain_mean_difference_intervals=_mean_interval(lo, hi, axis=1)))
    return dict(field=list(field), posterior_shape=list(lower.shape[:2]),
        complete_draw_and_field_inventory=True, drivers=rows, pairs=pairs,
        interpretation="Bounds on the complete fixed trace, including all tie-respecting orderings; not authenticated posterior probabilities or season ranks.")


def envelope(points_mean, replicate_counts, field, *, error_probability, probe_loss_mean=None):
    """Simultaneous Hoeffding/union bounds, conditional on the fixed trace.

    Independent replicated seasons within each draw are an assumption of this
    calculation. It never assumes independence or convergence of posterior draws.
    """
    means, counts = np.asarray(points_mean), np.asarray(replicate_counts)
    if (means.ndim != 3 or min(means.shape[:2]) < 2 or means.dtype.kind != "f"
            or not np.isfinite(means).all() or ((means < 0) | (means > MAX_POINTS)).any()
            or counts.shape != means.shape[:2] or counts.dtype.kind not in "iu"
            or ((counts < 2) | (counts > np.iinfo(np.int64).max)).any()):
        raise ValueError("Need complete bounded point means and positive replicate counts")
    _field(field, means.shape[-1])
    if (type(error_probability) not in (float, int) or not np.isfinite(error_probability)
            or not 0 < error_probability < 1):
        raise ValueError("Declare an error probability strictly between zero and one")
    losses = None if probe_loss_mean is None else np.asarray(probe_loss_mean)
    if losses is not None and (losses.shape != means.shape + (len(PROBES),)
            or losses.dtype.kind != "f" or not np.isfinite(losses).all()
            or ((losses < -MAX_POINTS) | (losses > MAX_POINTS)).any()):
        raise ValueError("Retain every bounded paired explanation mean")
    family = means.size + (0 if losses is None else losses.size)
    # Log sums avoid overflow for very small declared error probabilities.
    log_factor = np.log(2.) + np.log(family) - np.log(error_probability)
    radius = MAX_POINTS * np.sqrt(log_factor / (2 * counts.astype(float)))
    radius = np.nextafter(radius * (1 + 64 * np.finfo(float).eps), np.inf)
    low = np.maximum(0., np.nextafter(means - radius[..., None], -np.inf))
    high = np.minimum(MAX_POINTS, np.nextafter(means + radius[..., None], np.inf))
    result = ordering_bounds(low, high, field)
    explanations = []
    if losses is not None:
        loss_low = np.maximum(-MAX_POINTS, np.nextafter(losses - 2 * radius[..., None, None], -np.inf))
        loss_high = np.minimum(MAX_POINTS, np.nextafter(losses + 2 * radius[..., None, None], np.inf))
        for i, driver in enumerate(field):
            explanations.append(dict(driver_id=driver, losses={name: dict(
                trace_mean_interval=_mean_interval(loss_low[..., i, k], loss_high[..., i, k]),
                chain_mean_intervals=_mean_interval(loss_low[..., i, k], loss_high[..., i, k], axis=1))
                for k, name in enumerate(PROBES)}))
    result.update(purpose="conditional_simulation_ordering_sensitivity", error_probability=float(error_probability),
        family_size=family,
        family="all_baseline_and_required_explanation_means" if losses is not None else "points_only_fixture",
        units="points_per_grand_prix", outcome_bounds=[0., MAX_POINTS],
        replicate_counts=counts.tolist(), point_radius_by_draw=radius.tolist(), explanations=explanations,
        assumptions=["independent replicated seasons conditional on each fixed draw",
                     "bounded points and fixed complete draw, field and replicate inventories",
                     "declared error probability; no outcome-dependent stopping or repetition"],
        conditional_simulation_error_only=True, mcmc_precision_verified=False,
        rank_precision_established=False, scientific_evidence_verified=False, publication_allowed=False)
    return result


def source_fingerprints():
    if json.loads((ROOT / "contracts/overall.json").read_text()) != CONTRACT:
        raise ValueError("Scoring contract changed after import")
    current = {name: sha(ROOT / name) for name in SOURCES}
    if current != LOADED_SOURCE_HASHES:
        raise ValueError("Diagnostic sources changed after import")
    return current


def analyse_checked(folder, *, expected_simulation_key, expected_archive_sha256, error_probability, output_dir):
    folder, destination = Path(folder), Path(output_dir)
    if destination.exists():
        raise FileExistsError("Preserve the existing diagnostic result")
    sources = source_fingerprints()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, fingerprint in sources.items():
        content = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != fingerprint:
            raise ValueError("Commit diagnostic sources before analysis")
    inputs = {name: sha(folder / name) for name in (FILENAME, "receipt.json", "summary.json")}
    meta, arrays = load_checked(folder, expected_simulation_key=expected_simulation_key,
                               expected_archive_sha256=expected_archive_sha256)
    result = envelope(arrays["points_mean"], arrays["replicate_counts"], meta["field"],
        error_probability=error_probability, probe_loss_mean=arrays["probe_loss_mean"])
    result.update(source_commit=source_commit, source_sha256=sources,
        input_sha256=inputs, input_identity=meta["identity"], input_sources=meta["sources"],
        analysis_of_completed_development_companion=True)
    if (sources != source_fingerprints()
            or any(sha(folder / name) != fingerprint for name, fingerprint in inputs.items())):
        raise ValueError("Diagnostic sources or complete input changed during analysis")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "sensitivity.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--simulation-key", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--error-probability", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    analyse_checked(args.input, expected_simulation_key=args.simulation_key,
        expected_archive_sha256=args.archive_sha256, error_probability=args.error_probability, output_dir=args.output_dir)
