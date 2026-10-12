"""Post hoc saved-draw diagnostics, including failed fits; never inference output."""
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.overall_posterior_archive import _mapped_numeric_archive
from analysis.overall_ranking_resources import scope_limits
from f1rank.overall.diagnostics import check
from f1rank.overall.provenance import require_snapshot

PROFILE = "overall-control-rotation-profile-v1"
SCALARS = ("sd_driver_form", "overall_sd_day", "overall_shared_weekend",
    "sd_car_event", "overall_sd_car_day", "overall_qualifying_link",
    "overall_car_link", "overall_team_link", "sd_compat", "overall_sd_team",
    "sd_level", "overall_sd_level", "sd_skill_race", "sd_skill_season",
    "overall_sd_drift_race", "overall_sd_drift_season", "sigma0", "sigma_tau",
    "overall_sd_lap", "overall_lap_rho")
VECTORS = ("driver_form_z", "driver_form", "overall_day_z", "overall_day",
    "overall_car_day_z", "skill", "overall_race_skill", "compat", "overall_team",
    "car", "car_event", "car_track", "overall_car")
SOURCES = ("analysis/overall_weekend_diagnosis.py", "docs/overall_weekend_diagnosis.md",
    "analysis/overall_posterior_archive.py", "analysis/overall_ranking_resources.py")
METRICS = ("rhat_max", "bulk_ess_min", "tail_ess_min", "mcse_sd_max")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def centre(values, groups):
    values, groups = np.asarray(values, float), np.asarray(groups)
    if (groups.ndim != 1 or groups.dtype.kind not in "iu" or (groups < 0).any()
            or values.shape[-1] != len(groups) or not np.isfinite(values).all()):
        raise ValueError("Invalid grouped state")
    result = values.copy()
    for group in np.unique(groups):
        selected = groups == group
        result[..., selected] -= values[..., selected].mean(axis=-1, keepdims=True)
    return result


def weekend_moments(sf, sr, shared):
    sf, sr, shared = (np.asarray(value, float) for value in (sf, sr, shared))
    if (sf.shape != sr.shape or sf.shape != shared.shape or (sf <= 0).any()
            or (sr <= 0).any() or not all(np.isfinite(v).all() for v in (sf, sr, shared))):
        raise ValueError("Weekend moments require matching finite draws and positive scales")
    marginal = np.hypot(shared * sf, sr)
    return dict(race_form_marginal_sd=marginal, form_cross_covariance=shared * sf**2,
        form_correlation=shared * sf / marginal,
        independent_race_variance_fraction=(sr / marginal)**2)


def correlation_matrix(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Correlation requires finite observations by quantity")
    varying = np.ptp(values, axis=0) > 0
    indices = np.flatnonzero(varying)
    result = [[None for _ in varying] for _ in varying]
    if len(indices):
        correlation = np.atleast_2d(np.corrcoef(values[:, indices], rowvar=False))
        if not np.isfinite(correlation).all():
            raise ValueError("Invalid correlation among varying quantities")
        for i, row in enumerate(indices):
            for j, col in enumerate(indices):
                result[row][col] = float(correlation[i, j])
    return dict(values=result, constant_columns=np.flatnonzero(~varying).tolist())


def describe(quantities, divergences):
    names = list(quantities)
    values = np.stack([quantities[name] for name in names], axis=-1)
    if values.ndim != 3:
        raise ValueError("Retain the complete chain and draw axes")
    return dict(names=names, diagnostics=check(quantities, divergences),
        chain_means=values.mean(axis=1).tolist(),
        chain_standard_deviations=values.std(axis=1, ddof=1).tolist(),
        chain_quantile_probabilities=[.05, .5, .95],
        chain_quantiles=np.quantile(values, [.05, .5, .95], axis=1).transpose(1, 0, 2).tolist(),
        correlations_by_chain=[correlation_matrix(chain) for chain in values],
        pooled_correlations=correlation_matrix(values.reshape(-1, len(names))))


def derived_quantities(saved, q, outcomes):
    sf, sr, shared = (saved[k] for k in ("sd_driver_form", "overall_sd_day", "overall_shared_weekend"))
    scalar = {name: np.asarray(saved[name], float) for name in SCALARS}
    scalar.update(weekend_moments(sf, sr, shared))
    zq = centre(saved["driver_form_z"], q["entry_event"])
    zr = centre(saved["overall_day_z"], q["entry_event"])
    zc = centre(saved["overall_car_day_z"], q["car_event"])
    form = np.asarray(sf)[..., None] * zq
    independent = np.asarray(sr)[..., None] * zr
    day = np.asarray(shared)[..., None] * form + independent
    car_day = np.asarray(saved["overall_sd_car_day"])[..., None] * zc
    # Recorded states were calculated in the sampler's floating-point dtype.
    # This tolerance checks reconstruction only; it is not a convergence gate.
    errors = {}
    for name, reconstructed in (("driver_form", form), ("overall_day", day)):
        actual = np.asarray(saved[name])
        tolerance = (64 * np.finfo(actual.dtype).eps * actual.shape[-1]
                     * max(1., float(np.abs(actual).max()), float(np.abs(reconstructed).max())))
        error = float(np.abs(actual - reconstructed).max())
        if error > tolerance:
            raise ValueError(f"Stored {name} does not reproduce its model definition")
        errors[name] = dict(max_absolute_error=error, allowed_roundoff=float(tolerance))
    for name, values in (("centred_qualifying_innovation", zq), ("centred_race_innovation", zr),
            ("centred_car_innovation", zc), ("qualifying_form", form),
            ("independent_race_form", independent), ("total_race_form", day),
            ("race_car_day", car_day)):
        scalar[name + "_rms"] = np.sqrt(np.mean(values**2, axis=-1))
    mapping = np.asarray(outcomes["entry_car"])
    fields = dict(qualifying_ability=np.asarray(saved["skill"]), qualifying_team=np.asarray(saved["compat"]),
        qualifying_form=np.asarray(saved["driver_form"]),
        qualifying_car=(np.asarray(saved["car"]) + saved["car_event"] + saved["car_track"])[..., mapping],
        race_ability=np.asarray(saved["overall_race_skill"]), race_team=np.asarray(saved["overall_team"]),
        race_form=np.asarray(saved["overall_day"]), race_car=np.asarray(saved["overall_car"])[..., mapping])
    fields["qualifying_base_location"] = sum(fields[k] for k in
        ("qualifying_ability", "qualifying_team", "qualifying_form", "qualifying_car"))
    fields["race_total_location"] = sum(fields[k] for k in
        ("race_ability", "race_team", "race_form", "race_car"))
    return scalar, fields, errors


@contextmanager
def diagnostic_archive(path, expected_hash, axes, names):
    """A numeric diagnostic reader; does not admit this file as a posterior."""
    if sha(path) != expected_hash:
        raise ValueError("Saved diagnostic archive changed")
    with Path(path).open("rb") as handle:
        with _mapped_numeric_archive(handle, names=names, axes=axes) as (saved, _):
            yield saved
    if sha(path) != expected_hash:
        raise ValueError("Saved diagnostic archive changed during analysis")


def diagnose():
    from f1rank.overall.recovery import generate, synthetic_design
    from f1rank.overall.fitting import arrays_digest
    from analysis.overall_rotation_profile import matched_truth_record, checked_registration

    folder = ROOT / "outputs/championship/development" / PROFILE
    summary = json.loads((folder / "summary.json").read_text())
    review = json.loads((folder / "review.json").read_text())
    registration_path = ROOT / summary["registration"]
    registration = checked_registration(registration_path)
    files = [*(ROOT / name for name in SOURCES), registration_path,
        folder / "summary.json", folder / "review.json",
        *(folder / f"{name}.json" for name in registration["studies"])]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, expected in hashes.items():
        committed = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError("Commit unchanged diagnostic inputs before execution")
    require_snapshot(summary["source"])
    if ([s["study"] for s in review["studies"]] != registration["studies"]
            or sha(folder / "summary.json") != review["summary_sha256"]
            or sha(registration_path) != review["registration_sha256"]):
        raise ValueError("The complete reviewed experiment is required")
    results = []
    for study in review["studies"]:
        name = study["study"]
        record_path = folder / f"{name}.json"
        if sha(record_path) != study["study_file_sha256"]:
            raise ValueError("Reviewed study changed")
        receipt = json.loads(record_path.read_text())["receipt"]
        require_snapshot(receipt["source"])
        spec = registration["synthetic"][name]
        blocks, catalog, truth = generate(spec["case"], spec["seed"])
        matched_truth_record(registration, name, blocks, truth)
        if (arrays_digest(blocks) != receipt["specification"]["arrays_key"]
                or catalog["field"] != receipt["catalog"]["field"]
                or catalog["field_entries"] != receipt["catalog"]["field_entries"]):
            raise ValueError("Reconstructed study differs from saved observations/catalogue")
        design = synthetic_design(spec["case"])
        q = design.qualifying.arrays()
        indices = catalog["field_entries"]
        attempts = []
        if len(study["attempts"]) != len(receipt["attempts"]):
            raise ValueError("The complete attempt inventory is required")
        for number, (reviewed, attempt) in enumerate(zip(study["attempts"], receipt["attempts"]), 1):
            if reviewed["attempt"] != number:
                raise ValueError("Attempt order changed")
            expected_name = receipt["posterior_file"] if attempt["diagnostics"]["converged"] else f"failed_{number}.npz"
            archive = ROOT / "outputs/championship/cache" / PROFILE / name / expected_name
            if str(archive.relative_to(ROOT)) != reviewed["archive"]:
                raise ValueError("Review points outside the declared attempt archive")
            axes = (attempt["chains"], attempt["samples"])
            with diagnostic_archive(archive, reviewed["archive_sha256"], axes, SCALARS + VECTORS) as saved:
                original = {p["parameter"]:p for p in attempt["diagnostics"]["parameters"]}
                reproduced = check(saved, attempt["diagnostics"]["divergences"])
                for row in reproduced["parameters"]:
                    reference = original[row["parameter"]]
                    if row["passed"] != reference["passed"]:
                        raise ValueError("Saved-site diagnostic verdict changed")
                    np.testing.assert_allclose([row[k] for k in METRICS], [reference[k] for k in METRICS], rtol=1e-12, atol=1e-12)
                scalar, fields, reconstruction = derived_quantities(saved, q, design.outcomes)
                pair_rows = []
                for i, driver_i in enumerate(catalog["field"]):
                    for j, driver_j in enumerate(catalog["field"][:i]):
                        contrasts = {key: value[..., indices[i]] - value[..., indices[j]] for key, value in fields.items()}
                        pair_rows.append(dict(driver_i=driver_i, driver_j=driver_j,
                            **describe(contrasts, attempt["diagnostics"]["divergences"])))
                result = dict(attempt=number, original_attempt_converged=attempt["diagnostics"]["converged"],
                    archive_sha256=reviewed["archive_sha256"], saved_site_diagnostics_reproduced=True,
                    reconstruction=reconstruction, scalar_report=describe(scalar, attempt["diagnostics"]["divergences"]),
                    current_driver_pairs=pair_rows)
            attempts.append(result)
        results.append(dict(study=name, original_state=study["state"], arrays_and_truths_reproduced=True, attempts=attempts))
        print(json.dumps(dict(study=name, attempts_diagnosed=len(attempts))), flush=True)
    require_snapshot(summary["source"])
    if any(sha(ROOT / name) != expected for name, expected in hashes.items()):
        raise ValueError("Diagnostic inputs changed during analysis")
    return dict(purpose="posthoc_saved_weekend_diagnosis", source_commit=source_commit,
        source_hashes=hashes, original_model_inputs_key=summary["source"]["key"], studies=results,
        all_registered_studies_and_attempts_included=True, refit_performed=False,
        original_failures_changed=False, scientific_evidence_verified=False, publication_allowed=False,
        interpretation="Descriptive chain and correlation evidence, including failed draws. It neither proves a failure cause nor supplies valid driver estimates, new acceptance rules, a preferred retry or permission to refit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    limits = scope_limits()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = diagnose()
    result.update(recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(), seconds=time.monotonic()-started,
        process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, limits=limits,
        runtime={name:importlib.metadata.version(name) for name in ("numpy", "scipy", "arviz-stats", "jax", "numpyro")})
    (destination / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
