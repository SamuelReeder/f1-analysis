"""Post hoc design/scale diagnosis of the original numerical-profile failures.

Reproduce the registered synthetic arrays and truths before inspecting known
generating scales and saved control-parameter correlations. No MCMC, settings
search, parameter estimate or acceptance decision is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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


def local_geometry(matrix, scales, previous, rho, sigma, nu):
    """Conditional control-block curvature at zero lap residual, not a Hessian fit.

    Student-t location curvature at zero is (nu + 1) / (nu * scale**2).
    This is a local geometry diagnostic, conditional on generating hyperparameters
    and other states. It is not the full posterior's condition number.
    """
    whitened = matrix * scales
    ar = whitened - rho * previous[:, None] * np.vstack([np.zeros(whitened.shape[1]), whitened[:-1]])
    noise = np.where(previous, sigma, sigma / np.sqrt(1 - rho**2))
    local = np.sqrt((nu + 1) / nu) * ar / noise[:, None]
    curvature = np.eye(matrix.shape[1]) + local.T @ local
    eigenvalues = np.linalg.eigvalsh(curvature)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise ValueError("Invalid curvature diagnostic; do not replace it with a favorable value")
    active = np.any(matrix != 0, axis=0)
    singular = np.linalg.svd(whitened[:, active], compute_uv=False)
    return dict(active_columns=int(active.sum()), zero_columns=int((~active).sum()),
                active_design_rank=int(np.linalg.matrix_rank(whitened[:, active])),
                active_design_singular_values=singular.tolist(),
                conditional_zero_residual_curvature_eigenvalues=eigenvalues.tolist(),
                conditional_zero_residual_curvature_ratio=float(eigenvalues[-1] / eigenvalues[0]))


def saved_controls(archive, names):
    columns = []
    for name in names:
        site, coordinate = name.rstrip("]").split("[")
        index = tuple(int(x) for x in coordinate.split(","))
        columns.append(np.asarray(archive[site])[(slice(None), slice(None), *index)])
    return np.stack(columns, axis=-1)


def diagnose(profile_id):
    import jax
    import jax.numpy as jnp
    from numpyro import handlers

    from f1rank.overall.fitting import arrays_digest
    from f1rank.overall.model import model
    from f1rank.overall.recovery import TRUTH_SITES, generate, synthetic_design

    directory = ROOT / "outputs/championship/development" / profile_id
    summary = json.loads((directory / "summary.json").read_text())
    registration = json.loads((ROOT / summary["registration"]).read_text())
    previous_diagnosis = ROOT / "outputs/championship/development/overall-profile-diagnosis-v1/results.json"
    prior_checks = json.loads(previous_diagnosis.read_text())
    archive_hashes = {(a["study"], a["attempt"]): a["archive_sha256"] for a in prior_checks["attempts"]}
    # These sources define the generative distributions and the inherited lap
    # controls. The weather correction may differ, but regenerated arrays below
    # must match the original registration exactly, including their dtypes.
    required_sources = ("f1rank/overall/model.py", "f1rank/overall/qualifying.py",
                        "f1rank/overall/recovery.py", "f1rank/model.py", "f1rank/race_total_model.py")
    for name in required_sources:
        if digest(ROOT / name) != summary["source"]["inputs"][name]:
            raise ValueError(f"Original model/generator source changed: {name}")
    results = []
    for study in registration["studies"]:
        spec = registration["synthetic"][study]
        record = json.loads((directory / f"{study}.json").read_text())
        receipt = record["receipt"]
        if receipt["state"] not in ("converged", "failed_convergence"):
            raise ValueError("Profile must be terminal")
        if digest(ROOT / summary["registration"]) != receipt["specification"]["registration_sha256"]:
            raise ValueError("Registered profile changed")
        if spec["case"] not in ("connected", "confounded", "sparse_rookie"):
            raise ValueError("This post hoc diagnosis is restricted to the original profile inventory")
        blocks, catalog, truth = generate(spec["case"], spec["seed"])
        original_truth = json.loads((directory / f"{study}.truth.json").read_text())
        if arrays_digest(blocks) != receipt["specification"]["arrays_key"]:
            raise ValueError("Regenerated observation arrays differ from the original profile")
        if any(not np.array_equal(truth[k], original_truth["truth"][k]) for k in TRUTH_SITES):
            raise ValueError("Registered synthetic truth did not reproduce")
        design = synthetic_design(spec["case"])
        initial = (design.qualifying.arrays(), design.laps, design.outcomes)
        as_jax = lambda b: {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in b.items()}
        trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(spec["seed"]))).get_trace(
            *[as_jax(b) for b in initial], generate=True)
        if any(not np.array_equal(np.asarray(trace[k]["value"]), truth[k]) for k in TRUTH_SITES):
            raise ValueError("Generating trace disagrees with the registered truth")
        scale_names = ("overall_sd_lap", "overall_lap_nu", "overall_lap_rho", "sigma0", "sigma_tau",
                       "sd_level", "sd_driver_form", "overall_sd_level", "overall_sd_day", "overall_sd_car_day")
        scales = {k: float(trace[k]["value"]) for k in scale_names}
        geometries, matrices = [], []
        laps = blocks["laps"]
        for race in range(laps["n_race"]):
            matrix, names, prior_scales, mask = control_design(laps, race)
            geometry = local_geometry(matrix, prior_scales, np.asarray(laps["prev"])[mask],
                                      scales["overall_lap_rho"], scales["overall_sd_lap"], scales["overall_lap_nu"])
            geometry.update(race_index=race, controls=names, fixed_prior_scales=prior_scales.tolist())
            geometries.append(geometry)
            matrices.append(names)
        attempts = []
        for number, attempt in enumerate(receipt["attempts"], 1):
            passed = attempt["diagnostics"]["converged"]
            archive = ROOT / "outputs/championship/cache" / profile_id / study / (
                receipt["posterior_file"] if passed else f"failed_{number}.npz")
            sha = digest(archive)
            if sha != archive_hashes[(study, number)]:
                raise ValueError("Saved draws differ from the earlier diagnosis")
            correlations = []
            with np.load(archive, allow_pickle=False) as saved:
                for race, names in enumerate(matrices):
                    values = saved_controls(saved, names)
                    if values.shape[:2] != (attempt["chains"], attempt["samples"]):
                        raise ValueError("Saved control draws have different axes")
                    matrices_by_chain = []
                    for chain in values:
                        correlation = np.corrcoef(chain, rowvar=False)
                        if not np.isfinite(correlation).all():
                            raise ValueError("Nonfinite saved-control correlation")
                        matrices_by_chain.append(correlation.tolist())
                    correlations.append(dict(race_index=race, controls=names,
                                             correlations_by_chain=matrices_by_chain))
            attempts.append(dict(attempt=number, original_converged=passed, archive_sha256=sha,
                                  control_correlations=correlations))
        results.append(dict(study=study, arrays_key=arrays_digest(blocks), arrays_and_truth_reproduced=True,
            original_result_sha256=digest(directory / f"{study}.json"),
            generating_scales=scales, local_geometry=geometries, attempts=attempts))
        jax.clear_caches()
    return dict(purpose="posthoc_numerical_geometry", profile_id=profile_id,
        profile_summary_sha256=digest(directory / "summary.json"),
        analysis_sha256=digest(__file__), verified_original_sources={s:digest(ROOT / s) for s in required_sources},
        earlier_diagnosis_sha256=digest(previous_diagnosis), studies=results,
        refit_performed=False, publication_allowed=False, original_failures_changed=False,
        interpretation="Generating scales and conditional control-block curvature describe the known synthetic experiment, not real drivers. Saved correlations are reported for every attempt and chain; failed draws are not valid posterior estimates. This post hoc evidence does not isolate causes, revise acceptance gates, authorize different sampler settings, or replace the complete recovery requirement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="overall-computation-profile-v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve previous evidence; use a new named analysis")
    result = diagnose(args.profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Saved geometry diagnosis; no fit or acceptance change")
