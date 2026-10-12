"""Reject mismatched or incomplete diagnostic companions; no scientific fit."""
import json

import numpy as np
import pytest

from analysis.overall_simulation_moments import FILENAME, load_checked, write_once
from f1rank.artifacts import digest
from test_overall_simulation_controller import case, execute


@pytest.mark.parametrize("change", ["failed_receipt", "missing_block", "receipt_flag", "summary_bytes",
    "archive_bytes", "simulation_key", "expected_archive", "metadata_field", "metadata_units", "metadata_identity"])
def test_changed_or_failed_companion_is_rejected(case, change):
    result, receipt = execute(case)
    simulation_key = receipt["simulation_key"]
    archive_key = result["diagnostic_moments"]["sha256"]
    if change == "failed_receipt": receipt["state"] = "failed_exception"
    elif change == "missing_block": receipt["completed_blocks"] -= 1
    elif change == "receipt_flag": receipt["publication_allowed"] = True
    elif change == "summary_bytes":
        with (case.output/"summary.json").open("a") as handle: handle.write("\n")
    elif change == "archive_bytes":
        with (case.output/FILENAME).open("ab") as handle: handle.write(b"changed")
    elif change == "simulation_key": simulation_key = "0"*64
    elif change == "expected_archive": archive_key = "0"*64
    else:
        if change == "metadata_field": result["diagnostic_moments"]["field"] = ["changed"]
        elif change == "metadata_units": result["diagnostic_moments"]["units"] = "season_points"
        else: result["diagnostic_moments"]["identity"]["scenario_key"] = "0"*64
        (case.output/"summary.json").write_text(json.dumps(result))
        receipt["summary_sha256"] = digest(case.output/"summary.json")
    (case.output/"receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        load_checked(case.output, expected_simulation_key=simulation_key, expected_archive_sha256=archive_key)


@pytest.mark.parametrize("change", ["missing_member", "extra_member", "bad_axis", "nonfinite", "incomplete_counts",
    "negative_variance", "asymmetric_covariance", "indefinite_covariance"])
def test_rehashed_malformed_arrays_are_still_rejected(case, change):
    result, receipt = execute(case)
    path = case.output/FILENAME
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name:saved[name] for name in saved.files}
    if change == "missing_member": del arrays["points_mean"]
    elif change == "extra_member": arrays["unregistered"] = np.zeros(1)
    elif change == "bad_axis": arrays["points_mean"] = arrays["points_mean"][..., :1]
    elif change == "nonfinite": arrays["points_mean"][0,0,0] = np.nan
    elif change == "incomplete_counts": arrays["replicate_counts"][0,0] -= 1
    elif change == "negative_variance": arrays["probe_loss_sample_variance"][0,0,0,0] = -1
    elif change == "asymmetric_covariance": arrays["points_sample_covariance"][0,0,0,1] += 1
    else:
        cov = arrays["points_sample_covariance"]
        cov[0,0] = np.eye(cov.shape[-1])
        cov[0,0,0,1] = cov[0,0,1,0] = 2
    np.savez_compressed(path, **arrays)
    meta = result["diagnostic_moments"]
    meta["sha256"] = digest(path)
    meta["array_schema"] = {name:dict(shape=list(a.shape), dtype=a.dtype.str) for name,a in arrays.items()}
    receipt["diagnostic_moments"] = meta
    (case.output/"summary.json").write_text(json.dumps(result))
    receipt["summary_sha256"] = digest(case.output/"summary.json")
    (case.output/"receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        load_checked(case.output, expected_simulation_key=receipt["simulation_key"], expected_archive_sha256=meta["sha256"])


def test_archive_writer_preserves_existing_file_and_cleans_temporary_output(case):
    result, receipt = execute(case)
    meta, arrays = load_checked(case.output, expected_simulation_key=receipt["simulation_key"],
        expected_archive_sha256=result["diagnostic_moments"]["sha256"])
    before = digest(case.output/FILENAME)
    with pytest.raises(FileExistsError):
        write_once(case.output, meta, arrays)
    assert digest(case.output/FILENAME) == before
    assert not list(case.output.glob("*.tmp"))
