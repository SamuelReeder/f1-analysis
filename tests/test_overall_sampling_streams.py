"""Predeclared independent SBC sampling streams; original profiles stay fixed."""
import json

import pytest

from f1rank.overall import fitting
from test_overall_fitting import runner, sampled


def recovery_registration(path):
    reg = json.loads(path.read_text())
    reg.update(purpose="recovery_calibration", studies=["first", "second"],
        synthetic={"first":dict(case="connected", seed=1), "second":dict(case="connected", seed=2)},
        sampling_seeds={"first":[10, 11, 12, 13], "second":[20, 21, 22, 23]})
    path.write_text(json.dumps(reg))
    return reg


def test_recovery_fits_use_every_predeclared_sampling_seed(runner):
    blocks, path, directory = runner
    reg = recovery_registration(path)
    seen, receipts = {}, {}
    for study in reg["studies"]:
        seen[study] = []
        def sample(blocks, settings):
            seen[study].append(settings.copy())
            return sampled(settings, len(seen[study]) == len(fitting.RETRY))
        _, receipts[study] = fitting.run(blocks, {}, directory / study, path,
            study=study, synthetic=True, sample=sample)
    assert {study:[s["seed"] for s in attempts] for study, attempts in seen.items()} == reg["sampling_seeds"]
    for study, receipt in receipts.items():
        assert receipt["settings"]["seed"] == reg["sampling_seeds"][study][0]
        assert [attempt["seed"] for attempt in receipt["attempts"]] == reg["sampling_seeds"][study]
