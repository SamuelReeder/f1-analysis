"""Bounded forecast work must keep all samples and their original joint identity."""
import numpy as np
import pytest

from analysis import overall_forecast_states as states
from analysis.overall_forecast_catalog import build_catalog
from f1rank.overall.fitting import arrays_digest
from test_overall_forecast_states import fixture, future_path, posterior


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_terminal_history_matches_full_reconstruction_without_retaining_it(dtype):
    _, _, design, _ = fixture()
    p = {k:v.astype(dtype) for k, v in posterior(design).items()}
    q = design.qualifying.arrays()
    full = states.reconstruct_history(design, p)
    drivers, cars = states._terminal_history(design, p, q)
    for row in design.qualifying.entries.groupby("driver_id").last().itertuples():
        event, season, qualifying, race = drivers[row.Index]
        assert (event, season) == (row.event_idx, row.season)
        np.testing.assert_allclose(qualifying, full["qualifying_walk"][..., row.entry_idx], rtol=2e-7, atol=1e-7)
        np.testing.assert_allclose(race, full["race_residual"][..., row.entry_idx], rtol=2e-7, atol=1e-7)
    for row in design.qualifying.cars.groupby("team").last().itertuples():
        season, value = cars[row.Index]
        assert season == row.season
        np.testing.assert_allclose(value, full["car_persistent"][..., row.car_idx], rtol=2e-7, atol=1e-7)


def test_chunked_hash_matches_original_for_strided_and_mapped_arrays(tmp_path):
    _, _, design, _ = fixture()
    p = posterior(design, (2, 7))
    for name in list(p):
        value = p[name][:, ::-1]
        if name == "skill_z":
            path = tmp_path / "latent.npy"
            np.save(path, value)
            value = np.load(path, mmap_mode="r")
        p[name] = value
    original = arrays_digest({"parameters": p})
    for batch in (1, 3, 10):
        assert states._checked_parameter_digest(p, batch) == original


@pytest.mark.parametrize("student", [False, True])
def test_noise_batches_are_exact_slices_of_the_original_stream(student):
    full = states._Noise(41, (4, 11)).draw("example", ["team", "event"], student=student)
    parts = [states._Noise(41, (4, 11), slice(lo, lo + 3)).draw("example", ["team", "event"], student=student)
             for lo in range(0, 11, 3)]
    np.testing.assert_array_equal(full, np.concatenate(parts, axis=1))


def test_batch_sizes_and_final_single_draw_keep_every_joint_state(monkeypatch):
    _, _, design, snap = fixture()
    p, path = posterior(design, (2, 7)), future_path(snap)
    key = build_catalog(design, path[0])["training_arrays_key"]
    args = dict(seed=41, training_arrays_key=key)
    # Forecasting may not fall back to materializing all historical states.
    def forbidden(*args):
        raise AssertionError("Full historical state materialization was used")
    monkeypatch.setattr(states, "_history", forbidden)
    unbatched = states.forecast_states(design, path, p, draw_batch_size=20, **args)
    for size in (1, 2, 3, 7):
        batched = states.forecast_states(design, path, p, draw_batch_size=size, **args)
        assert batched["consumed_posterior_key"] == unbatched["consumed_posterior_key"]
        for left, right in zip(unbatched["events"], batched["events"]):
            for name, value in left["states"].items():
                np.testing.assert_allclose(value, right["states"][name], atol=1e-12, rtol=1e-12)
    batches = list(states.iter_forecast_states(design, path, p, draw_batch_size=3, **args))
    assert [b["draw_range"] for b in batches] == [[0, 3], [3, 6], [6, 7]]
    assert all(b["posterior_shape"] == [2, 7] and not b["publication_allowed"] for b in batches)


def test_corrupt_final_draw_is_rejected_before_first_batch_is_yielded():
    _, _, design, snap = fixture()
    p = posterior(design, (2, 7))
    p["skill_z"][-1, -1, -1] = np.nan
    key = build_catalog(design, snap)["training_arrays_key"]
    batches = states.iter_forecast_states(design, [snap], p, seed=41,
        training_arrays_key=key, draw_batch_size=2)
    with pytest.raises(ValueError, match="Nonfinite"):
        next(batches)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, None])
def test_invalid_batch_size_fails(bad):
    _, _, design, snap = fixture()
    with pytest.raises(ValueError, match="batch size"):
        states.forecast_states(design, [snap], posterior(design), seed=41,
            training_arrays_key="unused", draw_batch_size=bad)
