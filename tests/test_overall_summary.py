"""Paired summary arithmetic and complete inventory; no fitted-driver evidence."""
import copy
import json

import numpy as np
import pytest

from analysis.overall_summary import OverallSummary, PROBES, MAX_BLOCK_REPLICATES
from f1rank.overall.simulation import MAX_EXPECTED_POINTS_MCSE, POINTS, uncertainty


IDENTITY = dict(fit_id="synthetic-fit", posterior_key="a" * 64,
                scenario_key="b" * 64, simulation_key="c" * 64)
FIELD = ["charlie", "alpha", "beta"]


def result_for(points, drivers=FIELD, races=2):
    low = 1 + (points[..., None, :] > points[..., :, None]).sum(axis=-1)
    high = low + (points[..., None, :] == points[..., :, None]).sum(axis=-1) - 1
    return dict(drivers=list(drivers), n_races=races, points=points,
        rank_low=low, rank_high=high, title_share=np.where(low == 1, 1 / (high - low + 1), 0.))


def fixture():
    rng = np.random.default_rng(67)
    points = rng.integers(5, 41, size=(2, 3, 5, 3)).astype(float)
    points[0, 0, 0] = [20, 20, 20]
    points[1, 1, 1] = [30, 30, 20]
    probes = np.stack([points - 2, points + 3, points - 1], axis=-1)
    probes[:, :, 1::2, :, 0] += 1
    return points, probes


def consume(points, probes, *, drivers=FIELD, draws=2, replicates=2):
    accumulator = OverallSummary(drivers=drivers, posterior_shape=points.shape[:2],
        replicates=points.shape[2], n_races=2, identity=IDENTITY)
    # Chains and draw blocks need not arrive in canonical order. Replicate
    # coverage must remain complete and consecutive within each draw.
    for chain in reversed(range(points.shape[0])):
        for draw in range(0, points.shape[1], draws):
            for rep in range(0, points.shape[2], replicates):
                block = points[chain, draw:draw + draws, rep:rep + replicates]
                accumulator.add(result_for(block, drivers),
                    probes[chain, draw:draw + draws, rep:rep + replicates], chain=chain,
                    draw_start=draw, replicate_start=rep, identity=IDENTITY)
    return accumulator, accumulator.finalize()


def test_streaming_moments_and_rank_quantiles_match_direct_nested_arrays():
    points, probes = fixture()
    accumulator, actual = consume(points, probes)
    direct = uncertainty(dict(points=points.reshape(-1, 5, 3), n_races=2))
    season = result_for(points)
    rows = {row["driver_id"]: row for row in actual["standings"]}
    for i, driver in enumerate(FIELD):
        row = rows[driver]
        for output, expected in (("mean", "expected_points"), ("posterior_sd", "posterior_sd"),
                                 ("mcse", "expected_points_mcse"), ("max_conditional_mcse", "max_conditional_mcse")):
            np.testing.assert_allclose(row["expected_points"][output], direct[expected][i], rtol=1e-13)
        assert row["expected_points"]["q05"] is None
        assert row["expected_points"]["q95"] is None
        np.testing.assert_allclose(row["p_title"], season["title_share"][..., i].mean())
        for name, data, q in (("q05", season["rank_low"], .05),
                              ("median", (season["rank_low"] + season["rank_high"]) / 2, .5),
                              ("q95", season["rank_high"], .95)):
            np.testing.assert_allclose(row["season_rank"][name], np.quantile(data[..., i], q))
    assert sum(row["p_title"] for row in actual["standings"]) == pytest.approx(1)
    assert actual["complete_draw_and_replicate_inventory"]
    assert not actual["expected_points_precision_passed"]
    assert not actual["scientific_evidence_verified"] and not actual["publication_allowed"]
    assert not actual["mcmc_precision_verified"] and not actual["rank_precision_established"]
    json.dumps(actual, allow_nan=False)
    with pytest.raises(ValueError, match="already finalized"):
        accumulator.finalize()


def test_pairwise_comparisons_and_explanations_use_paired_season_errors():
    points, probes = fixture()
    _, result = consume(points, probes)
    for pair in result["pairwise"]:
        i, j = [FIELD.index(pair[key]) for key in ("driver_a", "driver_b")]
        differences = ((points[..., i] - points[..., j]) / 2).reshape(-1, 5)
        means, variance = differences.mean(axis=1), differences.var(axis=1, ddof=1)
        actual = pair["points_difference"]
        np.testing.assert_allclose(actual["mean"], means.mean())
        np.testing.assert_allclose(actual["mcse"], np.sqrt(variance.sum() / (len(means) ** 2 * 5)))
        np.testing.assert_allclose(actual["max_conditional_mcse"], np.sqrt(variance / 5).max())
        assert pair["finite_mean_probability_greater"] == (means > 0).mean()
        assert pair["finite_mean_probability_equal"] == (means == 0).mean()
    for row in result["contributions"]:
        i = FIELD.index(row["driver_id"])
        for k, name in enumerate(PROBES):
            losses = ((points[..., i] - probes[..., i, k]) / 2).reshape(-1, 5)
            actual = row["losses"][name]
            np.testing.assert_allclose(actual["mean"], losses.mean())
            np.testing.assert_allclose(actual["max_conditional_mcse"], np.sqrt(losses.var(axis=1, ddof=1) / 5).max(), atol=1e-14)
        assert row["losses"]["total_race_pace"]["mean"] < 0
        assert row["losses"]["all"]["mean"] != sum(row["losses"][key]["mean"] for key in PROBES[:-1])


def test_perfectly_correlated_driver_outcomes_do_not_add_marginal_variances():
    y = np.broadcast_to(np.arange(4.) + 5, (2, 2, 4))
    points = np.stack([y, y + 1, y + 2], axis=-1)
    probes = np.repeat(points[..., None] - 2, len(PROBES), axis=-1)
    _, result = consume(points, probes)
    assert all(row["expected_points"]["mcse"] > 0 for row in result["standings"])
    assert all(pair["points_difference"]["mcse"] == 0 for pair in result["pairwise"])
    assert all(loss["mcse"] == 0 for row in result["contributions"] for loss in row["losses"].values())


def compare_nested(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_nested(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare_nested(a, b)
    elif type(left) is float:
        assert left == pytest.approx(right, abs=1e-13, rel=1e-13)
    else:
        assert left == right


def test_field_posterior_and_block_permutations_preserve_paired_summary():
    points, probes = fixture()
    _, first = consume(points, probes, draws=3, replicates=5)
    _, blocked = consume(points, probes, draws=1, replicates=1)
    compare_nested(first, blocked)
    field = [1, 2, 0]
    order = [5, 1, 4, 0, 3, 2]
    shuffled_points = points.reshape(6, 5, 3)[order].reshape(points.shape)[..., field]
    shuffled_probes = probes.reshape(6, 5, 3, 3)[order].reshape(probes.shape)[..., field, :]
    _, shuffled = consume(shuffled_points, shuffled_probes, drivers=[FIELD[i] for i in field])
    assert set(first.pop("field")) == set(shuffled.pop("field"))
    compare_nested(first, shuffled)


def test_sporting_ties_retain_whole_interval_and_share_title_mass():
    points = np.full((2, 2, 3, 3), 20.)
    probes = np.repeat(points[..., None], 3, axis=-1)
    _, result = consume(points, probes)
    for row in result["standings"]:
        assert row["season_rank"] == dict(q05=1., median=2., q95=3.)
        assert row["p_title"] == pytest.approx(1 / 3)
    assert all(pair["finite_mean_probability_equal"] == 1 for pair in result["pairwise"])
    assert all(pair["finite_mean_probability_greater"] == 0 for pair in result["pairwise"])
    assert not result["expected_points_precision_passed"]  # Zero sample variance is insufficient.


def test_fixed_sufficient_workload_unlocks_interval_without_claiming_acceptance():
    replicates = int(np.ceil((max(POINTS) / MAX_EXPECTED_POINTS_MCSE) ** 2))
    base = np.array([[[10., 10.], [11., 9.]], [[9., 11.], [12., 8.]]]) * 2
    field = ["alpha", "beta"]
    summary = OverallSummary(drivers=field, posterior_shape=(2, 2), replicates=replicates,
                             n_races=2, identity=IDENTITY)
    for chain in range(2):
        for first in range(0, replicates, MAX_BLOCK_REPLICATES):
            size = min(MAX_BLOCK_REPLICATES, replicates - first)
            points = np.broadcast_to(base[chain, :, None, :], (2, size, 2))
            probes = np.broadcast_to(points[..., None], points.shape + (len(PROBES),))
            summary.add(result_for(points, field), probes, chain=chain, draw_start=0,
                        replicate_start=first, identity=IDENTITY)
    actual = summary.finalize()
    assert actual["expected_points_precision_passed"]
    assert actual["pairwise_points_precision_passed"] and actual["explanation_precision_passed"]
    for row in actual["standings"]:
        i = field.index(row["driver_id"])
        np.testing.assert_allclose([row["expected_points"][key] for key in ("q05", "q95")],
                                   np.quantile(base[..., i] / 2, [.05, .95]))
    assert not actual["publication_allowed"] and not actual["rank_precision_established"]


@pytest.mark.parametrize("case", ["identity", "field", "races", "missing_probe", "probe_axes", "nonfinite",
    "points_bound", "probe_bound", "fractional_rank", "incomplete_tie", "overlap", "title_mass", "gap",
    "chain", "draw", "replicate", "oversized_block"])
def test_invalid_block_is_rejected_before_any_state_mutation(case):
    points, probes = fixture()
    summary = OverallSummary(drivers=FIELD, posterior_shape=(2, 3), replicates=5,
                             n_races=2, identity=IDENTITY)
    row, probe = result_for(points[0, :2, :2].copy()), probes[0, :2, :2].copy()
    kwargs = dict(chain=0, draw_start=0, replicate_start=0, identity=copy.deepcopy(IDENTITY))
    if case == "identity": kwargs["identity"]["posterior_key"] = "d" * 64
    if case == "field": row["drivers"] = FIELD[::-1]
    if case == "races": row["n_races"] = 3
    if case == "missing_probe": probe = None
    if case == "probe_axes": probe = probe[..., :2]
    if case == "nonfinite": row["points"][0, 0, 0] = np.nan
    if case == "points_bound": row["points"][0, 0, 0] = -1
    if case == "probe_bound": probe[0, 0, 0, 0] = 51
    if case == "fractional_rank": row["rank_low"] = row["rank_low"] + .5
    if case == "incomplete_tie": row["rank_high"][0, 0, 0] = 1
    if case == "overlap":
        row["rank_low"][0, 0] = [1, 2, 2]
        row["rank_high"][0, 0] = [2, 3, 3]
    if case == "title_mass": row["title_share"][0, 0, 0] = 1
    if case == "gap": kwargs["replicate_start"] = 1
    if case == "chain": kwargs["chain"] = True
    if case == "draw": kwargs["draw_start"] = 3
    if case == "replicate": kwargs["replicate_start"] = -1
    if case == "oversized_block": row["points"] = np.ones((1, MAX_BLOCK_REPLICATES + 1, 3))
    before = {k: v.copy() for k, v in vars(summary).items() if isinstance(v, np.ndarray)}
    with pytest.raises(ValueError):
        summary.add(row, probe, **kwargs)
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(summary, key), value)


def test_repeated_and_missing_replicates_cannot_finalize_a_partial_inventory():
    points, probes = fixture()
    summary = OverallSummary(drivers=FIELD, posterior_shape=(2, 3), replicates=5,
                             n_races=2, identity=IDENTITY)
    row, probe = result_for(points[0, :1, :2]), probes[0, :1, :2]
    summary.add(row, probe, chain=0, draw_start=0, replicate_start=0, identity=IDENTITY)
    with pytest.raises(ValueError, match="duplicate"):
        summary.add(row, probe, chain=0, draw_start=0, replicate_start=0, identity=IDENTITY)
    with pytest.raises(ValueError, match="complete replicate inventory"):
        summary.finalize()


def test_storage_does_not_grow_with_declared_replicate_count():
    common = dict(drivers=FIELD, posterior_shape=(2, 3), n_races=2, identity=IDENTITY)
    small = OverallSummary(**common, replicates=2)
    large = OverallSummary(**common, replicates=1000000)
    assert small.storage_bytes == large.storage_bytes


@pytest.mark.parametrize("case", ["reversed_points", "unequal_tied_points", "excess_field_points"])
def test_rank_and_points_cannot_contradict_the_common_sporting_scenario(case):
    summary = OverallSummary(drivers=FIELD, posterior_shape=(2, 2), replicates=2,
                             n_races=2, identity=IDENTITY)
    row = result_for(np.array([[[40., 30., 20.]]]))
    if case == "reversed_points":
        row["points"][0, 0] = [20., 30., 40.]
    elif case == "unequal_tied_points":
        row["rank_low"][0, 0] = [1, 1, 3]
        row["rank_high"][0, 0] = [2, 2, 3]
        row["title_share"][0, 0] = [.5, .5, 0]
    else:
        row = result_for(np.full((1, 1, 3), 50.))
    probes = np.repeat(row["points"][..., None], len(PROBES), axis=-1)
    with pytest.raises(ValueError, match="scoring allocation|contradict championship points"):
        summary.add(row, probes, chain=0, draw_start=0, replicate_start=0, identity=IDENTITY)
    assert summary._counts.sum() == 0


def test_equal_points_may_be_separated_by_sporting_countback():
    summary = OverallSummary(drivers=FIELD, posterior_shape=(2, 2), replicates=2,
                             n_races=2, identity=IDENTITY)
    row = result_for(np.full((2, 2, 3), 20.))
    row["rank_low"][:] = [1, 2, 3]
    row["rank_high"][:] = [1, 2, 3]
    row["title_share"][:] = [1, 0, 0]
    probes = np.repeat(row["points"][..., None], len(PROBES), axis=-1)
    for chain in range(2):
        summary.add(row, probes, chain=chain, draw_start=0, replicate_start=0, identity=IDENTITY)
    actual = summary.finalize()
    assert {r["driver_id"]: r["p_title"] for r in actual["standings"]} == dict(zip(FIELD, [1., 0., 0.]))
    assert all(pair["finite_mean_probability_equal"] == 1 for pair in actual["pairwise"])


def test_changed_definition_cannot_label_an_existing_summary(monkeypatch, tmp_path):
    from analysis import overall_summary as module
    definition = tmp_path / "definition.md"
    original = module.DEFINITION.read_text()
    definition.write_text(original)
    monkeypatch.setattr(module, "DEFINITION", definition)
    points, probes = fixture()
    summary = OverallSummary(drivers=FIELD, posterior_shape=(2, 3), replicates=5,
                             n_races=2, identity=IDENTITY)
    for chain in range(2):
        summary.add(result_for(points[chain]), probes[chain], chain=chain,
                    draw_start=0, replicate_start=0, identity=IDENTITY)
    definition.write_text(original + "\nChanged scenario interpretation.\n")
    with pytest.raises(ValueError, match="changed during accumulation"):
        summary.finalize()
    definition.write_text(original)
    assert summary.finalize()["complete_draw_and_replicate_inventory"]


@pytest.mark.parametrize("override", [dict(drivers=["a", "a"]), dict(posterior_shape=(1, 3)),
    dict(replicates=True), dict(n_races=0), dict(identity={}), dict(identity={**IDENTITY, "scenario_key":"wrong"})])
def test_invalid_declared_inventory_rejected(override):
    kwargs = dict(drivers=FIELD, posterior_shape=(2, 3), replicates=5, n_races=2, identity=IDENTITY)
    with pytest.raises(ValueError):
        OverallSummary(**{**kwargs, **override})
