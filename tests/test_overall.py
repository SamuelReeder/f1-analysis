"""Successor Overall invariants, all using synthetic data and known answers."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from f1rank.overall.scoring import classify, score_grand_prix
from f1rank.overall.simulation import (JointDraws, Scenario, championship_countback,
                                      qualifying_grid, simulate, uncertainty)


def draws(n=6, nd=12):
    return JointDraws(tuple(f"d{i}" for i in range(n)), np.zeros((nd, n)), np.zeros((nd, n)),
        *[np.full(nd, value) for value in (.2, .3, 1., .5, .5, 4.)],
        np.ones((nd, 1)), np.ones((nd, 1)), np.zeros((nd, 1)))


def tables():
    events = ["2021-22", "2022-01", "2022-02", "2026-01"]
    drivers = [f"d{i}" for i in range(6)]
    ent, quali, races, clean = [], [], [], []
    for event in events:
        for i, driver in enumerate(drivers):
            # A team swap across the era and a missing qualifying entrant.
            team = f"t{(i // 2 + (event >= '2022-01' and i == 0)) % 3}"
            row = dict(event_id=event, driver_id=driver, team=team, constructor_id=team,
                       constructor_name=team, quali_position=i + 1)
            if not (event == "2022-02" and i == 5):
                ent.append(row)
                for segment in ("Q1", "Q2", "Q3"):
                    quali.append(dict(event_id=event, driver_id=driver, segment=segment, time_s=90 + i / 10))
            races.append(dict(**{k: v for k, v in row.items() if k != "quali_position"},
                              grid=i + 1, position=i + 1, status="Finished", laps=60))
            for lap in (2, 3, 4):
                clean.append(dict(event_id=event, driver_id=driver, team=team, season=int(event[:4]),
                    compound="HARD", stint_key=driver + "|1", lap_number=lap, tyre_life=lap,
                    y=i / 10, close=0., near=0., unpressured=1.))
    t = dict(events=pd.DataFrame(dict(event_id=events, season=[int(e[:4]) for e in events],
                                    date=[e[:4] + "-01-01" for e in events], circuit_id=["track"] * len(events))),
             drivers=pd.DataFrame(dict(driver_id=drivers, dob=["2000-01-01"] * 6, debut_season=[2021] * 6)),
             entries=pd.DataFrame(ent), quali_times=pd.DataFrame(quali), race=pd.DataFrame(races),
             race_laps=pd.DataFrame(clean), race_weather=pd.DataFrame(dict(event_id=events, rainfall=False)),
             timeline=pd.DataFrame(columns=["event_id", "kind"]))
    return t, pd.DataFrame(clean)


def test_current_race_retiree_can_score_and_distance_precedes_line_order():
    positions, points = score_grand_prix([60, 54, 53, 59], [2, 1, 3, 4],
        scheduled_laps=60, leader_laps=60, consecutive_racing_laps=60)
    np.testing.assert_array_equal(positions, [1, 3, 0, 2])
    np.testing.assert_array_equal(points, [25, 15, 0, 18])
    # The FIA threshold rounds the required completed laps down.
    positions = classify([53, 47, 46], [1, 2, 3])
    np.testing.assert_array_equal(positions, [1, 2, 0])


def test_no_completed_laps_means_no_points_and_bad_inputs_fail():
    positions, points = score_grand_prix(np.zeros(3), np.arange(1, 4),
        scheduled_laps=60, leader_laps=0, consecutive_racing_laps=0)
    assert not positions.any() and not points.any()
    with pytest.raises(ValueError):
        classify(np.array([60, 54.5]), np.array([1, 2]))
    with pytest.raises(ValueError):
        classify(np.array([60, 54]), np.array([1, 3]))


def test_countback_uses_wins_then_seconds_then_qualifying_then_shared_tie():
    finishes = np.array([[1, 2, 3], [3, 2, 1]])
    qualifying = np.array([[1, 2, 3], [1, 2, 3]])
    result = championship_countback(np.array([40., 40., 40.]), finishes, qualifying)
    np.testing.assert_array_equal(result["rank_low"], [1, 3, 2])
    np.testing.assert_array_equal(result["title_share"], [1, 0, 0])
    # Identical race and qualifying counts: no invented sporting tiebreak.
    finishes = np.array([[1, 2], [2, 1]])
    result = championship_countback(np.array([43., 43.]), finishes, finishes)
    np.testing.assert_array_equal(result["rank_low"], [1, 1])
    np.testing.assert_array_equal(result["rank_high"], [2, 2])
    np.testing.assert_array_equal(result["title_share"], [.5, .5])


def test_simulation_is_exactly_invariant_to_driver_array_permutation():
    d = draws()
    d = replace(d, qualifying=d.qualifying + np.arange(6) / 10, race=d.race - np.arange(6) / 10)
    ix = [3, 0, 5, 1, 4, 2]
    shuffled = replace(d, drivers=tuple(d.drivers[i] for i in ix), qualifying=d.qualifying[:, ix], race=d.race[:, ix])
    args = dict(scenarios=[Scenario("ref", 0, 60, 60)], replicates=3, seed=17, advancing=(4, 2))
    a, b = simulate(d, **args), simulate(shuffled, **args)
    for name in ("points", "title_share", "rank_low", "rank_high"):
        np.testing.assert_array_equal(a[name][:, :, ix], b[name])


def test_faster_race_pace_cannot_lose_points_under_common_randomness():
    d = draws()
    race = d.race.copy()
    race[:, 0] += 1.
    args = dict(scenarios=[Scenario("ref", 0, 60, 60)], replicates=10, seed=14, advancing=(4, 2))
    a, b = simulate(d, **args), simulate(replace(d, race=race), **args)
    assert (b["points"][:, :, 0] >= a["points"][:, :, 0]).all()
    assert (b["points"][:, :, 0] > a["points"][:, :, 0]).any()


def test_qualifying_knockout_retains_earlier_eliminations():
    ability = np.array([[6., 5., 4., 3., 2., 1.]])
    grid = qualifying_grid(ability, np.zeros_like(ability), np.zeros(1), np.zeros(1),
                           np.ones(1) * 4, np.random.default_rng(0), (4, 2))
    np.testing.assert_array_equal(grid, [[1, 2, 3, 4, 5, 6]])


def test_actual_field_format_and_shared_ties_are_order_independent():
    from f1rank.overall.simulation import QUALIFYING_FIELDS
    assert QUALIFYING_FIELDS[22] == (16, 10)
    points = np.array([43., 43., 0.])
    finishes = np.array([[1, 2, 0], [2, 1, 0]])
    a = championship_countback(points, finishes, finishes)
    ix = [2, 1, 0]
    b = championship_countback(points[ix], finishes[:, ix], finishes[:, ix])
    for name in a:
        np.testing.assert_array_equal(a[name][ix], b[name])


def test_equal_driver_simulations_are_exchangeable_in_distribution():
    d = draws(nd=128)
    result = simulate(d, [Scenario("ref", 0, 60, 60)], replicates=128, seed=8, advancing=(4, 2))
    points = result["points"].reshape(-1, 6)
    expected = np.array([25, 18, 15, 12, 10, 8]).mean()
    # A broad fixed Monte Carlo check catches systematic slot/identity advantage.
    se = points.std(axis=0, ddof=1) / np.sqrt(len(points))
    assert (np.abs(points.mean(axis=0) - expected) < 6 * se).all()
    np.testing.assert_allclose(result["title_share"].sum(axis=-1), 1)


def test_uncertainty_separates_posterior_and_season_variation():
    fixed = np.broadcast_to(np.array([1., 3., 5., 7.])[:, None, None], (4, 8, 2))
    u = uncertainty(dict(points=fixed, n_races=1))
    assert (u["expected_points_mcse"] == 0).all()
    assert (u["posterior_sd"] > 0).all() and u["precision_passed"].all()
    noisy = np.broadcast_to(np.array([0., 25.])[None, :, None], (4, 2, 2))
    u = uncertainty(dict(points=noisy, n_races=1))
    assert not u["precision_passed"].any()
    assert np.isnan(u["expected_points_interval"]).all()
    assert (u["posterior_sd"] == 0).all()


def test_joint_draws_reject_mismatched_sources_and_unchecked_fit():
    d = draws()
    with pytest.raises(ValueError, match="share"):
        replace(d, race=d.race[:-1])
    with pytest.raises(ValueError, match="convergence"):
        JointDraws.from_posterior({}, [], [], checked={"converged": False})


def test_joint_draw_factory_preserves_cross_component_dependence():
    post = {"skill": np.arange(4 * 10 * 6).reshape(4, 10, 6).astype(float)}
    post["overall_race_skill"] = -post["skill"]
    for name, value in (("sd_driver_form", .2), ("overall_sd_day", .3), ("overall_shared_weekend", 1.),
                        ("sigma0", .5), ("sigma_tau", .5), ("nu", 4.)):
        post[name] = np.full((4, 10), value)
    for name in ("overall_grid", "overall_pace", "overall_hazard"):
        post[name] = np.ones((4, 10, 1)) * (.001 if name == "overall_hazard" else 1)
    d = JointDraws.from_posterior(post, [4, 1], ["a", "b"], checked={"converged": True})
    np.testing.assert_array_equal(d.qualifying + d.race, 0)
    np.testing.assert_array_equal(d.qualifying[:, 0], post["skill"][:, :, 4].reshape(-1))
    with pytest.raises(ValueError, match="valid"):
        JointDraws.from_posterior(post, [-1, 1], ["a", "b"], checked={"converged": True})
    with pytest.raises(ValueError, match="valid"):
        JointDraws.from_posterior(post, [1, 1], ["a", "b"], checked={"converged": True})


def test_missing_weather_is_not_clean_dry_race_evidence():
    from f1rank.overall.design import clean_training_laps
    t, _ = tables()
    t["race_weather"] = t["race_weather"].iloc[:0]
    with pytest.raises(ValueError, match="No eligible dry laps"):
        clean_training_laps(t)


def test_shared_catalog_preserves_missing_qualifier_transfer_and_era():
    from f1rank.overall.design import build
    t, clean = tables()
    t["race"].loc[(t["race"].event_id == "2022-01") & (t["race"].driver_id == "d5"), "status"] = "Engine"
    d = build("2022-02", tables=t, clean=clean)
    e = d.qualifying.entries.set_index(["event_id", "driver_id"])
    assert not e.loc[("2022-02", "d5"), "has_time"]
    assert set(d.catalog["field"]) == {f"d{i}" for i in range(6)}
    assert e.loc[("2021-22", "d0"), "team"] != e.loc[("2022-01", "d0"), "team"]
    assert e.loc[("2021-22", "d1"), "era_stint_idx"] != e.loc[("2022-01", "d1"), "era_stint_idx"]
    assert (d.qualifying.entries.iloc[d.laps["entry"]].event_id.to_numpy() == clean.loc[clean.event_id <= "2022-02", "event_id"].to_numpy()).all()


def test_future_rows_cannot_change_any_training_array():
    from f1rank.overall.design import build
    t, clean = tables()
    before = build("2022-02", tables=t, clean=clean)
    t["quali_times"].loc[t["quali_times"].event_id == "2026-01", "time_s"] = .01
    t["entries"].loc[t["entries"].event_id == "2026-01", "team"] = "future_team"
    clean.loc[clean.event_id == "2026-01", "y"] = 9000
    after = build("2022-02", tables=t, clean=clean)
    for a, b in ((before.qualifying.arrays(), after.qualifying.arrays()), (before.laps, after.laps), (before.outcomes, after.outcomes)):
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
    assert before.catalog == after.catalog


def test_joint_likelihood_has_shared_latents_and_finite_gradient():
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_median
    from f1rank.overall.design import build
    from f1rank.overall.model import model
    t, clean = tables()
    t["race"].loc[(t["race"].event_id == "2022-01") & (t["race"].driver_id == "d5"), "status"] = "Engine"
    d = build("2022-02", tables=t, clean=clean)
    convert = lambda x: {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in x.items()}
    args = (convert(d.qualifying.arrays()), convert(d.laps), convert(d.outcomes))
    info = initialize_model(jax.random.PRNGKey(0), model, model_args=args,
                            init_strategy=init_to_median(num_samples=3))
    assert np.isfinite(info.param_info.potential_energy)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(info.param_info.z_grad))
    trace = info.model_trace
    assert "skill" in trace and "overall_race_skill" in trace
    assert {"y", "overall_laps", "overall_finish_order", "overall_completion"} <= set(trace)
    # Perturbing the one shared qualifying state changes total race ability too.
    assert "overall_qualifying_link" in trace
    for site in ("skill", "overall_race_skill", "overall_team", "overall_day"):
        values = np.asarray(trace[site]["value"])
        for event in range(len(d.qualifying.events)):
            assert abs(values[d.qualifying.entries.event_idx == event].mean()) < 1e-5


def test_rank_diagnostics_reject_stuck_and_disagreeing_chains():
    from f1rank.overall.diagnostics import check
    rng = np.random.default_rng(10)
    good = rng.normal(size=(4, 1000, 2))
    assert check({"latent": good}, 0)["converged"]
    bad = good + np.arange(4)[:, None, None]
    assert not check({"latent": bad}, 0)["converged"]
    assert not check({"latent": np.zeros((4, 1000))}, 0)["converged"]
    assert not check({"latent": good}, 5)["converged"]


def test_nested_source_changes_invalidate_candidate(tmp_path):
    from f1rank.overall.provenance import snapshot, require_snapshot
    (tmp_path / "f1rank" / "overall").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "overall_candidate.md").write_text("contract")
    source = tmp_path / "f1rank" / "overall" / "model.py"
    source.write_text("first")
    original = snapshot(tmp_path)
    require_snapshot(original, tmp_path)
    source.write_text("second")
    with pytest.raises(ValueError, match="changed"):
        require_snapshot(original, tmp_path)
