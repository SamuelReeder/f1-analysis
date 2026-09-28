"""Racing modules on small hand-made inputs (no downloaded data needed)."""

import jax.numpy as jnp
import numpy as np
import pandas as pd

from f1rank import battles, benchmark, championship


def test_plackett_luce_matches_direct_computation():
    s = jnp.array([[2.0, 1.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0]])
    mask = jnp.array([[1, 1, 1, 0], [1, 1, 0, 0]], bool)

    def direct(v):
        return sum(v[k] - np.log(np.exp(v[k:]).sum()) for k in range(len(v)))

    np.testing.assert_allclose(np.asarray(benchmark.plackett_luce(s, mask)),
                               [direct(np.array([2, 1, 0.5])), direct(np.array([1.0, 0.0]))], rtol=1e-5)


def laps(rows):
    cols = ["driver_id", "team", "lap_number", "time", "pit_in_time", "pit_out_time", "track_status",
            "tyre_life", "compound", "speed_st"]
    return pd.DataFrame(rows, columns=cols)


def test_a_pass_is_detected_and_ends_the_episode():
    # b leads a by 0.5 s on laps 2-3, a is ahead on lap 4
    rows = []
    for lap, (tb, ta) in enumerate([(100, 100.5), (190, 190.5), (280, 280.4), (370.2, 370.0)], start=1):
        rows += [("b", "B", lap, tb, np.nan, np.nan, "1", lap, "SOFT", 300),
                 ("a", "A", lap, ta, np.nan, np.nan, "1", lap, "SOFT", 305)]
    E = battles.race_battles(laps(rows), {1}, pd.Series({"a": 0.3, "b": 0.0}))
    assert list(E.passed) == [False, True]
    assert E.episode.nunique() == 1 and (E.attacker == "a").all()
    np.testing.assert_allclose(E.pace_diff, 0.3)


def test_teammates_and_pit_laps_are_not_battles():
    rows = []
    for lap, (tb, ta) in enumerate([(100, 100.5), (190, 190.5), (280, 280.4)], start=1):
        rows += [("b", "A", lap, tb, np.nan, np.nan, "1", lap, "SOFT", 300),
                 ("a", "A", lap, ta, np.nan, np.nan, "1", lap, "SOFT", 305)]
    assert battles.race_battles(laps(rows), {1}, pd.Series(dtype=float)).empty
    rows = [r if not (r[0] == "a" and r[2] == 3) else (*r[:4], 279.0, *r[5:]) for r in rows]
    rows = [(r[0], "B" if r[0] == "b" else "A", *r[2:]) for r in rows]
    E = battles.race_battles(laps(rows), {1}, pd.Series(dtype=float))
    assert len(E) == 0  # the lap before a pit stop has no outcome


def test_simulated_points_are_consistent():
    rng = np.random.default_rng(0)
    Q = rng.normal(0, 0.2, (50, 20))
    pts = championship.simulate(Q, np.full(50, 0.05), np.full(50, 0.1), np.full(50, 5.0), np.full(50, 1.0),
                                np.zeros(20), 3, rng)
    np.testing.assert_allclose(pts.sum(1), 3 * championship.POINTS.sum())
    # the fastest driver scores more on average than the slowest
    order = Q.mean(0).argsort()
    assert pts[:, order[-1]].mean() > pts[:, order[0]].mean()
