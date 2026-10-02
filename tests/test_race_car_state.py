"""The pre-registered within-season car state (docs/race_car_state.md)."""
import numpy as np
import pandas as pd

from f1rank import race_car_state_model as model


def laps():
    rows = []
    for event in ("2025-01", "2025-02", "2025-03", "2026-01", "2026-02"):
        for i, driver in enumerate(("a", "b", "c", "d")):
            for lap in (2, 3):
                rows.append(dict(event_id=event, season=int(event[:4]), driver_id=driver,
                                 team="x" if i < 2 else "y", stint_key=driver + "|1", lap_number=lap,
                                 tyre_life=lap, compound="HARD", y=0., close=0., near=0., unpressured=0.))
    return pd.DataFrame(rows)


def test_car_walk_restarts_each_season_and_is_centred_per_race():
    import jax
    import numpyro.handlers as handlers
    d, catalog = model.design(laps())
    z = np.arange(1, d["n_cr"] + 1, dtype=float)  # step z[i] enters at car-race i
    trace = handlers.trace(handlers.substitute(
        handlers.seed(model.model, jax.random.PRNGKey(0)),
        data={"sd_drift": .1, "drift_z": z[np.asarray(d["cr_order"])]})).get_trace(d)
    drift = np.asarray(trace["drift"]["value"])
    assert catalog["car_races"][:4] == ["2025-01|x", "2025-01|y", "2025-02|x", "2025-02|y"]
    # x walks 0, .3, .3+.5 and y 0, .4, .4+.6 in 2025; both restart in 2026 (0, .9 and 0, 1.0).
    # Centring each race leaves only the difference between teams.
    expected = [0, 0, -.05, .05, -.1, .1, 0, 0, -.05, .05]
    np.testing.assert_allclose(drift, expected, atol=1e-6)
    for event in ("2025-01", "2026-01"):  # first race of each season
        mask = np.array([n.startswith(event) for n in catalog["car_races"]])
        np.testing.assert_allclose(drift[mask], 0, atol=1e-6)


def test_prediction_uses_last_state_and_a_walk_shared_by_teammates():
    n = 4000
    catalog = dict(drivers=[], driver_seasons=[], cars=["2026|x"],
                   car_races=["2026-01|x", "2026-02|x"])
    post = dict(package=np.full((n, 1), .5), drift=np.tile([[0., .2]], (n, 1)),
                sd_drift=np.full(n, .1), sd_day=np.zeros(n), sd_car_day=np.zeros(n))
    meta = dict(diagnostics={"n_draws": n}, catalog=catalog)
    entries = pd.DataFrame(dict(event_id=["2026-05", "2026-05", "2026-08"], driver_id=["a", "b", "a"],
                                team=["x"] * 3))
    draws = model.predict(post, meta, entries, {"2026-05": 1, "2026-08": 4})
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
    assert abs(draws[:, 0].mean() - .7) < .01
    assert .09 < draws[:, 0].std() < .11
    assert .18 < draws[:, 2].std() < .22  # sqrt(4) steps ahead
