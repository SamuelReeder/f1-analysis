"""Read-only model diagnosis using frozen fits; no fitting or publication."""
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from analysis.championship_preview import field_quality
from f1rank import artifacts, championship as c
from f1rank.design import build_design
from f1rank.fit import FITS, check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/championship_model_audit"
PLAN = ROOT / "docs/championship_model_audit_plan.md"
SEEDS = tuple(range(5))


def quantiles(x):
    return dict(zip(("q05", "median", "q95"), map(float, np.percentile(x, [5, 50, 95]))))


def center(x):
    return x - x.mean(axis=1, keepdims=True)


def paired_points(points, ids, target, n_races):
    """Season outcome probabilities and simulation error; not skill probabilities."""
    i = list(ids).index(target)
    rows = []
    for j, driver in enumerate(ids):
        delta = (points[:, i] - points[:, j]) / n_races
        rows.append({"driver_id": driver, "mean_points_per_race_difference": float(delta.mean()),
                     "monte_carlo_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                     "p_more_season_points": float(np.mean(delta > 0)),
                     "p_equal_season_points": float(np.mean(delta == 0))})
    return rows


def standings(points, ids, scenario, n_races):
    ranks = (-points).argsort(axis=1).argsort(axis=1) + 1
    expected = points.mean(axis=0) / n_races
    order = (-expected).argsort().argsort() + 1
    return [{"scenario": scenario, "driver_id": driver,
             "expected_points_order": int(order[i]), "points_per_race": float(expected[i]),
             "points_mean_monte_carlo_se": float(points[:, i].std(ddof=1) / n_races / np.sqrt(len(points))),
             "p_title_existing_tiebreak": float(np.mean(ranks[:, i] == 1)),
             "rank_median": float(np.median(ranks[:, i])),
             "rank_q05": float(np.percentile(ranks[:, i], 5)),
             "rank_q95": float(np.percentile(ranks[:, i], 95))}
            for i, driver in enumerate(ids)]


def main():
    if OUT.exists():
        raise FileExistsError("This diagnostic already has an outcome")
    started = time.monotonic()
    preview = ROOT / "outputs/analysis/championship_preview"
    parent = artifacts.require(preview, required_outputs=[preview / "standings.csv", preview / "summary.json"])
    artifacts.require(ROOT / "outputs/race_total", required_outputs=[ROOT / "outputs/race_total/pace.json"])
    reference = json.loads((preview / "summary.json").read_text())
    inputs = [ROOT / p for p in parent["inputs"]]
    inputs += [preview / name for name in ("manifest.json", "standings.csv", "summary.json")]
    inputs += [ROOT / "outputs/race_total" / name for name in ("pace.json", "manifest.json")]
    inputs += [ROOT / "outputs/ratings/current_drivers.csv", ROOT / "outputs/race/multi_drivers.csv",
               ROOT / "outputs/race/stage_a_pairs.csv",
               ROOT / "outputs/validation/sensitivity_current_drivers_in_team.csv", PLAN, Path(__file__)]
    inputs = sorted(set(inputs))
    before = {p: artifacts.digest(p) for p in inputs}
    design = build_design(2010)
    check(FITS / "main.npz", design)
    last = design.events.event_idx.max()
    rows = np.flatnonzero(design.entries.event_idx.to_numpy() == last)
    entries = design.entries.iloc[rows]
    ids = entries.driver_id.to_numpy()
    h = list(ids).index("hadjar")
    assert len(set(ids)) == len(ids)
    # Load only needed arrays, releasing full histories after taking current columns.
    with np.load(FITS / "main.npz") as z:
        def current(key):
            a = z[key]
            return a.reshape(-1, a.shape[-1])[:, rows].copy()
        skill, compat = current("skill"), current("compat")
        hyper = {key: z[key].reshape(-1).copy() for key in (
            "sd_driver_form", "sd_car_event", "sigma0", "sigma_tau", "nu", "sd_car_segment")}
        car_rows = design.cars.set_index(["event_idx", "team"]).loc[
            [(last, team) for team in entries.team], "car_idx"].to_numpy()
        cars = np.zeros_like(skill)
        for key in ("car", "car_track", "car_event"):
            a = z[key]
            cars += a.reshape(-1, a.shape[-1])[:, car_rows]
            del a
    drivers, draws = c.full_quality_draws("race_specific_pace")
    race_post = dict(np.load(FITS / "championship_preview/race_stage.npz"))
    rel = json.loads((ROOT / "outputs/reliability/summary.json").read_text())
    assert not rel["gate_driver_error_ranking"], "Rebuild audit retirement inputs if this changes"
    retire = np.full(len(ids), reference["p_retire_common"])
    n_races = reference["n_races_simulated"]
    assert reference["simulated_seasons"] == c.N_SEASONS

    def prepare(seed):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(skill), c.N_SEASONS, replace=True)
        form = hyper["sd_driver_form"][pick]
        noise = np.sqrt(hyper["sd_car_event"][pick] ** 2 + hyper["sigma0"][pick] ** 2)
        rp = rng.integers(len(race_post["b_driver"]), size=c.N_SEASONS)
        quality = field_quality(drivers, draws, ids, rng.integers(len(draws), size=c.N_SEASONS))
        extra = race_post["b_quality"][rp, 0][:, None] * quality
        return rng, pick, rp, form, noise, quality, extra

    rng, pick, rp, form, noise, quality, extra = prepare(0)
    Q = center((skill + compat)[pick])
    state = copy.deepcopy(rng.bit_generator.state)

    def run(q=Q, x=extra, n=noise):
        stream = np.random.default_rng()
        stream.bit_generator.state = copy.deepcopy(state)
        return c.simulate(q, form, n, race_post["b_driver"][rp], race_post["c_grid"][rp],
                          retire, n_races, stream, x)

    baseline = run()
    published = pd.read_csv(preview / "standings.csv").query("version == 'in_team'").set_index("driver_id")
    replay_error = float(np.max(np.abs(baseline.mean(0) / n_races - published.loc[ids].points_per_race)))
    np.testing.assert_allclose(baseline.mean(0) / n_races, published.loc[ids].points_per_race, rtol=0, atol=1e-12)
    only_h_compat = Q.copy()
    only_h_compat[:, h] -= center(compat[pick])[:, h]
    only_h_quali = Q.copy()
    only_h_quali[:, h] = 0
    only_h_race = extra.copy()
    only_h_race[:, h] = 0
    # Posterior-mean ability probe retains coefficient and race-outcome uncertainty.
    mean_q = np.broadcast_to(center(skill + compat).mean(0), Q.shape)
    all_quality = field_quality(drivers, draws, ids, np.arange(len(draws)))
    mean_extra = race_post["b_quality"][rp, 0][:, None] * all_quality.mean(0)
    scenarios = {
        "original": baseline,
        "portable_all": run(q=center(skill[pick])),
        "no_extra_race_pace": run(x=np.zeros_like(extra)),
        "field_average_qualifying_all": run(q=np.zeros_like(Q)),
        "posterior_mean_ability": run(q=mean_q, x=mean_extra),
        "no_independent_car_weekend_noise": run(n=hyper["sigma0"][pick]),
        "hadjar_average_team_effect": run(q=center(only_h_compat)),
        "hadjar_average_qualifying": run(q=only_h_quali),
        "hadjar_average_extra_race_pace": run(x=only_h_race),
    }
    scenario_rows = [row for name, pts in scenarios.items() for row in standings(pts, ids, name, n_races)]
    seed_rows = []
    for seed in SEEDS:
        stream, pi, ri, fs, ns, _, xs = prepare(seed)
        pts = c.simulate(center((skill + compat)[pi]), fs, ns, race_post["b_driver"][ri],
                         race_post["c_grid"][ri], retire, n_races, stream, xs)
        seed_rows += [{"seed": seed, **row} for row in standings(pts, ids, "original", n_races)]

    components = []
    quali_ranks = (-center(skill + compat).mean(0)).argsort().argsort() + 1
    portable_ranks = (-center(skill).mean(0)).argsort().argsort() + 1
    residual_ranks = (-all_quality.mean(0)).argsort().argsort() + 1
    published_race = json.loads((ROOT / "outputs/race_total/pace.json").read_text())
    total_rows = sorted(published_race["drivers"], key=lambda d: -d["pace"]["median"])
    total_by_id = {r["id"]: {"current_total_race_order": i + 1,
                              "current_total_race_pct_median": r["pace"]["median"],
                              "current_total_race_n_races": r["races"]} for i, r in enumerate(total_rows)}
    coverage = pd.read_csv(ROOT / "outputs/race/multi_drivers.csv").set_index("driver_id")
    for i, driver in enumerate(ids):
        row = {"driver_id": driver, "qualifying_mean_order": int(quali_ranks[i]),
               "portable_qualifying_mean_order": int(portable_ranks[i]),
               "career_residual_mean_order_current_field": int(residual_ranks[i]),
               "corr_skill_team_effect": float(np.corrcoef(skill[:, i], compat[:, i])[0, 1]),
               "corr_in_team_current_car": float(np.corrcoef((skill + compat)[:, i], cars[:, i])[0, 1]),
               "qualifying_race_strength_mean": float((race_post["b_driver"][rp] * Q[:, i]).mean()),
               "extra_race_strength_mean": float(extra[:, i].mean()),
               "qualifying_events": int((design.entries.driver_id == driver).sum()),
               "career_residual_available": bool(driver in drivers),
               **total_by_id.get(driver, {})}
        for key, a in (("portable", skill), ("team_effect", compat),
                       ("in_team", skill + compat), ("career_residual", all_quality)):
            row.update({f"{key}_{k}": v for k, v in quantiles(a[:, i]).items()})
        if driver in coverage.index:
            row.update({"career_race_events": int(coverage.loc[driver, "races"]),
                        "career_race_laps": int(coverage.loc[driver, "clean_laps"])})
        components.append(row)

    # All historical matching qualifying segments; aggregate events with equal weight.
    observations = design.obs.merge(design.entries[["entry_idx", "season"]], on="entry_idx")
    hobs = observations[observations.driver_id == "hadjar"]
    mates = hobs.merge(observations[observations.driver_id != "hadjar"],
                      on=["session_idx", "team"], suffixes=("_hadjar", "_mate"))
    mates["gap_pct"] = mates.y_hadjar - mates.y_mate
    event_gaps = mates.groupby(["season_hadjar", "driver_id_mate", "event_id_hadjar"]).gap_pct.mean().reset_index()
    teammate_rows = []
    for (year, mate), g in event_gaps.groupby(["season_hadjar", "driver_id_mate"]):
        matches = mates[(mates.season_hadjar == year) & (mates.driver_id_mate == mate)]
        teammate_rows.append({"season": int(year), "teammate": mate, "events": len(g),
                              "matched_segments": len(matches), "mean_gap_pct": float(g.gap_pct.mean()),
                              "median_gap_pct": float(g.gap_pct.median()),
                              "events_hadjar_faster": int((g.gap_pct > 0).sum())})
    pairs = pd.read_csv(ROOT / "outputs/race/stage_a_pairs.csv")
    pairs = pairs[(pairs.a == "hadjar") | (pairs.b == "hadjar")].copy()
    pairs["teammate"] = np.where(pairs.a == "hadjar", pairs.b, pairs.a)
    pairs["hadjar_gap_pct"] = np.where(pairs.a == "hadjar", pairs.pace_gap, -pairs.pace_gap)
    pairs["season"] = pairs.event_id.str[:4].astype(int)
    race_comparisons = [{"season": int(y), "teammate": mate, "races": len(g),
                         "mean_adjusted_gap_pct": float(g.hadjar_gap_pct.mean()),
                         "median_adjusted_gap_pct": float(g.hadjar_gap_pct.median()),
                         "races_hadjar_faster": int((g.hadjar_gap_pct > 0).sum())}
                        for (y, mate), g in pairs.groupby(["season", "teammate"])]
    sensitivity = pd.read_csv(ROOT / "outputs/validation/sensitivity_current_drivers_in_team.csv").set_index("driver_id")
    sensitivity_cols = [s for s in sensitivity if s == "main" or s.startswith("sens_")]
    sens = [{"variant": s, "hadjar_in_team_s_median": float(sensitivity.loc["hadjar", s]),
             "hadjar_order": int(sensitivity[s].rank(ascending=False, method="min").loc["hadjar"])}
            for s in sensitivity_cols]
    ties = (baseline == baseline.max(axis=1, keepdims=True)).sum(axis=1) > 1
    summary = {
        "status": "exploratory diagnosis; no new validation or model selection", "gate": False,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "data_as_of": reference["data_as_of"], "fits_run": 0,
        "baseline_replay_max_abs_points_error": replay_error,
        "simulated_seasons_per_seed": c.N_SEASONS, "races_per_season": n_races, "seeds": list(SEEDS),
        "missing_current_race_residuals": sorted(set(ids) - set(drivers)),
        "race_coefficients": {k: quantiles(v) for k, v in race_post.items()},
        "qualifying_hyperparameters": {k: quantiles(v) for k, v in hyper.items()},
        "hadjar_qualifying_teammate_comparisons": teammate_rows,
        "hadjar_adjusted_race_teammate_comparisons": race_comparisons,
        "existing_qualifying_structural_sensitivity": sens,
        "points_tied_for_title_seasons": int(ties.sum()), "points_tied_for_title_fraction": float(ties.mean()),
        "hadjar_vs_current_teammate_in_team_pace_correlation": float(np.corrcoef(
            (skill + compat)[:, h], (skill + compat)[:, list(ids).index("max_verstappen")])[0, 1]),
        "independent_total_race_validation": published_race["validation"]["metrics"]["drivers"],
        "interpretation": "Scenario differences hold coefficients fixed. They are not refitted models, causal effects or held-out improvements. Pairwise probabilities concern simulated season points; MC errors concern integration accuracy, not uncertainty in ability.",
        "elapsed_seconds": time.monotonic() - started}
    if before != {p: artifacts.digest(p) for p in inputs}:
        raise RuntimeError("An audit input changed")
    OUT.mkdir(parents=True)
    pd.DataFrame(scenario_rows).to_csv(OUT / "scenarios.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "simulation_seeds.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "driver_components.csv", index=False)
    pd.DataFrame(paired_points(baseline, ids, "hadjar", n_races)).to_csv(OUT / "hadjar_pairwise.csv", index=False)
    event_gaps.to_csv(OUT / "hadjar_qualifying_event_gaps.csv", index=False)
    pairs.to_csv(OUT / "hadjar_race_event_gaps.csv", index=False)
    artifacts.atomic_json(OUT / "summary.json", summary)
    artifacts.record(OUT, list(OUT.iterdir()), inputs=inputs, model="equal-car-preview-audit-v1")
    print(json.dumps({k: summary[k] for k in ("status", "baseline_replay_max_abs_points_error", "fits_run", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
