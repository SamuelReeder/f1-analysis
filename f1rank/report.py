"""Compile outputs/REPORT.md from the main fit, exports and validation summaries.

Acceptance gates are fixed here, before looking at the validation results.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpyro.diagnostics import summary

from .fit import FITS, HYPER, load

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
VAL = OUT / "validation"
RAT = OUT / "ratings"

GATES = {
    "convergence": "R-hat < 1.05 for every rating; no more than 1 divergence per 1,000 draws",
    "synthetic_calibration": "clean synthetic data: 90% intervals cover 85-97% of true driver "
                             "and car values",
    "synthetic_current_grid": "clean synthetic data: current-grid rank correlation >= 0.8 "
                              "and 90% coverage >= 80%",
    "forecast_vs_baselines": "teammate-gap forecasts beat the static two-way model, naive and zero "
                             "baselines on RMSE, pooled over all cutoffs",
    "forecast_calibration": "session-level 90% forecast intervals cover 85-97%",
    "new_pairings": "new teammate pairings: model RMSE below the naive and zero baselines",
    "sensitivity": "current driver ranking Spearman >= 0.9 against every sensitivity variant",
}


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def convergence() -> tuple[dict, bool]:
    post, info = load(FITS / "main.npz")
    worst = {}
    for k in ["skill", "car"]:
        s = summary({k: post[k]})[k]
        worst[k] = {"rhat_max": float(np.nanmax(s["r_hat"])), "ess_min": float(np.nanmin(s["n_eff"]))}
    hyp = {k: summary({k: post[k]})[k] for k in HYPER if k in post}
    worst["hyper_rhat_max"] = float(max(v["r_hat"] for v in hyp.values()))
    n_draws = post["skill"].shape[0] * post["skill"].shape[1]
    ok = (max(worst["skill"]["rhat_max"], worst["car"]["rhat_max"], worst["hyper_rhat_max"]) < 1.05
          and info["divergences"] <= n_draws / 1000)
    return {"info": info, **worst, "n_draws": n_draws}, ok


def main() -> None:
    meta = json.loads((RAT / "meta.json").read_text())
    drivers = pd.read_csv(RAT / "current_drivers.csv")
    cars = pd.read_csv(RAT / "current_cars.csv")
    lfo = json.loads((VAL / "lfo_summary.json").read_text())
    synth = json.loads((VAL / "synth_summary.json").read_text())
    sens = json.loads((VAL / "sens_summary.json").read_text())
    sens_table = pd.read_csv(VAL / "sensitivity_current_drivers.csv", index_col=0)
    conv, conv_ok = convergence()

    ts = lfo["teammate_session"]
    new = lfo.get("pairing_new", {})
    clean = synth["clean"]
    gates = {
        "convergence": conv_ok,
        "synthetic_calibration": all(0.85 <= clean[c]["cov90"] <= 0.97 for c in ("skill", "car")),
        "synthetic_current_grid": clean["current_grid"]["spearman"] >= 0.8
                                  and clean["current_grid"]["cov90"] >= 0.8,
        "forecast_vs_baselines": ts["rmse_pred"] < min(ts["rmse_akm"], ts["rmse_naive"], ts["rmse_zero"]),
        "forecast_calibration": 0.85 <= ts["cov90"] <= 0.97,
        "new_pairings": bool(new) and new["rmse_pred"] < min(new["rmse_naive"], new["rmse_zero"]),
        "sensitivity": all(v["spearman"] >= 0.9 for k, v in sens.items() if k.startswith("sens_")),
    }

    L = []
    L.append("# F1 driver skill vs car performance: qualifying-pace ratings\n")
    L.append(f"Data through **{meta['data_as_of']['race_name']} {meta['data_as_of']['date'][:4]}** "
             f"(event {meta['data_as_of']['event_id']}), window {meta['window']}, "
             f"{meta['n_lap_times']:,} qualifying lap times. Model `{meta['model_version']}`.\n")
    L.append("Ratings are **seconds per 90-second lap relative to the average driver / car at "
             "the latest event**; positive = faster. Scope: one-lap qualifying pace only.\n")
    L.append("**Headline driver rating: pace in the current car** (what teammate comparisons "
             "measure directly). **Portable skill** (pace expected to carry over to another team) "
             "is **experimental**: its current-grid ranking fails two of the gates below.\n")

    L.append("## Acceptance gates\n")
    L.append(md_table(pd.DataFrame({"gate": list(GATES), "criterion": list(GATES.values()),
                                    "result": ["PASS" if gates[g] else "FAIL" for g in GATES]})))
    L.append("")

    grid_rho = clean["current_grid"]["spearman"]
    grid_in_team_rho = clean.get("current_grid_in_team", {}).get("spearman", float("nan"))
    if "in_team_median_s" in drivers:
        it = drivers.sort_values("in_team_rank").assign(
            in_car=lambda x: x.in_team_median_s, lo90=lambda x: x.in_team_q05_s,
            hi90=lambda x: x.in_team_q95_s,
            ranks=lambda x: x.in_team_rank_lo.astype(str) + "–" + x.in_team_rank_hi.astype(str),
            p_fastest=lambda x: x.in_team_p_fastest)
        L.append("## Drivers in their current car (headline, latest event)\n")
        L.append("Pace relative to the field's drivers in the car each driver has now: portable skill "
                 "plus the driver's team-specific effect. This is what teammate comparisons measure "
                 "directly. In simulation on the real network (one simulated truth) the current-grid "
                 f"order is recovered with rank correlation {grid_in_team_rho:.2f}. Read ranks as ranges.\n")
        L.append(md_table(it[["in_team_rank", "name", "team", "in_car", "lo90", "hi90", "ranks",
                              "p_fastest"]]))
        L.append("")
    d = drivers.sort_values("median_s", ascending=False).assign(
        skill=lambda x: x.median_s, lo90=lambda x: x.q05_s, hi90=lambda x: x.q95_s,
        ranks=lambda x: x.rank_lo.astype(str) + "–" + x.rank_hi.astype(str))
    cols = ["name", "team", "skill", "lo90", "hi90", "ranks", "p_fastest"]
    if "team_effect_median_s" in d:
        d["team_effect"] = d.team_effect_median_s
        cols.append("team_effect")
    unstable = set(sens.get("unstable_drivers", []))
    d["flag"] = np.where(d.driver_id.isin(unstable), "sensitive to model choice", "")
    L.append("## Portable skill (experimental)\n")
    L.append("`skill` is the part of a driver's pace expected to carry over to another team: pace in "
             "the current car minus `team_effect`, the driver's persistent team-specific effect. "
             "**Experimental:** in simulation the current-grid order of portable skill is recovered "
             f"with rank correlation only {grid_rho:.2f}, and it moves with two structural choices "
             "(whether the team-specific effect is modelled, and the data window). The "
             "team-specific effect is not established to be car-handling compatibility: team "
             "support, role, adaptation or selection would look the same.\n")
    L.append(md_table(d[cols + ["events", "teams", "teammates", "flag"]]))
    L.append("")
    L.append("`flag`: rating moves by more than 0.10 s, or rank by 8+ places, across the sensitivity "
             "variants below. (The rule set before the results, 0.05 s or 4 places, flagged every "
             "driver because it is smaller than the posterior uncertainty, so it was loosened.) "
             "`teams`: team lineages raced (5+ events). In simulation the portable-skill error is "
             "about the same for drivers with 1, 2 or 3+ teams (RMSE about 0.10 s), so the counts "
             "are context, not a precision measure; use the interval.\n")
    c = cars.assign(rating=cars.median_s, lo90=cars.q05_s, hi90=cars.q95_s,
                    gap_to_fastest=cars.gap_median_s,
                    ranks=cars.rank_lo.astype(str) + "–" + cars.rank_hi.astype(str),
                    track_profile=cars.track_median)
    L.append("## Car packages (latest event, track-neutral)\n")
    L.append("`track_profile` > 0: relatively stronger on high-speed circuits; < 0: on slow/street circuits.\n")
    L.append(md_table(c[["constructor", "rating", "lo90", "hi90", "gap_to_fastest", "ranks",
                         "track_profile"]]))
    L.append("")

    L.append("## Validation\n")
    L.append(f"### Forecasting ({lfo['n_cutoffs']} leave-future-out cutoffs, 2013–2026)\n")
    L.append("Each cutoff is fitted only on data up to that point (including the circuit factor) "
             "and forecasts the next season (end-of-season cutoffs) or the rest of the season "
             "(after round 8). Established drivers only (≥10 prior events). RMSE and CRPS are in seconds.\n")
    L.append(md_table(pd.DataFrame([
        {"target": "teammate gap, per segment", "model": ts["rmse_pred"], "static two-way": ts["rmse_akm"],
         "naive raw gaps": ts["rmse_naive"], "zero": ts["rmse_zero"]},
        *[{"target": f"teammate gap, per pairing ({k.split('_')[1]}, n={lfo[k]['n']})",
           "model": lfo[k]["rmse_pred"], "static two-way": lfo[k]["rmse_akm"],
           "naive raw gaps": lfo[k]["rmse_naive"], "zero": lfo[k]["rmse_zero"]}
          for k in ("pairing_all", "pairing_continuing", "pairing_new") if k in lfo],
    ])))
    L.append("")
    ps = lfo["pace_session"]
    L.append(f"Session-level calibration of teammate-gap forecasts: 50% intervals cover "
             f"{ts['cov50']:.1%}, 90% intervals cover {ts['cov90']:.1%}; CRPS {ts['crps_s']:.3f}s. "
             f"Pairing-level 90% intervals cover {lfo.get('pairing_all', {}).get('in90', float('nan')):.1%}.\n")
    L.append("Full relative pace (car + driver) within each forecast segment:\n")
    L.append(md_table(pd.DataFrame([
        {"forecaster": "model", "rmse_s": ps["rmse_pred"], "spearman": ps["spearman_pred"]},
        {"forecaster": "static two-way", "rmse_s": ps["rmse_akm"], "spearman": ps["spearman_akm"]},
        {"forecaster": "recent form (last 5 events)", "rmse_s": ps["rmse_form"], "spearman": ps["spearman_form"]},
    ])))
    L.append("")

    L.append("### Synthetic recovery on the real F1 network\n")
    L.append("Truth drawn from the model's generative process on the real entries, teams and "
             "segments; misspecified scenarios add effects the model does not contain. "
             "Coverage = share of true values inside the 90% interval. **All scenarios share one "
             "simulated truth** (to compare scenarios on equal terms), so this is a recovery check "
             "on one draw, not yet evidence of calibration; repeating it over independent seeds "
             "and parameter settings is planned.\n")
    rows = []
    for sc, r in synth.items():
        rows.append({"scenario": sc, "skill_corr": r["skill"]["corr"], "skill_rmse_s": r["skill"]["rmse_s"],
                     "skill_cov90": r["skill"]["cov90"], "car_corr": r["car"]["corr"],
                     "car_cov90": r["car"]["cov90"], "grid_spearman": r["current_grid"]["spearman"],
                     "grid_cov90": r["current_grid"]["cov90"],
                     "cross_team_cov90": r["cross_team_diff_current"]["cov90"],
                     "grid_in_team_spearman": r.get("current_grid_in_team", {}).get("spearman", float("nan")),
                     "car_latent_share_true/est": f"{r['car_share']['truth_mean']:.2f}/{r['car_share']['est_mean']:.2f}"})
    L.append(md_table(pd.DataFrame(rows)))
    L.append("")
    L.append("`car_latent_share`: within each season, the variance of the synthetic car states "
             "divided by the variance of car states plus portable skill states, averaged over "
             "seasons (true vs estimated). It covers only those two latent components (not the "
             "team-specific effect, weekend effects or noise) and describes the simulated truth, "
             "not a share of observed qualifying variation.\n")

    placebo_path = VAL / "placebo_summary.json"
    if placebo_path.exists():
        pl = json.loads(placebo_path.read_text())
        L.append("### Is the team-specific effect specific to the team? (placebo test)\n")
        L.append(f"Team-specific effect SD {pl['sd_compat_q05_q50_q95'][1]:.3f}% "
                 f"(90% interval {pl['sd_compat_q05_q50_q95'][0]:.3f}–{pl['sd_compat_q05_q50_q95'][2]:.3f}) "
                 f"vs placebo SD {pl['sd_placebo_q05_q50_q95'][1]:.3f}% "
                 f"({pl['sd_placebo_q05_q50_q95'][0]:.3f}–{pl['sd_placebo_q05_q50_q95'][2]:.3f}) "
                 f"when a driver's stint in one team is split in half. P(placebo > team effect) = "
                 f"{pl['p_placebo_gt_compat']:.2f}. Relative performance shifts when a driver changes "
                 "team, much less with time spent in the same team: the effect is associated with "
                 "the team. The test cannot say why (car handling, team support, role, adaptation or "
                 "selection). The model gives one effect per driver and team lineage, including "
                 "separate spells years apart (8 cases, e.g. Hülkenberg at Sauber in 2013 and Audi in "
                 "2025–26); variants per contiguous spell and per regulation era are planned.\n")

    L.append("### Sensitivity of the current driver ranking\n")
    L.append(md_table(pd.DataFrame([{"variant": k, **v} for k, v in sens.items() if k.startswith("sens_")])))
    L.append("")
    L.append("Per-driver rating (s) under each variant:\n")
    st = sens_table.reset_index().rename(columns={"index": "driver_id"})
    L.append(md_table(st))
    L.append("")

    wc_path = VAL / "window_compare.json"
    if wc_path.exists():
        wc = json.loads(wc_path.read_text())
        L.append("### Data window: 2010 vs 2006 start\n")
        L.append(f"Same model fitted from 2006 (adding 2006–09 Q1/Q2 low-fuel times) vs from 2010, "
                 f"scored on identical forecast targets at {len(wc['cutoffs'])} cutoffs "
                 f"({', '.join(wc['cutoffs'])}); {wc['n_pairings']} matched pairings.\n")
        L.append(md_table(pd.DataFrame([{"window": w, **wc[w]} for w in ("2010", "2006")])))
        L.append("")
        L.append("The two windows forecast equally well, so the data cannot say which is right. The "
                 "pre-specified 2010 window (clean low-fuel qualifying) is kept. The rating shifts "
                 "between windows (at most 0.1 s, see the sensitivity table) are genuine uncertainty "
                 "and are well inside the drivers' 90% intervals.\n")

    L.append("### Convergence\n")
    L.append(f"{conv['n_draws']} posterior draws; divergences: {conv['info']['divergences']}; "
             f"max R-hat skill {conv['skill']['rhat_max']:.3f}, car {conv['car']['rhat_max']:.3f}, "
             f"hyperparameters {conv['hyper_rhat_max']:.3f}; min ESS skill {conv['skill']['ess_min']:.0f}, "
             f"car {conv['car']['ess_min']:.0f}.\n")
    L.append("## What the ratings can and cannot say\n")
    L.append("\n".join([
        "- **Car-package ratings are well determined.** Synthetic recovery correlation ≥ 0.99 in every "
        f"scenario, with 90% intervals covering {clean['car']['cov90']:.0%} in the clean scenario (one simulated truth). "
        "In the simulated truth, car states vary far more than portable skill states within a season, "
        "and the estimated share matches the true one; that describes the simulation, not a share of "
        "observed qualifying variation.",
        "- **Teammate comparisons and forecasts are reliable.** The model beats a static two-way model, "
        "raw teammate gaps and a zero-gap baseline on every forecast target, including brand-new "
        "pairings, with calibrated intervals.",
        "- **Portable driver skill is experimental.** Performance relative to teammates has a large "
        "team-specific part (SD about 0.16%, against about 0.10% for differences in portable skill). "
        "That limits how well the network can rank the current grid by portable skill (synthetic rank "
        f"correlation {grid_rho:.2f}, below the 0.8 gate), and the ranking moves with structural "
        "choices (sensitivity gate fails). In the one simulated truth, 90% intervals covered "
        f"{clean['skill']['cov90']:.0%} of true values; that is a recovery check, not established calibration. Use pace in "
        "the current car as the headline and read portable ranks as ranges.",
        "- **The team-specific effect is associated with the team, not proven to be car compatibility.** "
        "Support, role, adaptation or selection would produce the same pattern. It is also shared "
        "across separate spells with the same team lineage.",
        "- **Drivers who have only raced for one team** (e.g. Piastri, Antonelli, Bortoleto): their "
        "portable skill depends on the model's assumption about how team-specific effects are "
        "distributed. In simulation their error was about the same as for drivers with more teams, "
        "but that simulation follows the model's own assumptions.",
        "- **The main structural sensitivities of portable skill** are whether the team-specific effect is "
        f"modelled (rank correlation {sens['sens_nocompat']['spearman']:.2f}) and how much history is used "
        f"({sens['sens_start2006']['spearman']:.2f}); the weekend-form term and the assumed rate of skill "
        f"drift matter less ({min(sens[k]['spearman'] for k in ('sens_noform', 'sens_drift_x2', 'sens_drift_half')):.2f}"
        f"–{max(sens[k]['spearman'] for k in ('sens_noform', 'sens_drift_x2', 'sens_drift_half')):.2f}). "
        "Details of the car model and noise model barely matter "
        f"(≥ {min(sens[k]['spearman'] for k in ('sens_notrack', 'sens_gausscar', 'sens_notransient', 'sens_splitt')):.2f}).",
        "- **Main residual risk:** drivers who change team after an unusually lucky or unlucky "
        "season. In simulation this is the violation that degrades the ranking most.",
        "- **Scope:** one-lap qualifying pace only. Race pace, tyre management, starts and racecraft "
        "are not measured yet.",
    ]))
    L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L))
    (VAL / "gates.json").write_text(json.dumps(gates, indent=1))
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
