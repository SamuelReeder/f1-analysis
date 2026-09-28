"""Compile outputs/REPORT.md from the main fit, exports and validation summaries.

Acceptance gates are fixed here, before looking at the validation results they judge.
Changes to them are listed in GATE_HISTORY with what was known when they were made.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpyro.diagnostics import summary

from .design import build_design
from .fit import FITS, HYPER, load

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
VAL = OUT / "validation"
RAT = OUT / "ratings"
BASIS_LABEL = {"status code": "from a coded status", "inferred from evidence": "inferred from race evidence",
               "class base rates": "from class base rates (uncoded, no race evidence)"}
FILL_CHECK = OUT / "analysis" / "quali_fill" / "check.json"

GATES = {
    "convergence": "R-hat < 1.05 for every rating (portable skill, team-specific effect, car) and "
                   "hyperparameter; no more than 1 divergence per 1,000 draws",
    "synthetic_calibration": "clean synthetic data, independent truths: mean 90% interval coverage "
                             "of true driver and car values 85-97%",
    "synthetic_current_grid": "clean synthetic data, independent truths: portable skill's current-grid "
                              "rank correlation >= 0.8 and 90% coverage >= 80% (means over truths)",
    "synthetic_current_grid_in_team": "the same for pace in the current car",
    "forecast_vs_baselines": "teammate-gap forecasts beat the static two-way model, naive and zero "
                             "baselines on RMSE, pooled over all cutoffs",
    "forecast_calibration": "session-level 90% forecast intervals cover 85-97%",
    "new_pairings": "new teammate pairings: model RMSE below the naive and zero baselines",
    "sensitivity": "portable skill: current-grid Spearman >= 0.9 against every sensitivity variant",
    "sensitivity_in_team": "pace in the current car: current-grid Spearman >= 0.9 against every "
                           "sensitivity variant",
}
GATE_HISTORY = [
    "2026-09-28, before the refit whose results they judge: the two synthetic gates use 8 "
    "independent clean truths (before: one truth shared by all scenarios). Two gates were added "
    "for the headline rating, pace in the current car (`synthetic_current_grid_in_team`, "
    "`sensitivity_in_team`). When they were added, its single-truth rank correlation (0.91) was "
    "known; its sensitivity results were not. The convergence gate now also covers the "
    "team-specific effect, which is part of the headline rating.",
]


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
    for k in ["skill", "compat", "car"]:
        s = summary({k: post[k]})[k]
        worst[k] = {"rhat_max": float(np.nanmax(s["r_hat"])), "ess_min": float(np.nanmin(s["n_eff"]))}
    hyp = {k: summary({k: post[k]})[k] for k in HYPER if k in post}
    worst["hyper_rhat_max"] = float(max(v["r_hat"] for v in hyp.values()))
    n_draws = post["skill"].shape[0] * post["skill"].shape[1]
    ok = (max(worst[k]["rhat_max"] for k in ("skill", "compat", "car")) < 1.05
          and worst["hyper_rhat_max"] < 1.05 and info["divergences"] <= n_draws / 1000)
    return {"info": info, **worst, "n_draws": n_draws}, ok


def _mean_range(summary_: dict, key: str, nd: int = 2) -> str:
    s = summary_[key]
    return f"{s['mean']:.{nd}f} ({s['min']:.{nd}f}–{s['max']:.{nd}f})"


def _paired_row(label: str, r: dict) -> dict:
    if not r.get("n"):
        return {"comparison": label, "n": 0}
    lo, hi = r["mse_diff_ci95"]
    return {"comparison": label, "n": r["n"], "rmse_main_s": r["rmse_main_s"],
            "rmse_variant_s": r["rmse_variant_s"], "mse_diff_s2": f"{r['mse_diff_s2']:+.4f}",
            "95% interval": f"{lo:+.4f} to {hi:+.4f}", "in90_main": r["in90_main"],
            "in90_variant": r["in90_variant"]}


def _num(x: float) -> str:
    return f"{x:+.4f}" if abs(x) >= 1e-3 or x == 0 else f"{x:+.1e}"


def _heldout_row(label: str, r: dict) -> dict:
    lo, hi = r["mse_diff_ci95"]
    return {"test": label, "pair-seasons": r["n_pair_seasons"], "rmse_baseline_s": r["rmse_baseline"] * 0.9,
            "rmse_model_s": r["rmse_model"] * 0.9, "mse_diff_%2": _num(r["mse_diff"]),
            "95% interval": f"{_num(lo)} to {_num(hi)}"}


def racing_section() -> list[str]:
    """Stage 2 (racing), from the timeline audit and the race-pace outputs, when present."""
    audit_path, race_path = OUT / "timeline" / "audit.json", OUT / "race" / "stage_b_summary.json"
    if not audit_path.exists():
        return []
    a = json.loads(audit_path.read_text())
    v = a["cause_model"]["validation"]
    ret = a["retirements"]
    L = ["## Racing (stage 2, in progress)\n",
         "### Event timeline (outcomes from 2010, race evidence from 2018)\n",
         f"{a['rows']:,} rows over {a['races']} races: neutralisations, retirements and other outcomes, and "
         "race-control evidence (incidents, penalties, stoppages, off-track moments), plus pit anomalies, "
         "suspected damage and possible team orders. Rules produce evidence, not verdicts. Full audit: "
         "`outputs/timeline/AUDIT.md`.\n",
         f"- **Retirement causes.** {ret['n']} retirements: "
         + ", ".join(f"{n} {BASIS_LABEL.get(k, k)}" for k, n in ret["by_basis"].items())
         + ". Jolpica codes causes through 2022 but records almost every later retirement only as "
           "\"Retired\"; those get probabilities from a model of the cause class given race-control and "
           f"lap evidence, fitted on {v['n']} coded retirements. Held out a season at a time, its log loss "
           f"is {v['log_loss_model']:.3f} against {v['log_loss_base_rates']:.3f} for base rates (accuracy "
           f"{v['accuracy_model']:.2f} vs {v['accuracy_base_rates']:.2f}): informative, far from certain.",
         "- **Who caused an incident** is split by stated assumptions (the audit lists them), not "
         "estimates; later models vary them.", ""]
    if not race_path.exists():
        return L
    r = json.loads(race_path.read_text())
    h = r["heldout"]
    L += ["### Race pace and tyre degradation (2018 onward, dry races)\n",
          f"Stage A estimates each driver's pace (at tyre age 10 laps) and degradation per race from clean laps "
          f"(robust regression with fuel/track trend, compounds, dirty air; block-bootstrap errors). Stage B "
          f"models {r['n_teammate_races']:,} teammate comparisons in {r['n_races']} races: pace gap = gamma × "
          "qualifying gap (stage 1, pace in the current car) + a race-specific part, and a degradation gap. "
          f"gamma = {r['gamma_q05_q50_q95'][1]:.2f} (90% interval {r['gamma_q05_q50_q95'][0]:.2f}–"
          f"{r['gamma_q05_q50_q95'][2]:.2f}); SD of race-specific pace {r['sd_race_specific_q05_q50_q95'][1]:.3f}% "
          f"({r['sd_race_specific_q05_q50_q95'][0]:.3f}–{r['sd_race_specific_q05_q50_q95'][2]:.3f}); max R-hat "
          f"{r['rhat_max']:.3f}, {r['divergences']} divergences.\n",
          "Held-out seasons (each predicted from the seasons before it; pair-season means; mse in %², "
          "negative = model better; bootstrap 95% interval over pair-seasons):\n",
          md_table(pd.DataFrame([
              _heldout_row("qualifying link vs zero", h["quali_link_vs_zero"]),
              _heldout_row("+ race-specific part vs qualifying link", h["race_specific_pace_vs_quali_link"]),
              _heldout_row("degradation effects vs zero", h["degradation_vs_zero"])])), "",
          f"- **Race-specific pace:** {'passes' if r['gate_race_specific_pace'] else 'does not pass'} its gate "
          "(the interval must lie below zero), so it " + ("gets its own ranking." if r["gate_race_specific_pace"] else
                                              "is not published as a ranking; race pace is represented by the "
                                              "qualifying link."),
          f"- **Degradation:** {'passes' if r['gate_degradation'] else 'does not pass'} its gate.", ""]
    jc_path = OUT / "race" / "joint_check_2024.json"
    if jc_path.exists():
        jc = json.loads(jc_path.read_text())
        ag = jc["agreement"]
        L += [f"- **Two-stage shortcut vs joint lap-level model ({jc['season']}, {jc['n_laps']:,} laps):** "
              f"{'accepted' if jc['accepted'] else 'not accepted'} by the rule set beforehand "
              f"({jc['acceptance_rule']}). Race-specific pace: {ag['race_specific_pace']['share_within_quarter_sd']:.0%} "
              f"of drivers within a quarter SD (correlation of driver means "
              f"{ag['race_specific_pace']['corr_of_means']:.2f}); degradation "
              f"{ag['degradation']['share_within_quarter_sd']:.0%} ({ag['degradation']['corr_of_means']:.2f}). "
              + _scale_note(jc)
              + ("The two-stage driver estimates are therefore not published; the held-out conclusion is "
                 "checked with the joint model below." if not jc["accepted"] else ""), ""]
    jh_path = OUT / "race" / "joint_heldout.json"
    if jh_path.exists():
        jh = json.loads(jh_path.read_text())
        rows = [{"method": m, "pair-seasons": r["n_pair_seasons"], "mse_diff_%2": f"{r['mse_diff']:+.4f}",
                 "95% interval": f"{r['mse_diff_ci95'][0]:+.4f} to {r['mse_diff_ci95'][1]:+.4f}",
                 "improves": "yes" if r["improves"] else "no"} for m, r in jh["per_method"].items()]
        same = len({r["improves"] for r in jh["per_method"].values()}) == 1
        L += [f"Race-specific pace with the joint model ({jh['design']}), against the qualifying link alone:\n",
              md_table(pd.DataFrame(rows)), "",
              ("Both methods give the same answer, so the conclusion on race-specific pace does not depend on "
               "the two-stage shortcut." if same else "The methods disagree: the race-specific pace conclusion "
               "depends on the method."), ""]
    L += _stage2_parts()
    return L


def _scale_note(jc: dict) -> str:
    """How much of the disagreement is scale: spread of the joint model's driver means over
    the two-stage ones (same drivers)."""
    d = pd.DataFrame(jc["drivers"])
    ratio = {q: g.joint_mean.std() / g.two_stage_mean.std() for q, g in d.groupby("quantity")}
    return (f"The driver means differ mostly in scale: the joint model's spread is {ratio['race_specific_pace']:.1f}× "
            f"the two-stage one for race-specific pace and {ratio['degradation']:.1f}× for degradation (less "
            "shrinkage). ")


def _ci(r: dict, nd: int = 3) -> str:
    return f"{r['mean_diff_per_race']:+.{nd}f} per race (95% interval {r['ci95'][0]:+.{nd}f} to {r['ci95'][1]:+.{nd}f})"


def _stage2_parts() -> list[str]:
    L = []
    rel_path = OUT / "reliability" / "summary.json"
    if rel_path.exists():
        r = json.loads(rel_path.read_text())
        h = r["heldout"]
        L += ["### Reliability and driver errors (2010 onward)\n",
              f"Retirements as competing risks per lap ({r['driver_races']:,} starts, {r['retirements']:,} "
              f"retirements, {r['laps_at_risk']:,.0f} laps at risk); each retirement counts fractionally for each "
              "cause by its timeline probabilities. Held-out seasons, paired log predictive density per race:\n",
              f"- **Driver own-error effects** (baseline keeps team-season terms): {_ci(h['driver_effects'])}: "
              f"{'passes' if r['gate_driver_error_ranking'] else 'does not pass'} its gate.",
              f"- **Team-season mechanical effects** (second half of each season from its first half): "
              f"{_ci(h['team_season_effects'])}: {'passes' if r['gate_team_reliability'] else 'does not pass'}.",
              "- Own-error attribution rests on the timeline's stated assumptions (who caused an incident); the "
              "driver ranking inherits them.", ""]
    b_path = OUT / "benchmark" / "summary.json"
    if b_path.exists():
        b = json.loads(b_path.read_text())
        rows = [{"model": v, "log_lik_per_race": x["mean_log_lik_per_race"], "spearman": x["mean_spearman"],
                 "teammate_h2h": x["teammate_h2h_accuracy"]} for v, x in b["per_variant"].items()]
        L += ["### Simple results benchmark\n",
              f"Rank-ordered logit on full classifications, held-out seasons {b['seasons_held_out'][0]}–"
              f"{b['seasons_held_out'][1]} ({b['races']} races):\n", md_table(pd.DataFrame(rows)), "",
              f"- Driver results effect over stage-1 ratings: {_ci(b['ratings_results_minus_ratings'])}.",
              f"- Stage-1 ratings vs grid position alone: {_ci(b['ratings_minus_grid'])}."]
        if "grid_ratings_minus_grid" in b:
            L.append(f"- Grid + ratings vs grid alone: {_ci(b['grid_ratings_minus_grid'])}.")
        L.append("")
    f_path = OUT / "firstlap" / "summary.json"
    if f_path.exists():
        f = json.loads(f_path.read_text())
        L += ["### First-lap performance (2018 onward)\n",
              f"Positions gained on lap 1 ({f['starts']:,} starts). Driver SD {f['sd_driver_q05_q50_q95'][1]:.2f} "
              f"positions (90% interval {f['sd_driver_q05_q50_q95'][0]:.2f}–{f['sd_driver_q05_q50_q95'][2]:.2f}); "
              f"held-out driver effects {_ci(f['heldout_driver_effects'])}. Rank correlation "
              f"with lap-1 contact incidents dropped: {f['sensitivity_no_lap1_contact']['driver_rank_corr']:.2f}."
              + (f" Transfer test (starts by drivers in a new team, {f['heldout_driver_effects']['transfer']['n_starts']} "
                 f"starts): {_ci(f['heldout_driver_effects']['transfer'])}. Held-out 90% intervals cover "
                 f"{f['heldout_driver_effects']['cov90']:.0%}." if "transfer" in f["heldout_driver_effects"] else ""),
              f"- **Gate (held-out improvement, and not worse for drivers who changed team):** "
              f"{'passes' if f['gate_driver_ranking'] else 'does not pass'}."
              + ("" if f["gate_driver_ranking"] else " The improvement comes from drivers staying in the same "
                 "team; for drivers in a new team the effects made predictions worse, so the effect looks tied "
                 "to the driver-team combination (e.g. a team's launch procedures), not a portable driver "
                 "skill. No driver ranking is published."), ""]
        if f["gate_driver_ranking"]:
            cur = pd.read_csv(RAT / "current_drivers.csv")[["driver_id", "name"]]
            fl = pd.read_csv(OUT / "firstlap" / "drivers.csv").merge(cur, on="driver_id")
            fl = fl.assign(gained=fl.gained_per_start_median, lo90=fl.q05, hi90=fl.q95)
            L += ["Current grid, positions gained per start relative to the model's expectation for the slot, "
                  "tyre and car (provisional: see the transfer and calibration results above):\n",
                  md_table(fl[["name", "gained", "lo90", "hi90", "starts"]]), ""]
    o_path = OUT / "battles" / "summary.json"
    fe_path = OUT / "battles" / "feasibility.json"
    if fe_path.exists():
        fe = json.loads(fe_path.read_text())
        a = fe["at_estimated_size"]
        L += ["### Overtaking and defending (2018 onward)\n",
              f"Synthetic feasibility at the estimated effect size (criteria: {fe['criteria']}): attacker "
              f"correlation {a['attack']['corr']:.2f}, coverage {a['attack']['cov90']:.0%}; defender correlation "
              f"{a['defend']['corr']:.2f}, coverage {a['defend']['cov90']:.0%}: "
              f"{'feasible' if fe['feasible'] else 'not feasible'}."]
        if o_path.exists():
            o = json.loads(o_path.read_text())
            L.append(f"{o['episodes']:,} battle episodes, {o['passes']:,} passes. "
                     + (f"Held-out attacker/defender effects (car terms kept): {_ci(o['heldout_driver_effects'])}; "
                        f"transfer (a driver in a new team): {_ci(o['heldout_driver_effects']['transfer'])}: "
                        f"{'passes' if o['gate_driver_ranking'] else 'does not pass'} its gate."
                        if "heldout_driver_effects" in o else "Not rated: reported as opportunity counts only."))
        L.append("")
    c_path = OUT / "championship" / "standings.csv"
    if c_path.exists():
        cs = json.loads((OUT / "championship" / "summary.json").read_text())
        st = pd.read_csv(c_path)
        top = st[st.version == "in_team"].head(10).assign(ranks=lambda x: x.rank_lo.astype(str) + "–" + x.rank_hi.astype(str))
        h = cs["heldout_race_stage"]
        L += ["### Equal-car championship (in progress)\n",
              f"{cs['simulated_seasons']:,} simulated {cs['n_races_simulated']}-race seasons with equal cars: "
              "qualifying from stage 1 (pace in the current car), race given the grid from a ranking model, "
              "retirements from the reliability model (equal mechanical risk; driver error rates "
              f"{'used' if cs['driver_error_rates_used'] else 'not used'}). No other driver-specific racing "
              "quality enters, because none has passed its gate. The race stage on held-out finishing orders: "
              f"vs grid alone {_ci(h['vs_grid'])}; vs ratings alone {_ci(h['vs_ratings'])}.\n",
              md_table(top[["name", "points_per_race", "p_title", "ranks"]]), ""]
    return L


def main() -> None:
    meta = json.loads((RAT / "meta.json").read_text())
    drivers = pd.read_csv(RAT / "current_drivers.csv")
    cars = pd.read_csv(RAT / "current_cars.csv")
    lfo = json.loads((VAL / "lfo_summary.json").read_text())
    synth = json.loads((VAL / "synth_summary.json").read_text())
    sens = json.loads((VAL / "sens_summary.json").read_text())
    sens_table = pd.read_csv(VAL / "sensitivity_current_drivers.csv", index_col=0)
    sens_in_team = pd.read_csv(VAL / "sensitivity_current_drivers_in_team.csv", index_col=0)
    compare = json.loads((VAL / "compare.json").read_text()) if (VAL / "compare.json").exists() else {}
    diag = json.loads((VAL / "diagnostics.json").read_text()) if (VAL / "diagnostics.json").exists() else None
    conv, conv_ok = convergence()
    design = build_design(2010, end_event=meta["data_as_of"]["event_id"])
    ent = design.entries

    ts = lfo["teammate_session"]
    new = lfo.get("pairing_new", {})
    seeds = synth["seeds_summary"]
    variants = {k: v for k, v in sens.items() if k.startswith("sens_")}
    gates = {
        "convergence": conv_ok,
        "synthetic_calibration": all(0.85 <= seeds[k]["mean"] <= 0.97 for k in ("skill_cov90", "car_cov90")),
        "synthetic_current_grid": seeds["grid_spearman"]["mean"] >= 0.8 and seeds["grid_cov90"]["mean"] >= 0.8,
        "synthetic_current_grid_in_team": seeds["grid_in_team_spearman"]["mean"] >= 0.8
                                          and seeds["grid_in_team_cov90"]["mean"] >= 0.8,
        "forecast_vs_baselines": ts["rmse_pred"] < min(ts["rmse_akm"], ts["rmse_naive"], ts["rmse_zero"]),
        "forecast_calibration": 0.85 <= ts["cov90"] <= 0.97,
        "new_pairings": bool(new) and new["rmse_pred"] < min(new["rmse_naive"], new["rmse_zero"]),
        "sensitivity": all(v["spearman"] >= 0.9 for v in variants.values()),
        "sensitivity_in_team": all(v["in_team_spearman"] >= 0.9 for v in variants.values()),
    }
    portable_gates = [g for g in ("synthetic_current_grid", "sensitivity") if not gates[g]]
    in_team_gates = [g for g in ("synthetic_current_grid_in_team", "sensitivity_in_team") if not gates[g]]

    L = []
    L.append("# F1 driver skill vs car performance: qualifying-pace ratings\n")
    L.append(f"Data through **{meta['data_as_of']['race_name']} {meta['data_as_of']['date'][:4]}** "
             f"(event {meta['data_as_of']['event_id']}), window {meta['window']}, "
             f"{meta['n_lap_times']:,} qualifying lap times. Model `{meta['model_version']}`, "
             f"fit `{meta['fit_id']}`.\n")
    L.append("Ratings are **seconds per 90-second lap relative to the average driver / car entered at "
             "the latest event**; positive = faster. Scope: one-lap qualifying pace only.\n")
    headline = ("**Headline driver rating: pace in the current car** (what teammate comparisons "
                "measure directly).")
    if in_team_gates:
        headline += f" It fails {len(in_team_gates)} of its own gates ({', '.join(in_team_gates)}); see below."
    L.append(headline + " **Portable skill** (pace expected to carry over to another team) is "
             "**experimental**" + (f": it fails {len(portable_gates)} of the gates below "
                                   f"({', '.join(portable_gates)})." if portable_gates else ".") + "\n")

    L.append("## Acceptance gates\n")
    L.append(md_table(pd.DataFrame({"gate": list(GATES), "criterion": list(GATES.values()),
                                    "result": ["PASS" if gates[g] else "FAIL" for g in GATES]})))
    L.append("")
    L.append("Changes to the gates: " + " ".join(GATE_HISTORY) + "\n")

    grid_rho = _mean_range(seeds, "grid_spearman")
    grid_in_team_rho = _mean_range(seeds, "grid_in_team_spearman")
    it = drivers.sort_values("in_team_rank").assign(
        in_car=lambda x: x.in_team_median_s, lo90=lambda x: x.in_team_q05_s,
        hi90=lambda x: x.in_team_q95_s,
        ranks=lambda x: x.in_team_rank_lo.astype(str) + "–" + x.in_team_rank_hi.astype(str),
        p_fastest=lambda x: x.in_team_p_fastest,
        note=lambda x: np.where(x.has_time, "", "no valid lap: carried forward"))
    L.append("## Drivers in their current car (headline, latest event)\n")
    L.append("Pace relative to the field's drivers in the car each driver has now: portable skill "
             "plus the driver's team-specific effect. This is what teammate comparisons measure "
             f"directly. In simulation on the real network ({seeds['n_seeds']} independent truths) the "
             f"current-grid order is recovered with rank correlation {grid_in_team_rho} (mean, range "
             "over truths). Read ranks as ranges.\n")
    cols = ["in_team_rank", "name", "team", "in_car", "lo90", "hi90", "ranks", "p_fastest"]
    L.append(md_table(it[cols + (["note"] if (~it.has_time).any() else [])]))
    L.append("")
    d = drivers.sort_values("median_s", ascending=False).assign(
        skill=lambda x: x.median_s, lo90=lambda x: x.q05_s, hi90=lambda x: x.q95_s,
        ranks=lambda x: x.rank_lo.astype(str) + "–" + x.rank_hi.astype(str),
        team_effect=lambda x: x.team_effect_median_s)
    unstable = set(sens.get("unstable_drivers", []))
    d["flag"] = np.where(d.driver_id.isin(unstable), "sensitive to model choice", "")
    L.append("## Portable skill (experimental)\n")
    L.append("`skill` is the part of a driver's pace expected to carry over to another team: pace in "
             "the current car minus `team_effect`, the driver's persistent team-specific effect. "
             "**Experimental:** in simulation the current-grid order of portable skill is recovered "
             f"with rank correlation {grid_rho} (mean, range over {seeds['n_seeds']} truths), and it "
             "moves with structural choices (see Sensitivity). The team-specific effect is not "
             "established to be car-handling compatibility: team support, role, adaptation or "
             "selection would look the same.\n")
    L.append(md_table(d[["name", "team", "skill", "lo90", "hi90", "ranks", "p_fastest", "team_effect",
                         "events", "teams", "teammates", "flag"]]))
    L.append("")
    L.append("`flag`: rating moves by more than 0.10 s, or rank by 8+ places, across the sensitivity "
             "variants below. (The rule set before the results, 0.05 s or 4 places, flagged every "
             "driver because it is smaller than the posterior uncertainty, so it was loosened.) "
             "`events`: events entered. `teams`: team lineages raced (5+ events). These counts are "
             "context, not a precision measure; use the interval.\n")
    c = cars.assign(rating=cars.median_s, lo90=cars.q05_s, hi90=cars.q95_s,
                    gap_to_fastest=cars.gap_median_s,
                    ranks=cars.rank_lo.astype(str) + "–" + cars.rank_hi.astype(str),
                    track_profile=cars.track_median)
    L.append("## Car packages (latest event, track-neutral)\n")
    L.append("`track_profile` > 0: relatively stronger on high-speed circuits; < 0: on slow/street circuits.\n")
    L.append(md_table(c[["constructor", "rating", "lo90", "hi90", "gap_to_fastest", "ranks",
                         "track_profile"]]))
    L.append("")

    L.append("## Data\n")
    no_time = int((~ent.has_time).sum())
    L.append(f"- **Entered drivers without a valid lap** ({no_time} of {len(ent):,} driver-events: no "
             "time set, or only laps more than 5% off the segment median) are kept. They have no "
             "observations, their rating is carried forward by the skill walk, and they count in the "
             "field that ratings are relative to.")
    if FILL_CHECK.exists():
        fc = json.loads(FILL_CHECK.read_text())
        filled = sorted(design.obs.loc[design.obs.source == "fastf1", "event_id"].unique()) \
            if "source" in design.obs else []
        L.append(f"- **Times missing from Jolpica** ({', '.join(filled) or 'none'}: every qualifying time "
                 "blank) are rebuilt from FastF1 lap timing (best non-deleted lap per driver and segment, "
                 "`extract/quali_fill.py`). On the other events of the same season the method matches "
                 f"Jolpica exactly for {fc['n_exact']:,} of {fc['n_compared']:,} times. FastF1 also has "
                 f"{len(fc['only_in_fastf1'])} times that Jolpica does not: laps deleted by race control "
                 "that FastF1 did not flag, or times removed after the session, which lap timing cannot "
                 "see. For the filled event, "
                 + ("no race-control lap deletion names a lap that was kept."
                    if not any(x["lap_is_a_kept_best"] for v in fc.get("filled_event_lap_deletions", {}).values()
                               for x in v)
                    else "a race-control lap deletion names a kept lap: check the fill."))
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
    L.append("Truth drawn from the model's generative process on the real entries, teams and segments "
             "(including the drivers entered without a valid lap). Coverage = share of true values "
             "inside the 90% interval.\n")
    L.append(f"**Independent clean truths ({seeds['n_seeds']}).** Each has its own truth, its own noise and "
             "its own hyperparameters: a different posterior draw from the main fit, so settings such as "
             "the size of team-specific effects vary between truths. The gates use the means.\n")
    rows = []
    for k, r in synth["seeds"].items():
        rows.append({"truth": k, "sd_team_effect": r["hyper"].get("sd_compat", float("nan")),
                     "skill_cov90": r["skill"]["cov90"], "car_cov90": r["car"]["cov90"],
                     "grid_spearman": r["current_grid"]["spearman"], "grid_cov90": r["current_grid"]["cov90"],
                     "grid_in_team_spearman": r["current_grid_in_team"]["spearman"],
                     "grid_in_team_cov90": r["current_grid_in_team"]["cov90"],
                     "cross_team_cov90": r["cross_team_diff_current"]["cov90"],
                     "divergences": r["divergences"]})
    L.append(md_table(pd.DataFrame(rows)))
    L.append("")
    L.append("Mean (range): " + "; ".join(f"{k} {_mean_range(seeds, k)}" for k in (
        "skill_cov90", "car_cov90", "grid_spearman", "grid_cov90", "grid_in_team_spearman",
        "grid_in_team_cov90")) + ". Eight truths show the spread between truths; they do not "
             "pin down coverage to better than a few percentage points.\n")
    L.append("**Misspecification scenarios (one shared truth).** These add effects the model does not "
             "contain. All share one simulated truth (posterior medians), so they compare scenarios on "
             "equal terms but each is a single draw.\n")
    rows = []
    for sc, r in synth["scenarios"].items():
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

    L.append("### The team-specific effect\n")
    multi = ent.groupby("stint_idx").spell_idx.nunique()
    multi = multi[multi > 1]
    examples = [f"{g.driver_id.iloc[0]} ({g.team.iloc[0]}: "
                + ", ".join(f"{a[:4]}–{b[:4]}" if a[:4] != b[:4] else a[:4]
                            for a, b in g.groupby("spell_idx").event_id.agg(["min", "max"]).values) + ")"
                for _, g in ent[ent.stint_idx.isin(multi.index[:3])].groupby("stint_idx")]
    placebo_path = VAL / "placebo_summary.json"
    if placebo_path.exists():
        pl = json.loads(placebo_path.read_text())
        L.append("**Placebo test.** "
                 f"Team-specific effect SD {pl['sd_compat_q05_q50_q95'][1]:.3f}% "
                 f"(90% interval {pl['sd_compat_q05_q50_q95'][0]:.3f}–{pl['sd_compat_q05_q50_q95'][2]:.3f}) "
                 f"vs placebo SD {pl['sd_placebo_q05_q50_q95'][1]:.3f}% "
                 f"({pl['sd_placebo_q05_q50_q95'][0]:.3f}–{pl['sd_placebo_q05_q50_q95'][2]:.3f}) "
                 f"when a driver's stint in one team is split in half. P(placebo > team effect) = "
                 f"{pl['p_placebo_gt_compat']:.2f}. A large team-specific SD with a small placebo SD means "
                 "relative performance shifts when a driver changes team, much more than with time spent "
                 "in the same team. The test cannot say why (car handling, team support, role, adaptation "
                 "or selection).\n")
    L.append("**What one effect covers.** The main model gives one effect per driver and team lineage, so "
             f"separate spells with the same lineage share it ({len(multi)} cases, e.g. "
             f"{'; '.join(examples)}). Two variants test this choice: one effect per spell (a return "
             "after 3+ events with other teams starts a new one; a one-off stand-in drive does not) and "
             "one per regulation era (2014, 2017, 2022 and 2026 start new ones).\n")
    trows = []
    for name, label in (("sens_team_spell", "per spell"), ("sens_team_era", "per era")):
        if name in sens:
            trows.append({"variant": label, "in_team_spearman": sens[name]["in_team_spearman"],
                          "in_team_max_shift_s": sens[name]["in_team_max_abs_shift_s"],
                          "portable_spearman": sens[name]["spearman"],
                          "portable_max_shift_s": sens[name]["max_abs_shift_s"]})
    if trows:
        L.append("Current grid under each variant vs the main model (all data):\n")
        L.append(md_table(pd.DataFrame(trows)))
        L.append("")
    frows = []
    for v, label in (("lfospell", "per spell"), ("lfoera", "per era")):
        if v in compare:
            frows.append(_paired_row(f"{label}: all pairings", compare[v]["pairings"]))
            frows.append(_paired_row(f"{label}: first season after a reset", compare[v]["pairings_after_reset"]))
    if frows:
        cuts = compare.get("lfoera", compare.get("lfospell"))["cutoffs"]
        L.append(f"Forecasts of teammate gaps per pairing, variant vs main model on identical targets at "
                 f"{len(cuts)} cutoffs ({', '.join(cuts)}). `mse_diff_s2`: mean difference in squared "
                 "error (variant minus main, s²; negative = variant better), with a bootstrap 95% "
                 "interval over pairings (pairings at one cutoff share a fit, so it is somewhat too "
                 "narrow).\n")
        L.append(md_table(pd.DataFrame(frows)))
        L.append("")
        better = [label for v, label in (("lfospell", "per spell"), ("lfoera", "per era"))
                  if v in compare and compare[v]["pairings"]["mse_diff_ci95"][1] < 0]
        L.append(("Forecasts favour one effect " + " and ".join(better) + " slightly (interval below zero, but "
                  "too narrow as noted). The main model keeps one effect per lineage in this version; switching "
                  "is a change for the next model version, with its own refit and validation.\n") if better else
                 "Neither variant forecasts clearly better than one effect per lineage.\n")

    L.append("### Sensitivity of the current driver ranking\n")
    L.append(md_table(pd.DataFrame([{"variant": k, "in_team_spearman": v["in_team_spearman"],
                                     "in_team_max_shift_s": v["in_team_max_abs_shift_s"],
                                     "in_team_max_rank_shift": v["in_team_max_rank_shift"],
                                     "portable_spearman": v["spearman"],
                                     "portable_max_shift_s": v["max_abs_shift_s"],
                                     "portable_max_rank_shift": v["max_rank_shift"]}
                                    for k, v in variants.items()])))
    L.append("")
    L.append("Per-driver pace in the current car (s) under each variant:\n")
    L.append(md_table(sens_in_team.reset_index().rename(columns={"index": "driver_id"})))
    L.append("")
    L.append("Per-driver portable skill (s) under each variant:\n")
    L.append(md_table(sens_table.reset_index().rename(columns={"index": "driver_id"})))
    L.append("")

    if "lfo2006" in compare:
        wc = compare["lfo2006"]
        L.append("### Data window: 2010 vs 2006 start\n")
        L.append(f"Same model fitted from 2006 (adding 2006–09 Q1/Q2 low-fuel times) vs from 2010, on "
                 f"identical forecast targets at {len(wc['cutoffs'])} cutoffs ({', '.join(wc['cutoffs'])}).\n")
        L.append(md_table(pd.DataFrame([_paired_row("2006 vs 2010: all pairings", wc["pairings"]),
                                        _paired_row("2006 vs 2010: new pairings", wc["pairings_new"])])))
        L.append("")
        L.append(md_table(pd.DataFrame([{"window": w, **wc[f"sessions_{k}"]}
                                        for w, k in (("2010", "main"), ("2006", "var"))])))
        L.append("")

    if diag:
        tc = diag["teammate_residual_corr"]
        q = diag["standardised_residual_quantiles"]
        L.append("### Residual checks (main fit)\n")
        L.append(f"- Correlation of teammates' standardised residuals in the same segment: {tc['all']:.3f} "
                 f"(Q1 {tc.get('Q1', float('nan')):.3f}, Q2 {tc.get('Q2', float('nan')):.3f}, "
                 f"Q3 {tc.get('Q3', float('nan')):.3f}; n = {tc['n']:,} pairs). Clearly positive values "
                 "would mean shared car effects within a segment are not fully captured; negative values "
                 "arise when a segment's two residuals are pulled apart by the shared car-segment effect.")
        L.append(f"- Lag-1 autocorrelation of a pairing's teammate-gap residual across events: "
                 f"{diag['teammate_gap_residual_lag1']['corr']:.3f} (n = {diag['teammate_gap_residual_lag1']['n']:,}). "
                 "Positive values would mean relative form persists beyond what the skill walk captures.")
        L.append(f"- Standardised residual quantiles 1/5/50/95/99%: {q['q01']:.2f} / {q['q05']:.2f} / "
                 f"{q['q50']:.2f} / {q['q95']:.2f} / {q['q99']:.2f}. The slow side has the heavier tail "
                 "(compromised laps); the variant with a wider slow side (`sens_splitt`) changes the rankings "
                 f"little (rank correlation {sens['sens_splitt']['in_team_spearman']:.2f} in the current car, "
                 f"{sens['sens_splitt']['spearman']:.2f} portable).\n")

    L.append("### Convergence\n")
    L.append(f"{conv['n_draws']} posterior draws; divergences: {conv['info']['divergences']}; "
             f"max R-hat skill {conv['skill']['rhat_max']:.3f}, team-specific effect "
             f"{conv['compat']['rhat_max']:.3f}, car {conv['car']['rhat_max']:.3f}, hyperparameters "
             f"{conv['hyper_rhat_max']:.3f}; min ESS skill {conv['skill']['ess_min']:.0f}, team-specific "
             f"effect {conv['compat']['ess_min']:.0f}, car {conv['car']['ess_min']:.0f}.\n")

    L.append("## What the ratings can and cannot say\n")
    sens_vals = lambda keys, k="spearman": [sens[x][k] for x in keys if x in sens]  # noqa: E731
    sc_grid = [r["current_grid"]["spearman"] for k, r in synth["scenarios"].items() if k != "clean"]
    minor = sens_vals(("sens_notrack", "sens_gausscar", "sens_notransient", "sens_splitt"))
    L.append("\n".join([
        "- **Car-package ratings are well determined in simulation.** Recovery correlation "
        f"{_mean_range(seeds, 'car_corr', 3)} over {seeds['n_seeds']} independent truths, 90% coverage "
        f"{_mean_range(seeds, 'car_cov90')}. In the simulated truth, car states vary far more than "
        "portable skill states within a season; that describes the simulation, not a share of observed "
        "qualifying variation.",
        "- **Teammate comparisons and forecasts** beat a static two-way model, raw teammate gaps and a "
        "zero-gap baseline on every forecast target above, including brand-new pairings, with 90% "
        f"intervals covering {ts['cov90']:.0%} per segment.",
        f"- **Pace in the current car (headline).** Current-grid rank correlation in simulation "
        f"{grid_in_team_rho}; against the sensitivity variants its rank correlation is "
        f"{min(v['in_team_spearman'] for v in variants.values()):.2f}–"
        f"{max(v['in_team_spearman'] for v in variants.values()):.2f}.",
        f"- **Portable skill is experimental.** Current-grid rank correlation in simulation {grid_rho}, "
        f"and against the sensitivity variants {min(v['spearman'] for v in variants.values()):.2f}–"
        f"{max(v['spearman'] for v in variants.values()):.2f}. Performance relative to teammates has a "
        "large team-specific part, which limits how well the network separates portable skill. Read "
        "portable ranks as ranges.",
        "- **The team-specific effect is associated with the team, not proven to be car compatibility.** "
        "Support, role, adaptation or selection would produce the same pattern.",
        "- **Drivers who have only raced for one team** (e.g. Piastri, Antonelli, Bortoleto): their "
        "portable skill depends on the model's assumption about how team-specific effects are "
        "distributed. Simulation follows the model's own assumptions, so it cannot check this.",
        "- **Structural sensitivities of portable skill:** whether the team-specific effect is modelled "
        f"({sens['sens_nocompat']['spearman']:.2f}), how much history is used "
        f"({sens['sens_start2006']['spearman']:.2f}), what one team-specific effect covers (per spell "
        f"{sens['sens_team_spell']['spearman']:.2f}, per era {sens['sens_team_era']['spearman']:.2f}); "
        "the weekend-form term and the assumed rate of skill drift "
        f"({min(sens_vals(('sens_noform', 'sens_drift_x2', 'sens_drift_half'))):.2f}–"
        f"{max(sens_vals(('sens_noform', 'sens_drift_x2', 'sens_drift_half'))):.2f}). Details of the car "
        f"and noise model matter least (≥ {min(minor):.2f}).",
        "- **Misspecification scenarios** (extra team-specific effects, unequal upgrades, persistent form, "
        "lucky seasons before a team change) give portable current-grid rank correlations of "
        f"{min(sc_grid):.2f}–{max(sc_grid):.2f} on their one shared truth, inside the spread between "
        f"independent clean truths ({seeds['grid_spearman']['min']:.2f}–{seeds['grid_spearman']['max']:.2f}). "
        "Single draws cannot rank these violations, and none of them falls outside that spread.",
        "- **Scope:** one-lap qualifying pace. Racing qualities are in the Racing section below.",
    ]))
    L.append("")
    L += racing_section()
    (OUT / "REPORT.md").write_text("\n".join(L))
    (VAL / "gates.json").write_text(json.dumps(gates, indent=1))
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
