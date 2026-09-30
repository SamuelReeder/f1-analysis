"""The racing (stage 2) section of outputs/REPORT.md, built from the racing modules' outputs.

Every number comes from a file under outputs/ or data/ written by the module named next to
it; a part whose outputs are missing is left out.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SEC_PER_PCT = 0.9  # seconds per percent of a 90 s lap
STALE = set()


def _load(rel: str):
    p = OUT / rel
    if not p.exists():
        return None
    manifest = None
    if p.parent.name in ("championship", "benchmark", "firstlap", "consistency", "reliability", "pitstops", "wet"):
        manifest = "manifest.json"
    elif p.parent.name == "battles" and p.name == "summary.json":
        manifest = "manifest.json"
    elif p.name.startswith("multi_heldout"):
        manifest = p.stem + ".manifest.json"
    elif p.name.startswith("multi_summary"):
        manifest = p.stem.replace("multi_summary", "multi_full") + ".manifest.json"
    if manifest:
        from .artifacts import StaleArtifact, require
        try:
            require(p.parent, name=manifest, required_outputs=[p])
        except StaleArtifact:
            STALE.add(rel)
            return None
    return json.loads(p.read_text()) if p.exists() else None


def md(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in df.values])


def ci(r: dict, nd: int = 3, key="mean_diff_per_race") -> str:
    return f"{r[key]:+.{nd}f} (95% interval {r['ci95'][0]:+.{nd}f} to {r['ci95'][1]:+.{nd}f})"


def mse(r: dict) -> str:
    lo, hi = r["mse_diff_ci95"]
    f = lambda x: f"{x:+.4f}" if abs(x) >= 1e-3 or x == 0 else f"{x:+.1e}"  # noqa: E731
    return f"{f(r['mse_diff'])} (95% interval {f(lo)} to {f(hi)}; {r['n_pair_seasons']} pair-seasons)"


def q3(v: list, nd: int = 3) -> str:
    return f"{v[1]:.{nd}f} (90% interval {v[0]:.{nd}f}–{v[2]:.{nd}f})"


def verdict(ok) -> str:
    return "**passes**" if ok else "**does not pass**"


# --------------------------------------------------------------------------- summary

def summary_table() -> list[str]:
    rows = []
    mh, rel, pit = _load("race/multi_heldout.json"), _load("reliability/summary.json"), _load("pitstops/summary.json")
    con, wet, fl = _load("consistency/summary.json"), _load("wet/summary.json"), _load("firstlap/summary.json")
    ov, fe, ch = _load("battles/summary.json"), _load("battles/feasibility.json"), _load("championship/summary.json")
    bm = _load("benchmark/summary.json")
    entry = (ch or {}).get("entry_tests", {})

    def ent(name):
        e = entry.get(name)
        if not e or "not_run" in e:
            return "not tested (no held-out draws)"
        return ("enters" if name in (ch or {}).get("qualities_entered", []) else "does not enter") + f" ({e['mean_diff_per_race']:+.3f}, " \
               f"{e['ci95'][0]:+.3f} to {e['ci95'][1]:+.3f})"

    if mh:
        rows.append({"quality": "race-specific pace (beyond the qualifying link)", "kind": "driver",
                     "own gate": "pass" if mh["gate_race_specific_pace"] else "fail",
                     "overall rating": ent("race_specific_pace")})
        rows.append({"quality": "tyre degradation", "kind": "driver",
                     "own gate": "pass" if mh["gate_degradation"] else "fail", "overall rating": ent("degradation")})
    if con:
        rows.append({"quality": "consistency (lap-time spread)", "kind": "driver",
                     "own gate": "pass" if con["gate_driver_ranking"] else "fail", "overall rating": ent("consistency")})
    if wet:
        rows.append({"quality": "wet-weather pace", "kind": "driver",
                     "own gate": "pass" if wet["gate_wet_ranking"] else "fail",
                     "overall rating": "not tested (the championship simulates dry races)"})
    if rel:
        rows.append({"quality": "error rate (own-error retirements)", "kind": "driver",
                     "own gate": "pass" if rel["gate_driver_error_ranking"] else "fail",
                     "overall rating": "enters" if rel["gate_driver_error_ranking"] else
                     "does not enter (its own held-out test is the incident stage's entry test)"})
    if fl:
        rows.append({"quality": "first-lap performance", "kind": "driver",
                     "own gate": "pass" if fl["gate_driver_ranking"] else "fail", "overall rating": ent("first_lap")})
    if fe:
        own = "not feasible" if not fe["feasible"] else ("pass" if (ov or {}).get("gate_driver_ranking") else "fail")
        rows.append({"quality": "overtaking and defending", "kind": "driver", "own gate": own,
                     "overall rating": ent("overtaking_attack") + "; defending: " + ent("overtaking_defend")
                     if fe["feasible"] else "not tested (no held-out test)"})
        if (ov or {}).get("heldout_kind") == "attacker_only":
            rows.append({"quality": "overtaking, attacker only (post hoc)", "kind": "driver",
                         "own gate": "pass" if ov["gate_attacker_ranking_post_hoc"] else "fail",
                         "overall rating": ent("overtaking_attack")})
    if bm:
        r = bm["ratings_results_minus_ratings"]
        rows.append({"quality": "unexplained results effect", "kind": "driver",
                     "own gate": "pass" if r["ci95"][0] > 0 else "fail",
                     "overall rating": "not in the simulation" if r["ci95"][0] > 0 else
                     "does not enter (its own test is on held-out finishing orders)"})
    if rel:
        rows.append({"quality": "mechanical reliability (team-season)", "kind": "team",
                     "own gate": "pass" if rel["gate_team_reliability"] else "fail", "overall rating": "equalised"})
        if "gate_supplier_reliability" in rel:
            rows.append({"quality": "mechanical reliability (power-unit supplier)", "kind": "team",
                         "own gate": "pass" if rel["gate_supplier_reliability"] else "fail",
                         "overall rating": "equalised"})
    if pit:
        rows.append({"quality": "pit stops (team operations)", "kind": "team",
                     "own gate": "pass" if pit["gate_team_ops_rating"] else "fail", "overall rating": "equalised"})
    if not rows:
        return []
    return ["### Summary of the racing gates\n",
            "`own gate`: the quality's standalone test on held-out data (baselines keep the car and context terms). "
            "`overall rating`: the nested entry test of the equal-car championship's race stage (held-out "
            "finishing orders, with vs without the quality, carrying its posterior draws; mean difference in log "
            "predictive density per race and its 95% interval). A quality can fail its own gate and still enter. "
            "Team qualities are set to the average in the equal-car championship.\n",
            md(pd.DataFrame(rows)), ""]


# --------------------------------------------------------------------------- data

def data_sources() -> list[str]:
    L = ["### Racing data\n"]
    dc, oc, cc = _load("analysis/oldlaps_dump_check.json"), _load("analysis/oldlaps_check.json"), \
        _load("analysis/conditions_check.json")
    rs = ROOT / "data" / "processed" / "race_sources.json"
    src = json.loads(rs.read_text()) if rs.exists() else {}
    L.append(f"- **FastF1 (2018 onward):** {len(src.get('races', {}))} races, {len(src.get('sprints', {}))} sprint "
             "races: lap times, positions, tyres, track status, race-control messages, weather.")
    if dc:
        L.append(f"- **Jolpica lap-by-lap data (2010-2017):** from Jolpica's CSV database dump "
                 f"(uploaded {dc['dump']['delayed_csv_2026-09-13.zip']['uploaded_at'][:10]}, sha256 matching the "
                 f"published value). Against the API pages on the {dc['races_compared']} races both have: "
                 f"{dc['laps'].get('both', 0):,} laps, lap times equal {dc['lap_time_equal']:.1%}, positions equal "
                 f"{dc['position_equal']:.1%}; pit stops on the same lap {dc['pit_lap_equal']:.1%}.")
    if oc:
        g, n, p = oc["gap_at_line"], oc["neutralised_laps"], oc["pit_laps_vs_fastf1"]
        L.append(f"- **What Jolpica lacks is inferred** (oldlaps.py), checked against FastF1 on {oc['races_compared']} "
                 f"races: positions agree {oc['position_agreement']:.1%}; a gap under 1 s at the line agrees "
                 f"{g['within_1s_agreement']:.1%}; neutralised laps (from the field's lap times) precision "
                 f"{n['precision']:.2f}, recall {n['recall']:.2f}; inferred pit laps precision {p['precision']:.2f}, "
                 f"recall {p['recall']:.2f}"
                 + (f" (against recorded stops 2011-2017: {oc['pit_laps_vs_recorded_2011_17']['precision']:.2f} / "
                    f"{oc['pit_laps_vs_recorded_2011_17']['recall']:.2f}; inference is used only for 2010)"
                    if "pit_laps_vs_recorded_2011_17" in oc else "")
                 + ". No tyre compounds or speed traps before 2018: those races enter through a weaker "
                   "observation model (stated per model).")
    if cc:
        L.append(f"- **Wet races before 2018** from hourly precipitation at the circuit (Open-Meteo archive, "
                 f"ERA5): {cc['rule']}. Against FastF1's wet flag on {cc['races_compared']} races: precision "
                 f"{cc['precision']:.2f}, recall {cc['recall']:.2f}, a weak proxy; models give proxy-wet races their "
                 "own coefficient. Threshold sweep in `outputs/analysis/conditions_check.json`.")
    pu = ROOT / "data" / "reference" / "power_units.csv"
    if pu.exists():
        t = pd.read_csv(pu)
        L.append(f"- **Power-unit suppliers:** {len(t)} team-seasons from each season's Wikipedia entry list "
                 "(powerunits.py; pages saved with hashes). One badge that names no supplier is set by hand "
                 "(2017 Toro Rosso: Renault).")
    tel = src.get("telemetry", {})
    if tel:
        bad = [e for e, v in tel.items() if v["speed_trap_median_abs_diff_kmh"] > 5]
        L.append(f"- **Telemetry:** per-lap coasting summaries for {len(tel)} dry races (extract/telemetry.py), "
                 f"aligned to the laps by matching FastF1's finish-line speed trap; {len(bad)} races miss it by "
                 f"more than 5 km/h ({', '.join(bad)}) and are not used.")
    return L + [""]


BASIS_LABEL = {"status code": "from a coded status", "inferred from evidence": "inferred from race evidence",
               "class base rates": "from class base rates (uncoded, no race evidence)"}


def timeline() -> list[str]:
    a = _load("timeline/audit.json")
    if not a:
        return []
    v = a["cause_model"]["validation"]
    ret = a["retirements"]
    return ["### Event timeline (outcomes from 2010, race evidence from 2018)\n",
            f"{a['rows']:,} rows over {a['races']} races: neutralisations, retirements and other outcomes, and "
            "race-control evidence (incidents, penalties, stoppages, off-track moments), plus pit anomalies, "
            "suspected damage and possible team orders. Rules produce evidence, not verdicts. Full audit: "
            "`outputs/timeline/AUDIT.md`.\n",
            f"- **Retirement causes.** {ret['n']} retirements: "
            + ", ".join(f"{n} {BASIS_LABEL.get(k, k)}" for k, n in ret["by_basis"].items())
            + ". Jolpica codes causes through 2022 but records almost every later retirement only as \"Retired\"; "
              "those get probabilities from a model of the cause class given race-control and lap evidence, fitted "
              f"on {v['n']} coded retirements. Held out a season at a time, its log loss is {v['log_loss_model']:.3f} "
              f"against {v['log_loss_base_rates']:.3f} for base rates (accuracy {v['accuracy_model']:.2f} vs "
              f"{v['accuracy_base_rates']:.2f}): informative, far from certain.",
            "- **Who caused an incident** is split by stated assumptions (the audit lists them), not estimates.", ""]


# --------------------------------------------------------------------------- qualities

def race_pace() -> list[str]:
    L = []
    mh, ms = _load("race/multi_heldout.json"), _load("race/multi_summary.json")
    if not mh:
        return L
    L += ["### Race pace and tyre degradation\n",
          "One joint lap-level model across seasons (racemulti.py): lap = race trend + compounds + degradation + "
          "dirty air + car (team x race) + gamma x qualifying pace (stage 1) + race-specific driver pace (lasting, "
          "per season, per race) + driver degradation, with AR(1) Student-t errors within stints. Laps before "
          "2018 have unknown compounds: a stint offset replaces the compound terms, with their own noise scale.\n"]
    if ms:
        L.append(f"Full fit: {ms['n_laps']:,} clean laps in {ms['n_races']} dry races "
                 f"({', '.join(f'{k} {v:,}' for k, v in ms['laps_by_source'].items())}), {ms['seasons'][0]}–"
                 f"{ms['seasons'][1]}; gamma {q3(ms['gamma_q05_q50_q95'], 2)}; SD of lasting race-specific pace "
                 f"{q3(ms['sd_u_q05_q50_q95'])}% of lap time; max R-hat {ms['rhat_max']:.3f}, "
                 f"{ms['divergences']} divergences.\n")
    fits = mh["fits"]
    worst = max(v["_rhat_max"] for v in fits.values())
    retried = sorted(s for s, v in fits.items() if len(v["_attempts"]) > 1)
    L.append(f"Held out ({mh['design']}; {len(fits)} training fits, max R-hat {worst:.3f}"
             + (f", {len(retried)} of them ({', '.join(retried)}) on the second attempt of the fixed retry rule "
                "(new seed, doubled warmup)" if retried else "")
             + "; squared error in %², negative = better, bootstrap interval over pair-seasons):\n")
    rows = [{"test": "qualifying link vs zero", "result": mse(mh["quali_link_vs_zero"])},
            {"test": "+ race-specific pace vs qualifying link", "result": mse(mh["race_specific_pace_vs_quali_link"])},
            {"test": "degradation effects vs zero", "result": mse(mh["degradation_vs_zero"])}]
    L += [md(pd.DataFrame(rows)), "",
          f"- **Race-specific pace** {verdict(mh['gate_race_specific_pace'])} its gate (interval below zero)."
          + ("" if mh["gate_race_specific_pace"] else " Race pace is represented by the qualifying link."),
          f"- **Degradation** {verdict(mh['gate_degradation'])} its gate.", ""]
    era = mh.get("by_era_descriptive")
    if era:
        rows = [{"held out": k, "race-specific pace vs qualifying link": mse(v["race_specific_pace_vs_quali_link"]),
                 "degradation vs zero": mse(v["degradation_vs_zero"])} for k, v in era.items()]
        L += ["The same held-out predictions split by era (descriptive, added after the gate result; the gates are "
              "the pooled intervals above):\n", md(pd.DataFrame(rows)), ""]
        e = list(era.values())
        splits = [(q, [x[q]["improves"] for x in e]) for q in ("race_specific_pace_vs_quali_link", "degradation_vs_zero")]
        if any(len(set(s)) > 1 for _, s in splits):
            L.append("Neither era alone carries both results: where an interval excludes zero in one era it does "
                     "not in the other. Before 2018 compounds are unknown, so a teammate difference in degradation "
                     "there can include a difference in compound choice.\n" if all(len(set(s)) > 1 for _, s in splits)
                     else "The eras disagree on at least one result (table above).\n")
    dpath = OUT / "race" / "multi_drivers.csv"
    if ms and dpath.exists() and (mh["gate_race_specific_pace"] or mh["gate_degradation"]):
        D = pd.read_csv(dpath)
        names = pd.read_parquet(ROOT / "data" / "processed" / "drivers.parquet").set_index("driver_id").name
        D["name"] = D.driver_id.map(names).fillna(D.driver_id)
        for gate, col, label, note in (
                ("gate_race_specific_pace", "race_specific_pct", "race-specific pace",
                 "lasting race pace beyond the qualifying link, % of lap time (positive = faster; "
                 f"0.1% is about {0.1 * SEC_PER_PCT:.2f} s on a 90 s lap)"),
                ("gate_degradation", "degradation_pct_per_lap", "degradation",
                 "lasting change in pace per lap of tyre age, % of lap time (positive = loses less pace as "
                 "the tyres age)")):
            if not mh[gate]:
                continue
            T = D.sort_values(f"{col}_median", ascending=False).head(15)
            T = pd.DataFrame({"driver": T.name, "median": T[f"{col}_median"].map(lambda x: f"{x:+.4f}"),
                              "90% interval": [f"{a:+.4f} to {b:+.4f}" for a, b in zip(T[f"{col}_q05"], T[f"{col}_q95"])],
                              "seasons": T.seasons, "clean laps": T.clean_laps})
            L += [f"Top 15 by {label} (full fit, {ms['seasons'][0]}–{ms['seasons'][1]}; {note}; identified only from "
                  "differences between teammates, linked across teams by drivers who move; all "
                  f"{len(D)} drivers in `outputs/race/multi_drivers.csv`; ordered by median, and most intervals "
                  "overlap):\n", md(T), ""]
    # variants
    rows = []
    for sfx, label in (("_fastf1only", "2018 onward only (FastF1)"),
                       ("_fastf1only_telemetry", "races with aligned telemetry, no coasting term"),
                       ("_fastf1only_coast", "same races, laps and targets adjusted for coasting")):
        v = _load(f"race/multi_heldout{sfx}.json")
        if v:
            r = v["race_specific_pace_vs_quali_link"]
            rows.append({"variant": label, "race-specific pace vs qualifying link": mse(r),
                         "degradation vs zero": mse(v["degradation_vs_zero"])})
    if rows:
        L += ["Variants of the same held-out test:\n", md(pd.DataFrame(rows)), ""]
        coast = _load("race/multi_heldout_fastf1only_coast.json")
        if coast:
            b = [f["b_coast"] for f in coast["fits"].values() if "b_coast" in f]
            L.append(f"The coasting term (telemetry use 1: lift-and-coast seconds per lap) costs "
                     f"{-min(b):.2f}–{-max(b):.2f}% of lap time per second of coasting across the training fits; "
                     "adjusting for it " + ("changes" if coast["gate_race_specific_pace"] !=
                                            (_load("race/multi_heldout_fastf1only_telemetry.json") or {}).get(
                                                "gate_race_specific_pace") else "does not change")
                     + " the race-specific pace conclusion.\n")
    jc = _load("race/joint_check_2024.json")
    if jc:
        L.append(f"The earlier two-stage shortcut (per-race estimates, then a model across races) was not accepted "
                 f"by the joint-model check on {jc['season']} (rule: {jc['acceptance_rule']}); its driver numbers are "
                 "not published and this joint model replaces it.\n")
    return L


def reliability() -> list[str]:
    r = _load("reliability/summary.json")
    if not r:
        return []
    h = r["heldout"]
    w = r["wet_fastf1_log_q05_q50_q95"]
    L = ["### Reliability and driver errors (2010 onward)\n",
         f"Competing risks per lap ({r['driver_races']:,} starts, {r['retirements']:,} retirements); uncertain causes "
         "count fractionally. Mechanical risk has era, power-unit supplier-season and team-season terms; own-error "
         f"risk has team-season, traffic and driver terms; every cause has wet-race terms ({r['wet_races']['fastf1']} "
         f"FastF1 wet races, {r['wet_races']['precipitation_proxy']} proxy-wet races before 2018). Traffic: "
         f"{q3(r['traffic_own_error_log_q05_q50_q95'], 2)} on the log own-error rate per unit share of laps in traffic; "
         f"wet races (FastF1) {q3(w['own_error'], 2)} own error, {q3(w['other_driver'], 2)} caused by another driver.\n",
         "Held-out seasons, paired log predictive density per race:\n",
         f"- **Driver own-error effects** (baseline keeps every other term): {ci(h['driver_effects'])}: "
         f"{verdict(r['gate_driver_error_ranking'])}.",
         f"- **Team-season mechanical effects** (second half of each season from its first half, supplier terms "
         f"kept): {ci(h['team_season_effects'])}: {verdict(r['gate_team_reliability'])}."]
    if "supplier_season_effects" in h:
        L.append(f"- **Power-unit supplier-season effects** (team-season terms kept): {ci(h['supplier_season_effects'])}: "
                 f"{verdict(r['gate_supplier_reliability'])}.")
    return L + [""]


def pit_stops() -> list[str]:
    p = _load("pitstops/summary.json")
    if not p:
        return []
    L = ["### Pit stops (team operations)\n",
         f"Pit-lane time relative to the race's median stop ({p['stops']:,} stops, {p['races']} races, "
         f"{p['seasons'][0]}–{p['seasons'][1]}; {', '.join(f'{k} {v:,}' for k, v in p['sources'].items())}); excluded: "
         f"{p['excluded_rule']}. Lasting team SD {q3(p['sd_team_q05_q50_q95'], 2)} s, team-season SD "
         f"{q3(p['sd_team_season_q05_q50_q95'], 2)} s; Student-t with very heavy tails (nu "
         f"{p['nu_q05_q50_q95'][1]:.2f}: slow stops). Held out (second half of each season from its first half): team "
         f"terms {ci(p['heldout_team_terms'])}: {verdict(p['gate_team_ops_rating'])}.\n"]
    t = ROOT / "outputs" / "pitstops" / "team_seasons.csv"
    if p["gate_team_ops_rating"] and t.exists():
        T = pd.read_csv(t)
        T = T[T.season == T.season.max()].sort_values("rel_s_median")
        L += [f"{int(T.season.max())}, seconds per stop relative to the race median (negative = faster):\n",
              md(T[["team", "rel_s_median", "rel_s_q05", "rel_s_q95", "stops", "share_slow_3s"]]), ""]
    return L


def consistency() -> list[str]:
    c = _load("consistency/summary.json")
    if not c:
        return []
    return ["### Consistency\n",
            f"Each driver's robust spread of clean-lap residuals in the stage-A regression, compared with the teammate's "
            f"({c['teammate_races']:,} teammate races in {c['races']} races; median spread {c['median_spread_pct']:.2f}% of "
            f"lap time). Driver SD on the log scale {q3(c['sd_driver_log_q05_q50_q95'])}; split-half correlation of "
            f"driver means {c['split_half_corr_of_driver_means']:.2f}. Held out against no driver effect: "
            f"{mse(c['heldout_vs_zero'])}: {verdict(c['gate_driver_ranking'])}.\n"]


def wet() -> list[str]:
    w = _load("wet/summary.json")
    if not w:
        return []
    h = w["heldout_wet_effects_vs_quali_link"]
    return ["### Wet-weather pace (experimental)\n",
            f"Laps on intermediates or wets, each compared with the field on the same lap ({w['wet_teammate_races']} "
            f"teammate races in {w['wet_races']} wet races, 2018 onward). Wet gaps follow qualifying gaps with gamma "
            f"{q3(w['gamma_q05_q50_q95'], 2)} (larger than in the dry). A driver-specific wet effect, held out against "
            f"the qualifying link: squared error {h['sq_err_diff_per_race']:+.3f} per race (95% interval "
            f"{h['ci95'][0]:+.3f} to {h['ci95'][1]:+.3f}; {h['n_races']} races): {verdict(w['gate_wet_ranking'])}.\n"]


def first_lap() -> list[str]:
    f = _load("firstlap/summary.json")
    if not f:
        return []
    h = f["heldout_driver_effects"]
    o = f["no_lasting_team"]["heldout_driver_effects"]
    L = ["### First-lap performance\n",
         f"Positions gained from the grid slot to the end of lap 1 ({f['starts']:,} starts: {f['races']} races from "
         f"{f['seasons'][0]}, {f['sprints']} sprint races). Slot, side of the grid, start tyre, sprint, a lasting team "
         f"term, team-season car and driver effects. Driver SD {q3(f['sd_driver_q05_q50_q95'], 2)} places, lasting team "
         f"SD {q3(f['sd_team_q05_q50_q95'], 2)}, team-season SD {q3(f['sd_car_q05_q50_q95'], 2)}.\n",
         "Held-out seasons (paired log predictive density per race or sprint):\n",
         md(pd.DataFrame([
             {"model": "with lasting team term (final)", "driver effects, all": ci(h),
              "drivers who changed team": ci(h["transfer"]) + f"; {h['transfer']['n_starts']} starts",
              "drivers who stayed": ci(h["stayed"])},
             {"model": "without it (first build's structure)", "driver effects, all": ci(o),
              "drivers who changed team": ci(o["transfer"]), "drivers who stayed": ci(o["stayed"])}])), "",
         f"The lasting team term itself (no driver effects in either model): "
         f"{ci(h['lasting_team_vs_none_without_drivers'])}. 90% predictive intervals cover {h['cov90']:.0%}.\n",
         f"- **Standalone ranking gate** (overall interval above zero, interval for drivers in a new team not entirely "
         f"below zero; the second condition was written down after the first build's result): "
         f"{verdict(f['gate_driver_ranking'])}. The lasting team term was added after the first build failed; the "
         "decision rests on the final run with 2010-2017 starts, fixed in advance "
         "(docs/racing_approach.md, decisions fixed before the final runs).", ""]
    return L


def overtaking() -> list[str]:
    fe, o = _load("battles/feasibility.json"), _load("battles/summary.json")
    if not fe:
        return []
    a = fe["at_estimated_size"]
    L = ["### Overtaking and defending\n"]
    if o:
        L.append(f"{o['episodes']:,} battle episodes and {o['passes']:,} passes (by source: "
                 + ", ".join(f"{k} {v:,} episodes" for k, v in o.get("episodes_by_source", {}).items()) + ").")
    L.append(f"Synthetic feasibility at the estimated effect sizes ({fe['criteria']}): attacker correlation "
             f"{a['attack']['corr']:.2f}, coverage {a['attack']['cov90']:.0%}; defender correlation "
             f"{a['defend']['corr']:.2f}, coverage {a['defend']['cov90']:.0%}: "
             f"**{'feasible' if fe['feasible'] else 'not feasible'}**.")
    if o and "heldout_driver_effects" in o:
        t = o["heldout_driver_effects"]
        L.append(f"Held out from {o.get('heldout_first_season', '')}: {ci(t)}; drivers in a new team {ci(t['transfer'])}: "
                 f"{verdict(o['gate_driver_ranking'])}.")
    elif o and "heldout_attacker_only_post_hoc" in o:
        t = o["heldout_attacker_only_post_hoc"]
        L.append(f"Defender effects are not recoverable, so attacker and defender effects are not rated together. "
                 f"Attacker effects alone meet the same feasibility criteria; a model with attacker effects only "
                 f"(a rule chosen after the defender result, so **post hoc**), held out from "
                 f"{o['heldout_first_season']} against no driver effects: {ci(t)}; attackers in a new team "
                 f"{ci(t['transfer'])}: {verdict(o['gate_attacker_ranking_post_hoc'])} (same rule as the joint test).")
        if not o["gate_attacker_ranking_post_hoc"]:
            L.append("Not rated: reported as opportunity counts (`outputs/battles/drivers.csv`).")
    elif o:
        L.append("Not rated: reported as opportunity counts (`outputs/battles/drivers.csv`).")
    return L + [""]


def benchmark() -> list[str]:
    b = _load("benchmark/summary.json")
    if not b:
        return []
    rows = [{"model": v, "log_lik_per_race": x["mean_log_lik_per_race"], "spearman": x["mean_spearman"],
             "teammate_h2h": x["teammate_h2h_accuracy"]} for v, x in b["per_variant"].items()]
    return ["### Results benchmark\n",
            f"Rank-ordered logit on full classifications, held-out seasons {b['seasons_held_out'][0]}–"
            f"{b['seasons_held_out'][1]} ({b['races']} races):\n", md(pd.DataFrame(rows)), "",
            f"- A driver results effect over the stage-1 ratings (the unexplained contribution): "
            f"{ci(b['ratings_results_minus_ratings'])}.",
            f"- Grid + ratings vs grid alone: {ci(b['grid_ratings_minus_grid'])}." if "grid_ratings_minus_grid" in b
            else "", ""]


def championship() -> list[str]:
    cs = _load("championship/summary.json")
    st_path = OUT / "championship" / "standings.csv"
    if not cs or not st_path.exists():
        return []
    st = pd.read_csv(st_path)
    h = cs["heldout_race_stage"]
    entered = cs.get("qualities_entered", [])
    rows = []
    for name, e in cs.get("entry_tests", {}).items():
        if "not_run" in e:
            rows.append({"quality": name, "held-out seasons": "", "held-out races": "", "result": e["not_run"],
                         "enters": "no"})
        else:
            rows.append({"quality": name, "held-out seasons": "–".join(map(str, e.get("seasons", []))),
                         "held-out races": str(e["n_races"]), "result": ci(e),
                         "enters": "yes" if e["enters"] else "no"})
    top = st[st.version == "in_team"].head(12).assign(
        ranks=lambda x: x.rank_lo.astype(str) + "–" + x.rank_hi.astype(str))
    L = ["### Equal-car championship\n",
         f"{cs['simulated_seasons']:,} simulated {cs['n_races_simulated']}-race seasons with equal cars, each from one "
         "posterior draw: qualifying (stage 1, pace in the current car, with weekend form and session noise) -> grid "
         "-> race given the grid (ranking model fitted on real finishing orders) -> retirements (reliability model, "
         "equal mechanical risk, driver error rates "
         f"{'used' if cs['driver_error_rates_used'] else 'at the average'}) -> points. Race stage on held-out "
         f"finishing orders: vs grid alone {ci(h['vs_grid'])}; vs ratings alone {ci(h['vs_ratings'])}.\n",
         "Conditional entry tests: the combined model with each quality removed in turn; backward removal "
         "retests the remaining qualities. Qualifying features are frozen before each held-out season. "
         "The entire selection procedure also has an outer test; its gate must pass before any quality enters.\n",
         md(pd.DataFrame(rows)), "",
         "Outer validation of selection vs no racing qualities: "
         + ci(cs["combined_validation"]) + "; " + verdict(cs["combined_validation"]["gate"]) + ".\n",
         "Qualities in the simulation: qualifying pace" + (", " + ", ".join(entered) if entered else
                                                            " only (no racing quality entered)") + ".\n",
         md(top[["name", "team", "points_per_race", "p_title", "ranks"]] if "team" in top else
            top[["name", "points_per_race", "p_title", "ranks"]]), ""]
    cb = OUT / "championship" / "contributions.csv"
    if cb.exists():
        C = pd.read_csv(cb).head(12)
        cols = [c for c in C.columns if c.startswith("loss_")]
        L += ["Contribution breakdown (expected points per race lost when a quality is set to the field average, "
              "the others kept; and all at once). Conditional model estimates: correlated qualities have no unique "
              "allocation, and setting a quality to the average in every draw also removes its uncertainty, which "
              "shifts expected points slightly through the convex points scale.\n",
              md(C[["name", "points_per_race"] + cols]), ""]
    return L


def section() -> list[str]:
    STALE.clear()
    L = ["## Racing (stage 2)\n",
         "Every racing quality has a standalone test on held-out data and, where it has held-out draws, the "
         "equal-car championship's entry test. Racing outcomes run from 2010; lap-level evidence from 2018 "
         "(FastF1), and from 2010 through Jolpica's lap data with a weaker observation model.\n"]
    for part in (summary_table, data_sources, timeline, race_pace, reliability, pit_stops, consistency, wet, first_lap,
                 overtaking, benchmark, championship):
        L += part()
    if STALE:
        L.insert(1, "**Racing validation needs regeneration.** Outputs without current provenance are excluded "
                 "from this report and cannot enter the championship. Earlier pass/fail results and equal-car "
                 "standings are historical results, not validated results of the corrected pipeline. "
                 "See README, Review fixes and regeneration.\n\nAffected files: "
                 + ", ".join(f"`{s}`" for s in sorted(STALE)) + ".\n")
    return L
