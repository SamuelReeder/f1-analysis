"""Exploratory equal-car simulation from converged current-data fits.

This is a provisional research preview, not a passing validation or a production
Overall result. It preserves every failed study and never writes dashboard data.
Run as a memory-capped systemd service.
"""
import html
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from f1rank import artifacts, championship as c
from f1rank.design import build_design
from f1rank.fit import FITS, load
from f1rank.qualifying import dependencies

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/championship_preview"
CACHE = ROOT / "outputs/fits/championship_preview"
PLAN = ROOT / "docs/championship_preview.md"
QUALITY = "race_specific_pace"


def field_quality(drivers, draws, ids, picks):
    """Original championship rule: an unseen driver gets the current field mean."""
    column = pd.Series(np.arange(len(drivers)), index=drivers).reindex(ids).to_numpy()
    known = ~np.isnan(column)
    if not known.any():
        raise ValueError("No current drivers have usable race-pace draws")
    value = np.zeros((len(picks), len(ids)))
    value[:, known] = draws[picks][:, column[known].astype(int)]
    value[:, ~known] = value[:, known].mean(1, keepdims=True)
    return value - value.mean(1, keepdims=True)


def write_html(summary, table):
    headline = table[table.version == "in_team"]
    rows = "".join(
        f"<tr><td>{rank}</td><th scope='row'>{html.escape(row.name)}</th>"
        f"<td>{row.points_per_race:.2f}</td><td>{row.p_title:.1%}</td>"
        f"<td>{row.rank_lo}–{row.rank_hi}</td></tr>"
        for rank, row in enumerate(headline.itertuples(index=False), 1))
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exploratory equal-car preview</title><style>
:root{font-family:system-ui,sans-serif;color:#e5eaf0;background:#11151e}body{max-width:960px;margin:auto;padding:40px 20px}
h1{font-size:clamp(1.8rem,5vw,3rem);line-height:1.1}.status{color:#f3c775;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
p{max-width:78ch;line-height:1.6;color:#bdc8d8}.note{border-left:4px solid #e6aa45;padding:8px 20px;background:#20232c}
.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:28px 0}th,td{padding:13px 10px;border-bottom:1px solid #343c4a;text-align:right;font-variant-numeric:tabular-nums}
th:nth-child(2),tbody th{text-align:left}thead{color:#aebbd0;font-size:.85rem}tbody tr:nth-child(-n+3){background:#1c2830}
a{color:#a5d8ff}small{color:#9aa9bf}</style><main>
<div class="status">Exploratory · validation not established</div>
<h1>Equal-car driver preview</h1>
"""
    document += f"<p>Data through <strong>{html.escape(summary['data_as_of'])}</strong>. "
    document += "Provisional estimates from qualifying pace and the fixed career race-pace component, with average machinery and mechanical risk.</p>"
    document += "<div class='note'><p><strong>This is not a validated best-driver ranking.</strong> "
    document += "The original combined prediction test failed, and both archive studies stopped on sampling failures before scoring. "
    document += "This preview uses the separately converged current-data fits. Driver–team compatibility remains in these estimates; the production Overall view remains withheld.</p></div>"
    document += "<div class='scroll'><table><thead><tr><th>Order</th><th>Driver</th><th>Expected points / race</th><th>Simulated title share</th><th>Season rank range</th></tr></thead><tbody>"
    document += rows + "</tbody></table></div>"
    document += f"<p>Based on {summary['simulated_seasons']:,} simulated seasons of {summary['n_races_simulated']} races. "
    document += "Rank ranges span the 5th to 95th percentiles of simulated season outcomes; they include race variability and model uncertainty. "
    document += "Title shares are conditional on this simplified simulation, not forecasts of the real championship.</p>"
    document += "<p>Race pace is included as the fixed exploratory candidate, without claiming it passed the combined publication gate. "
    document += "The simulation does not reproduce passing, traffic, strategy or weather lap by lap. It does not establish universal or purely transferable driver ability.</p>"
    document += "<p><a href='standings.csv'>Download estimates</a> · <a href='summary.json'>Assumptions and validation status</a></p></main></html>"
    (OUT / "preview.html").write_text(document)


def main():
    if OUT.exists():
        raise FileExistsError("Preview already has a recorded outcome")
    started = time.monotonic()
    original = ROOT / "outputs/championship"
    artifacts.require(original, required_outputs=[original / "summary.json"])
    reference = json.loads((original / "summary.json").read_text())
    rel_dir = ROOT / "outputs/reliability"
    artifacts.require(rel_dir, required_outputs=[rel_dir / "summary.json", rel_dir / "drivers.csv"])
    pace_dir = ROOT / "outputs/race"
    artifacts.require(pace_dir, name="multi_full.manifest.json", required_outputs=[pace_dir / "multi_summary.json"])
    pace_summary = json.loads((pace_dir / "multi_summary.json").read_text())
    artifacts.require_convergence(pace_summary)
    drivers, draws = c.full_quality_draws(QUALITY)
    parents = []
    failures = {}
    for name in ("championship_archive", "championship_archive_conservative"):
        directory = ROOT / "outputs/analysis" / name
        artifacts.require(directory, name="failure.manifest.json", required_outputs=[directory / "failure.json"])
        failures[name] = json.loads((directory / "failure.json").read_text())
        parents.extend([directory / "failure.json", directory / "failure.manifest.json"])
    pace_path = ROOT / "outputs" / c.FULL_FILES[c.QUALITY_FILES[QUALITY][0]]
    inputs = [*artifacts.input_files(), *dependencies([None]), pace_path,
              pace_dir / "multi_summary.json", pace_dir / "multi_full.manifest.json",
              rel_dir / "summary.json", rel_dir / "drivers.csv", rel_dir / "manifest.json",
              original / "summary.json", original / "manifest.json", *parents, PLAN, Path(__file__)]
    before = {p: artifacts.digest(p) for p in inputs}
    OUT.mkdir(parents=True)
    try:
        F = c.finishers(c.race_orders())
        F = F[F.season >= c.ENTRY_FIRST_SEASON].copy()
        F["quality_0"] = F.driver_id.map(pd.Series(draws.mean(0), index=drivers)).fillna(0.0)
        # Original championship full-fit settings and its unchanged convergence ladder.
        race_post = c.fit_race(F, "grid_ratings_quality", warmup=800, samples=800)
        CACHE.mkdir(parents=True, exist_ok=True)
        checkpoint = CACHE / "race_stage.npz"
        np.savez_compressed(checkpoint, **race_post)
        c.jax.clear_caches()

        design = build_design(2010)
        post, _ = load(FITS / "main.npz", design)
        last = design.events.event_idx.max()
        rows = np.flatnonzero(design.entries.event_idx.to_numpy() == last)
        ids = design.entries.driver_id.to_numpy()[rows]
        flat = lambda k: post[k].reshape(-1, *post[k].shape[2:])
        skill, compat = flat("skill")[:, rows], flat("compat")[:, rows]
        rng = np.random.default_rng(0)
        pick = rng.choice(skill.shape[0], c.N_SEASONS, replace=True)
        form_sd = flat("sd_driver_form")[pick]
        noise_sd = np.sqrt(flat("sd_car_event")[pick] ** 2 + flat("sigma0")[pick] ** 2)
        race_pick = rng.integers(len(race_post["b_driver"]), size=c.N_SEASONS)
        rel = json.loads((rel_dir / "summary.json").read_text())
        per100 = {name: value[1] / 100 for name, value in rel["hazard_per_100_laps_later_laps"].items()}
        lap1 = {name: value[1] / 100 for name, value in rel["hazard_lap1_per_100_starts"].items()}
        laps = 57  # unchanged championship simulation assumption
        common = 1 - np.exp(-(sum(lap1.values()) + (laps - 1) * sum(per100.values())))
        retire = np.full(len(ids), common)
        if rel["gate_driver_error_ranking"]:
            errors = pd.read_csv(rel_dir / "drivers.csv").set_index("driver_id")
            multiplier = errors.own_error_rate_multiplier_median.reindex(ids).fillna(1.0).to_numpy()
            own = lap1["own_error"] + (laps - 1) * per100["own_error"]
            retire = 1 - np.exp(-(sum(lap1.values()) + (laps - 1) * sum(per100.values()) + own * (multiplier - 1)))
        quality = field_quality(drivers, draws, ids, rng.integers(len(draws), size=c.N_SEASONS))
        extra = race_post["b_quality"][race_pick, 0][:, None] * quality
        n_races = max(int((design.events.season == design.events.season.max()).sum()), 24)
        names = design.drivers.set_index("driver_id").name
        records = []
        for version, Q in (("in_team", skill + compat), ("portable", skill)):
            Qs = Q[pick] - Q[pick].mean(1, keepdims=True)
            points = c.simulate(Qs, form_sd, noise_sd, race_post["b_driver"][race_pick],
                                race_post["c_grid"][race_pick], retire, n_races, rng, extra)
            if not np.isfinite(points).all():
                raise ValueError("Nonfinite simulation outcomes")
            rank = (-points).argsort(1).argsort(1) + 1
            for i, driver in enumerate(ids):
                records.append({"version": version, "driver_id": driver, "name": names.get(driver, driver),
                                "points_per_race": float(points[:, i].mean() / n_races),
                                "p_title": float((rank[:, i] == 1).mean()),
                                "rank_median": float(np.median(rank[:, i])),
                                "rank_lo": int(np.percentile(rank[:, i], 5)),
                                "rank_hi": int(np.percentile(rank[:, i], 95))})
        table = pd.DataFrame(records).sort_values(["version", "points_per_race"], ascending=[True, False])
        summary = {
            "status": "exploratory preview; validation not established", "gate": False,
            "production_publication_authorized": False, "fixed_candidate": [QUALITY],
            "data_as_of": str(design.events.event_id.max()),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "original_combined_validation": reference["combined_validation"],
            "archive_failures": {k: {field: v[field] for field in ("kind", "exception", "gate")}
                                 for k, v in failures.items()},
            "full_pace_diagnostics": {k: pace_summary[k] for k in ("rhat_max", "divergences", "n_draws", "converged")},
            "full_race_stage_fit_attempts": artifacts.fit_record(),
            "simulated_seasons": c.N_SEASONS, "n_races_simulated": n_races,
            "driver_error_rates_used": bool(rel["gate_driver_error_ranking"]),
            "p_retire_common": float(common), "rng_seed": 0,
            "headline": "Expected points; current driver-team compatibility retained",
            "scope": "Current qualifying and career race pace beyond qualifying; conditional equal-car simulation",
            "elapsed_seconds": time.monotonic() - started}
        if before != {p: artifacts.digest(p) for p in inputs}:
            raise ValueError("Preview inputs changed during computation")
        table.to_csv(OUT / "standings.csv", index=False)
        artifacts.atomic_json(OUT / "summary.json", summary)
        write_html(summary, table)
        artifacts.record(OUT, [OUT / name for name in ("summary.json", "standings.csv", "preview.html")],
                         model="exploratory-fixed-pace-preview", inputs=[*inputs, checkpoint],
                         details={"gate": False, "production_publication_authorized": False})
        print("Exploratory preview complete; validation remains not established", flush=True)
    except Exception as exc:
        artifacts.atomic_json(OUT / "failure.json", {"gate": False, "exception": str(exc),
                              "fit_attempts": artifacts.fit_record(), "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
