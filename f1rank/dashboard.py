"""Publish an atomic, versioned dashboard dataset, or serve the built dashboard.

publish reads verified exports only; refresh runs fetch/build/fit/export then publish.
A failure preserves the last good release and records the failure in status.json.
The frontend polls this small status file and pointer, never mixed CSV generations.
"""
import argparse
import contextlib
import datetime as dt
import functools
import hashlib
import http.server
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .artifacts import StaleArtifact, atomic_json, digest, require
from .race_publication import load as load_race_pace

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dashboard" / "public" / "data"
RATINGS = ROOT / "outputs" / "ratings"
SCHEMA_VERSION = 1
QUANTILES = ("q05", "q25", "median", "q75", "q95")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text())


def estimate(row, prefix=""):
    result = {q: round(float(row[prefix + q + "_s"]), 6) for q in QUANTILES}
    for key in ("rank_lo", "rank_hi", "rank_median", "p_fastest", "p_top3"):
        if prefix + key in row:
            result[key] = round(float(row[prefix + key]), 6)
    return result


def comparisons(ids, draws, probabilities=None):
    """Joint differences preserve posterior covariance; never subtract marginal intervals."""
    if draws.ndim != 2 or draws.shape[1] != len(ids) or not np.isfinite(draws).all():
        raise ValueError("Invalid joint comparison draws")
    result = {}
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            diff = draws[:, i] - draws[:, j]
            lo, med, hi = np.quantile(diff, [.05, .5, .95])
            p = float(probabilities.loc[a, b]) if probabilities is not None else float(
                (diff > 0).mean() + .5 * (diff == 0).mean())
            result[f"{a}|{b}"] = {"q05": round(float(lo), 6), "median": round(float(med), 6),
                                    "q95": round(float(hi), 6), "p_ahead": round(p, 6)}
    return result


def racing_health():
    rows = []
    specs = [
        ("Race pace & tyre management", "race/multi_heldout.json", "multi_heldout.manifest.json",
         ["gate_race_specific_pace", "gate_degradation"]),
        ("Starts", "firstlap/summary.json", "manifest.json", ["gate_driver_ranking"]),
        ("Consistency", "consistency/summary.json", "manifest.json", ["gate_driver_ranking"]),
        ("Reliability & errors", "reliability/summary.json", "manifest.json",
         ["gate_driver_error_ranking", "gate_team_reliability"]),
        ("Pit stop operations", "pitstops/summary.json", "manifest.json", ["gate_team_ops_rating"]),
        ("Wet pace", "wet/summary.json", "manifest.json", ["gate_wet_ranking"]),
        ("Overtaking & defending", "battles/summary.json", "manifest.json", ["gate_driver_ranking"]),
        ("Equal-car championship", "championship/summary.json", "manifest.json", []),
    ]
    for label, file, manifest, gates in specs:
        path = ROOT / "outputs" / file
        row = {"name": label, "source": f"outputs/{file}", "status": "unavailable",
               "reason": "No published result yet."}
        try:
            require(path.parent, name=manifest, required_outputs=[path])
            summary = read_json(path)
            passed = all(summary.get(g) is True for g in gates) if gates else bool(
                summary.get("combined_validation", {}).get("gate"))
            row.update(status="passed" if passed else "experimental",
                       reason="Recorded validation passed. Ranking integration is pending." if passed
                       else "Validation does not yet support a standalone ranking.")
        except (StaleArtifact, ValueError, OSError) as exc:
            if path.exists():
                row.update(status="stale", reason="Saved results need regeneration after the methodology review.")
            row["detail"] = str(exc).replace(str(ROOT) + "/", "")
        rows.append(row)
    return rows


def build_payload():
    manifest = require(RATINGS, required_outputs=[RATINGS / name for name in (
        "meta.json", "current_drivers.csv", "current_cars.csv", "driver_series.parquet",
        "car_series.parquet", "current_draws.npz", "pairwise_drivers_in_team.csv",
        "pairwise_drivers_portable.csv", "fit_metadata.json")])
    marker = digest(RATINGS / "manifest.json")
    meta = read_json(RATINGS / "meta.json")
    fit_meta = read_json(RATINGS / "fit_metadata.json")
    if (fit_meta.get("created_utc") != meta["fit_id"] or
            fit_meta.get("fingerprint") != meta["fit_fingerprint"]):
        raise ValueError("Export and portable fit metadata do not match")
    if not meta.get("diagnostics", {}).get("converged") or meta.get("pairwise_default") != "in_team":
        raise ValueError("Export is not approved for headline publication")
    d = pd.read_csv(RATINGS / "current_drivers.csv")
    c = pd.read_csv(RATINGS / "current_cars.csv")
    ds = pd.read_parquet(RATINGS / "driver_series.parquet")
    cs = pd.read_parquet(RATINGS / "car_series.parquet")
    events = pd.read_parquet(ROOT / "data" / "processed" / "events.parquet")
    events = events[events.event_id.isin(ds.event_id)].sort_values(["season", "round"])
    if events.empty or events.iloc[-1].event_id != meta["data_as_of"]["event_id"]:
        raise ValueError("Event coverage does not match the export")
    latest = ds[ds.event_id == meta["data_as_of"]["event_id"]].set_index("driver_id")
    drivers = [{"id": r.driver_id, "name": r["name"], "code": str(latest.loc[r.driver_id, "code"]),
                "team": r.team, "lineage": str(latest.loc[r.driver_id, "team"]),
                "has_time": bool(r.has_time), "headline": estimate(r, "in_team_"),
                "portable": estimate(r), "team_effect": estimate(r, "team_effect_"),
                "evidence": {k: int(r[k]) for k in ("events", "seasons", "teammates", "teams")}}
               for _, r in d.iterrows()]
    cars = [{"id": r.team, "name": r.constructor, "pace": estimate(r),
             "gap_to_best": estimate(r, "gap_")} for _, r in c.iterrows()]
    with np.load(RATINGS / "current_draws.npz", allow_pickle=False) as z:
        ids, teams = z["driver_ids"].tolist(), z["teams"].tolist()
        if set(ids) != {r["id"] for r in drivers} or set(teams) != {r["id"] for r in cars}:
            raise ValueError("Draw identifiers do not match the leaderboard")
        pairs = {m: comparisons(ids, z[field], pd.read_csv(RATINGS / filename, index_col=0))
                 for m, field, filename in (
                     ("headline", "in_team_s", "pairwise_drivers_in_team.csv"),
                     ("portable", "portable_skill_s", "pairwise_drivers_portable.csv"))}
        pairs["cars"] = comparisons(teams, z["car_s"])
        comparison_draws = len(z["car_s"])
    history = {"drivers": {}, "cars": {}}
    for id_, rows in ds.groupby("driver_id"):
        history["drivers"][id_] = [{"event": r.event_id, "team": r.constructor_name,
                                     "headline": estimate(r, "in_team_"), "portable": estimate(r)}
                                    for _, r in rows.sort_values("event_idx").iterrows()]
    for id_, rows in cs.groupby("team"):
        history["cars"][id_] = [{"event": r.event_id, "team": r.constructor_name, "pace": estimate(r),
                                  "at_circuit": estimate(r, "at_circuit_")}
                                 for _, r in rows.sort_values("event_idx").iterrows()]
    validations = {}
    for name in ("gates", "lfo_summary"):
        path = ROOT / "outputs" / "validation" / f"{name}.json"
        if path.exists():
            # Git checkout mtimes are not research-run timestamps.
            validations[name] = {"data": read_json(path), "recorded_at": None,
                                 "sha256": digest(path)}
    snapshots = []
    for path in sorted((ROOT / "outputs" / "snapshots").glob("*.json")):
        snap = read_json(path)
        sm = snap.get("meta", {})
        if sm.get("export_version") != "2":
            continue  # older schemas used a different comparison metric
        snapshots.append({"file": path.name, "fit_id": sm.get("fit_id"),
                          "generated_at": sm.get("generated_at"), "event": sm.get("data_as_of"),
                          "drivers": snap["drivers"], "cars": snap["cars"]})
    require(RATINGS)  # reject a concurrent export or source edit while assembling
    if digest(RATINGS / "manifest.json") != marker:
        raise ValueError("Export changed during publication; retry")
    return {"schema_version": SCHEMA_VERSION, "meta": meta, "drivers": drivers, "cars": cars,
            "catalog": {"drivers": ds[["driver_id", "name"]].drop_duplicates("driver_id").rename(
                columns={"driver_id": "id"}).sort_values("name").to_dict("records"),
                        "cars": cs.sort_values("event_idx").drop_duplicates("team", keep="last")[[
                            "team", "constructor_name"]].rename(columns={"team": "id", "constructor_name": "name"})
                            .sort_values("name").to_dict("records")},
            "events": events.to_dict("records"), "history": history, "comparisons": pairs,
            "comparison_draws": comparison_draws, "racing": racing_health(),
            "race_pace": load_race_pace(ROOT),
            "validation": validations, "snapshots": snapshots,
            "provenance": {"export_manifest": marker, "inputs": manifest["inputs"],
                           "outputs": manifest["outputs"]}}


def publish(data_dir=DATA):
    """Write immutable content first; switch the current pointer only after success."""
    payload = build_payload()
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    release = hashlib.sha256(encoded).hexdigest()[:20]
    target = data_dir / "releases" / f"{release}.json"
    if target.exists() and read_json(target) != payload:
        raise ValueError("Existing immutable release is corrupt; publication stopped")
    if not target.exists():
        atomic_json(target, payload)
    pointer = {"schema_version": SCHEMA_VERSION, "release": release,
               "url": f"releases/{release}.json", "published_at": now(),
               "data_as_of": payload["meta"]["data_as_of"]}
    atomic_json(data_dir / "latest.json", pointer)
    return pointer


@contextlib.contextmanager
def publication_lock(data_dir):
    """Process-owned lock; a crash releases it without leaving a stale lockout."""
    import fcntl
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / ".refresh.lock").open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another dashboard refresh is running") from exc
        yield


def run(refresh=False, data_dir=DATA):
    with publication_lock(data_dir):
        start = time.monotonic()
        status = {"state": "running", "started_at": now(), "stage": "checking exports"}
        atomic_json(data_dir / "status.json", status)
        log_dir = ROOT / "outputs" / "dashboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            if refresh:
                with (log_dir / "refresh.log").open("a") as log:
                    for module in ("fetch", "build", "fit", "export"):
                        status.update(stage=module, updated_at=now())
                        atomic_json(data_dir / "status.json", status)
                        log.write(f"\n{now()} {module}\n"); log.flush()
                        subprocess.run([sys.executable, "-m", f"f1rank.{module}"], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
            status.update(stage="publishing", updated_at=now())
            atomic_json(data_dir / "status.json", status)
            pointer = publish(data_dir)
            status.update(state="ok", stage="published", release=pointer["release"])
        except BaseException as exc:
            status.update(state="failed", error=str(exc).replace(str(ROOT) + "/", ""))
            raise
        finally:
            status.update(finished_at=now(), duration_s=round(time.monotonic() - start, 2))
            atomic_json(data_dir / "status.json", status)
            with (log_dir / "runs.jsonl").open("a") as log:
                log.write(json.dumps(status) + "\n")
    return pointer


def current_status(data_dir=DATA):
    """Report a terminated refresh accurately even if it could not write its final status."""
    status = read_json(data_dir / "status.json")
    if status.get("state") == "running":
        try:
            with publication_lock(data_dir):
                status = {**status, "state": "failed", "error": "The refresh process stopped before publishing."}
        except RuntimeError:
            pass  # the process still owns its lock
    return status


def serve(port, data_dir=DATA):
    dist = ROOT / "dashboard" / "dist"
    if not (dist / "index.html").exists():
        raise SystemExit("Build the dashboard first: cd dashboard && npm run build")
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?", 1)[0] == "/data/status.json":
                try:
                    body = json.dumps(current_status(data_dir)).encode()
                except (OSError, ValueError):
                    self.send_error(404, "No refresh status available")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def translate_path(self, path):
            # Serve only the dashboard and published datasets, never the repository.
            clean = path.split("?", 1)[0]
            if clean.startswith("/data/"):
                candidate = (data_dir / clean.removeprefix("/data/")).resolve()
                if not candidate.is_relative_to(data_dir.resolve()) or candidate.name.startswith("."):
                    return str(dist / "__not_found__")
                return str(candidate)
            return super().translate_path(path)
        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=dist))
    print(f"Dashboard: http://localhost:{port}", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["publish", "refresh", "serve"], nargs="?", default="publish")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.port)
    else:
        print(json.dumps(run(refresh=args.command == "refresh"), indent=2))


if __name__ == "__main__":
    main()
