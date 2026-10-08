"""Authorized computational follow-up; existing likelihood, one fixed sampler recipe.

Register and commit before running. Run only as a memory-capped systemd service.
"""
import argparse
import datetime as dt
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

import numpy as np
from numpyro.infer import init_to_median

from analysis import championship_archive as archive
from analysis import championship_followup as prospective
from f1rank import artifacts, championship as c, racemulti

ROOT = archive.ROOT
OUT = ROOT / "outputs/analysis/championship_archive_conservative"
CACHE = ROOT / "outputs/fits/championship_archive_conservative"
PROTOCOL = ROOT / "docs/championship_archive_conservative_protocol.json"
PLAN = ROOT / "docs/championship_archive_conservative.md"
# The final existing artifacts.RETRY stage applied to racemulti.fit's defaults.
SAMPLER = {"warmup": 1500, "samples": 2000, "chains": 4,
           "target_accept": 0.98, "seed": 3}


def source_hashes():
    return {**archive.sources(), **archive.hashes([Path(__file__), PLAN])}


def register():
    parent = archive.check_registration()
    artifacts.require(archive.OUT, name="failure.manifest.json",
                      required_outputs=[archive.OUT / "failure.json"])
    failure = json.loads((archive.OUT / "failure.json").read_text())
    if failure["kind"] != "exhausted_registered_fit" or list(archive.OUT.glob("fold_*.json")):
        raise ValueError("The authorized computational follow-up requires the recorded pre-scoring failure")
    if (archive.OUT / "result.json").exists():
        raise ValueError("Archive predictions have already been evaluated")
    reusable = [archive.CACHE / "clean_laps.parquet", archive.CACHE / "prepared.json",
                *sorted((archive.CACHE / "weather").glob("*.json")),
                *sorted(archive.CACHE.glob("quali_*.npz")),
                *sorted(archive.CACHE.glob("quali_*.meta.json"))]
    if not reusable[0].exists() or not reusable[1].exists():
        raise ValueError("Missing frozen prepared archive")
    parent_inputs = [*(ROOT / k for k in parent["input_hashes"]), archive.PROTOCOL,
                     archive.OUT / "failure.json", archive.OUT / "failure.manifest.json", *reusable]
    protocol = {
        **{k: parent[k] for k in (
            "candidate", "first_training_season", "first_test_season", "last_test_season",
            "min_converged_test_seasons", "decision_after_date", "test_event_ids",
            "primary_baseline", "secondary_baselines", "selection", "gate", "qualifying_attempts")},
        "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "registration_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "authorization": "User approved the documented separate computational follow-up after the archive sampling failure",
        "parent_protocol_sha256": artifacts.digest(archive.PROTOCOL),
        "parent_failure_sha256": artifacts.digest(archive.OUT / "failure.json"),
        "race_pace_sampler": SAMPLER,
        "race_pace_attempts": 1,
        "qualifying_and_race_stage": "Unchanged archive qualifying rules and championship.fit_race defaults",
        "ordering": "All qualifying and pace fits must converge before any race-stage predictions are scored",
        "failure_policy": "Every registered fold is required; an exhausted fit terminates this separate study",
        "publication": "Separate earlier-era evidence; preserve all previous studies; no automatic ranking publication",
        "source_hashes": source_hashes(), "input_hashes": archive.hashes(parent_inputs),
        "reused_input_hashes": archive.hashes(reusable),
    }
    with PROTOCOL.open("x") as f:
        json.dump(protocol, f, indent=2)
        f.write("\n")
    print("Conservative computational follow-up registered; commit before running", flush=True)


def check_registration():
    relative = str(PROTOCOL.relative_to(ROOT))
    revisions = subprocess.check_output(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", relative], cwd=ROOT, text=True).splitlines()
    if not revisions or subprocess.check_output(
            ["git", "show", f"{revisions[-1]}:{relative}"], cwd=ROOT) != PROTOCOL.read_bytes():
        raise ValueError("Computational follow-up registration must be committed and immutable")
    p = json.loads(PROTOCOL.read_text())
    if p["source_hashes"] != source_hashes() or any(
            artifacts.digest(ROOT / k) != v for k, v in p["input_hashes"].items()):
        raise ValueError("Registered computational follow-up sources or inputs changed")
    if p["race_pace_sampler"] != SAMPLER or p["race_pace_attempts"] != 1:
        raise ValueError("Sampler differs from the authorized fixed recipe")
    archive.check_registration()
    return p


def fit_pace(C, **settings):
    """Same arrays/model/initialization as racemulti.fit, with the authorized recipe."""
    expected = {**SAMPLER, "protocol_sha256": artifacts.digest(PROTOCOL)}
    if settings != expected:
        raise ValueError("Unregistered sampler settings")
    started = time.monotonic()
    d, drivers = racemulti.arrays(C)
    mcmc = racemulti.MCMC(
        racemulti.NUTS(racemulti.model, target_accept_prob=settings["target_accept"],
                      init_strategy=init_to_median(num_samples=15)),
        num_warmup=settings["warmup"], num_samples=settings["samples"],
        num_chains=settings["chains"], chain_method=racemulti.CHAINS, progress_bar=False)
    mcmc.run(c.jax.random.PRNGKey(settings["seed"]), d, extra_fields=("diverging",))
    checked = artifacts.diagnostics(mcmc.get_samples(group_by_chain=True),
                                   int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))
    elapsed = time.monotonic() - started
    attempt = {**settings, **checked, "elapsed_seconds": elapsed}
    print(f"  conservative pace diagnostics: {attempt}", flush=True)
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k in racemulti.KEEP}
    post.update(_rhat_max=checked["rhat_max"], _divergences=checked["divergences"],
                _n_draws=checked["n_draws"], _converged=checked["converged"],
                _attempts=[attempt], _minutes=elapsed / 60, _backend=c.jax.default_backend())
    return post, drivers


def run_folds(protocol, train_fold, score_fold):
    """Do not score any outcomes until every registered training fold is usable."""
    seasons = range(protocol["first_test_season"], protocol["last_test_season"] + 1)
    trained = [(s, train_fold(s)) for s in seasons]
    return [score_fold(s, value) for s, value in trained]


def run():
    protocol = check_registration()
    if (OUT / "result.json").exists() or (OUT / "failure.json").exists():
        raise FileExistsError("The computational follow-up already reached a recorded outcome")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    qualifying_attempts, pace_fits, used_caches = {}, {}, {}
    try:
        # These are frozen inputs. Preparation returns the verified existing cache;
        # a separate namespace holds every newly fitted or copied checkpoint.
        clean, preparation = archive.prepare(archive.check_registration())
        artifacts.atomic_json(OUT / "preparation.json", preparation)
        for relative, expected in protocol["reused_input_hashes"].items():
            source = ROOT / relative
            if source.name.startswith("quali_"):
                target = CACHE / source.name
                if not target.exists():
                    shutil.copy2(source, target)
                if artifacts.digest(target) != expected:
                    raise ValueError("Copied qualifying checkpoint changed")

        def train(season):
            print(f"== conservative training {season} start", flush=True)
            with patch.object(archive, "CACHE", CACHE):
                driver, car, attempts = archive.qualifying(season, protocol)
            qualifying_attempts[str(season)] = attempts
            used_caches.update(archive.hashes([CACHE / f"quali_{season}.npz", CACHE / f"quali_{season}.meta.json"]))
            c.jax.clear_caches()
            gc.collect()
            training = clean[clean.season < season].merge(
                driver.rename(columns={"driver": "quali"}), on=["event_id", "driver_id"],
                how="left", validate="many_to_one").dropna(subset=["quali"])
            if training.empty or int(training.season.max()) >= season:
                raise ValueError("Invalid pace training cutoff")
            with patch.object(racemulti, "fit", fit_pace):
                post, ids = racemulti.checkpointed(
                    training, CACHE / f"pace_{season}.npz", **SAMPLER,
                    protocol_sha256=artifacts.digest(PROTOCOL))
            pace_fits[str(season)] = {k: v for k, v in post.items() if k.startswith("_")}
            if not post["_converged"]:
                raise archive.ExhaustedFit(f"Conservative archive pace failed convergence for {season}")
            used_caches.update(archive.hashes([CACHE / f"pace_{season}.npz"]))
            effects = {"race_specific_pace": {season: (np.asarray(ids), post["u"].astype(np.float32))}}
            artifacts.atomic_json(OUT / f"training_{season}.json", {
                "season": season, "training_last_event": str(training.event_id.max()),
                "qualifying_attempts": attempts, "race_pace_fit": pace_fits[str(season)]})
            c.jax.clear_caches()
            gc.collect()
            print(f"== conservative training {season} done", flush=True)
            return driver, car, effects

        def score(season, trained):
            print(f"== conservative scoring {season} start", flush=True)
            driver, car, effects = trained
            frame = archive.race_frame(season, protocol["first_training_season"], driver, car)
            try:
                with patch.object(c, "ENTRY_FIRST_SEASON", protocol["first_training_season"]), \
                        patch.object(c, "race_orders", lambda s: frame):
                    record = prospective.score_fold(season, protocol["test_event_ids"], effects)
            except RuntimeError as exc:
                if artifacts._FITS and not artifacts._FITS[-1][-1]["converged"]:
                    raise archive.ExhaustedFit(str(exc)) from exc
                raise
            artifacts.atomic_json(OUT / f"fold_{season}.json", record)
            print(f"== conservative scoring {season} done", flush=True)
            return record

        records = run_folds(protocol, train, score)
        check_registration()
        if any(artifacts.digest(ROOT / k) != v for k, v in used_caches.items()):
            raise ValueError("A training checkpoint changed during evaluation")
        result = {**prospective.decide(records, protocol, dt.datetime.now(dt.timezone.utc).date()),
                  "evidence": "Separate earlier-era validation after an explicitly authorized computational change",
                  "protocol_sha256": artifacts.digest(PROTOCOL), "folds": records,
                  "qualifying_attempts": qualifying_attempts, "race_pace_fits": pace_fits,
                  "used_cache_hashes": used_caches, "race_stage_fit_attempts": artifacts.fit_record(),
                  "elapsed_seconds": time.monotonic() - started}
        artifacts.atomic_json(OUT / "result.json", result)
        inputs = [*(ROOT / k for k in protocol["input_hashes"]),
                  *(ROOT / k for k in protocol["source_hashes"]), PROTOCOL,
                  *(ROOT / k for k in used_caches)]
        outputs = [OUT / "result.json", OUT / "preparation.json",
                   *sorted(OUT.glob("training_*.json")), *sorted(OUT.glob("fold_*.json"))]
        artifacts.record(OUT, outputs, model="fixed-pace-archive-conservative", inputs=inputs,
                         details={"protocol_sha256": artifacts.digest(PROTOCOL), "gate": result["gate"]})
        print("Conservative archive study complete", flush=True)
    except Exception as exc:
        terminal = isinstance(exc, archive.ExhaustedFit)
        name = "failure.json" if terminal else (
            "operational_failure_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        artifacts.atomic_json(OUT / name, {
            "gate": False, "publication_authorized": False, "exception": str(exc),
            "kind": "exhausted_registered_fit" if terminal else "operational",
            "qualifying_attempts": qualifying_attempts, "race_pace_fits": pace_fits,
            "qualifying_failures": {p.stem: json.loads(p.read_text()) for p in CACHE.glob("quali_*.failed.json")},
            "race_stage_fit_attempts": artifacts.fit_record(), "used_cache_hashes": used_caches,
            "elapsed_seconds": time.monotonic() - started,
            "protocol_sha256": artifacts.digest(PROTOCOL)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    register() if parser.parse_args().action == "register" else run()


if __name__ == "__main__":
    main()
