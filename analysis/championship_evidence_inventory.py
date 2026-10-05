"""Inventory available event identifiers without inspecting potential test outcomes."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/championship_evidence_inventory.json"


def main():
    tables = {}
    for name in ("events", "race", "jolpica_laps", "race_laps"):
        path = ROOT / f"data/processed/{name}.parquet"
        # Do not load classification, timing, or qualifying outcomes for an archive audit.
        ids = sorted(set(pq.read_table(path, columns=["event_id"])["event_id"].to_pylist()))
        tables[name] = {"first_event": ids[0], "last_event": ids[-1],
                        "pre_2010_event_ids": [e for e in ids if e[:4] < "2010"]}
    sources = ["f1rank/benchmark.py", "f1rank/racemulti.py", "f1rank/compare.py",
               "docs/championship_followup.md", "analysis/championship_evidence_inventory.py"]
    result = {
        "tables": tables,
        "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "historical_evidence": "The completed championship scores have already informed candidate choice.",
        "earlier_archive": (
            "Earlier results exist, but matching earlier lap inputs are absent. Earlier qualifying "
            "data have been used in the data-window comparison. This inventory does not establish "
            "that earlier race outcomes were never inspected or used to guide development."),
        "untouched_holdout_identified": False,
        "next_action": (
            "Freeze the follow-up before obtaining new outcomes; confirm any proposed independent "
            "archive's prior use and input coverage before relying on it for validation."),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
