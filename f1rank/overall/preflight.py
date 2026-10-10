"""Validate real-data interfaces without fitting, scoring or publishing drivers.

    python -m f1rank.overall.preflight --cutoff YYYY-RR --output path.json

Run inside the repository's capped scope. This is development evidence only.
"""
import argparse
import resource
import time
from pathlib import Path

import numpy as np

from ..artifacts import atomic_json
from .design import build
from .provenance import require_snapshot, snapshot


def run(cutoff, output):
    started = time.monotonic()
    source = snapshot()
    d = build(cutoff)
    for block in (d.qualifying.arrays(), d.laps, d.outcomes):
        for name, value in block.items():
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError(f"Nonfinite model input: {name}")
    require_snapshot(source)
    result = dict(stage="data_interface_only", fitted=False, scored=False,
        publication_allowed=False, cutoff=cutoff, source=source,
        counts=dict(qualifying_observations=len(d.qualifying.obs), latent_driver_events=len(d.qualifying.entries),
                    laps=len(d.laps["y"]), outcome_races=len(d.outcomes["entry"]), field=len(d.catalog["field"])),
        catalog=d.catalog, seconds=time.monotonic() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.cutoff, args.output)
    print({key: result[key] for key in ("stage", "fitted", "scored", "counts", "seconds")})
