"""Fresh held-out race targets, retaining the full bootstrap covariance.

Targets are fitted only to each test race and used only for scoring. Their cache
includes every Stage-A design input, resampling group and source implementation.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import racepace

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "outputs/race_total/cache"


def target(C):
    event = str(C.event_id.iloc[0])
    if C.event_id.nunique() != 1:
        raise ValueError("A race target must contain exactly one event")
    X, drivers, _ = racepace.design(C)
    h = hashlib.sha256(Path(racepace.__file__).read_bytes() + Path(__file__).read_bytes())
    h.update(X.tobytes())
    h.update(C.y.to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(C[["driver_id", "stint_key"]], index=False).to_numpy().tobytes())
    key = h.hexdigest()
    path = CACHE / f"target_{event}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["key"]) == key and z["drivers"].tolist() == drivers:
                return (pd.DataFrame({"driver_id": drivers, "pace": z["pace"].copy()}),
                        z["draws"].copy())
    est, draws, _ = racepace.stage_a_race(C, np.random.default_rng(0))
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, key=key, drivers=np.array(drivers), pace=est.pace.to_numpy(), draws=draws[:, 0])
    temporary.replace(path)
    print(f"Scoring target: {event}, {len(C)} clean laps", flush=True)
    return est[["driver_id", "pace"]], draws[:, 0]


if __name__ == "__main__":
    from .race_total import FOLDS
    from .race_total_model import laps
    C = laps()
    for end in FOLDS:
        test = C[(C.event_id > end) & (C.season == int(end[:4]))]
        for _, group in test.groupby("event_id"):
            target(group)
