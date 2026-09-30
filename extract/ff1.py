"""Shared FastF1 setup for the extraction scripts (run in .venv-fastf1)."""

import logging
import time
from pathlib import Path

import fastf1
from fastf1.exceptions import RateLimitExceededError

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "fastf1"


def enable_cache() -> None:
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))


def patient(fn, *args, waits: int = 12, **kwargs):
    """Call fn, waiting out FastF1's API rate limit (500 calls/h; downloads are cached,
    so a rerun resumes where it stopped)."""
    for _ in range(waits):
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededError:
            print("FastF1 rate limit reached: waiting 10 minutes", flush=True)
            time.sleep(600)
    return fn(*args, **kwargs)
