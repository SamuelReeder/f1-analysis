"""Current-race-only evidence ablation, before fitting or result inspection.

Use the joint model's state prior and all its race likelihood pieces, without
qualifying observations or the qualifying-derived circuit factor. This is a
secondary comparison, not a different public ranking or a numerical rescue.
"""
import numpy as np
import pandas as pd

from f1rank.design import PROCESSED
from f1rank.overall.design import TABLES, build
from f1rank.overall.model import model as joint_model


ARM = "current_race_only"
POLICY = "race_evidence_only_v1"
EMPTY_OBSERVATION_INDICES = ("obs_session", "obs_entry", "obs_car", "obs_car_segment", "car_segment_session")


def without_qualifying(q):
    """Remove observations, their availability dimensions and the learned axis.

    Structural entrant/event/car histories and priors are retained. Replacing
    only y or its training mask would leave qualifying evidence in circuit_factor.
    Array values discarded here never enter a response-dependent branch.
    """
    result = dict(q)
    result.update(n_sessions=0, n_car_segments=0, y=np.empty(0, float),
                  train=np.empty(0, bool), circuit_factor=np.zeros(q["n_circuits"]))
    result.update({name: np.empty(0, int) for name in EMPTY_OBSERVATION_INDICES})
    return result


def build_race_only(cutoff, *, tables=None, clean=None):
    """Use the shared cutoff-first design without reading any qualifying times.

    `clean` has the same synthetic-fixture-only meaning as overall.design.build.
    Race observations and entrant metadata use the common eligibility rules.
    No model is fitted, and this function does not certify historical vintages.
    """
    if tables is None:
        tables = {name: pd.read_parquet(PROCESSED / f"{name}.parquet")
                  for name in TABLES if name != "quali_times"}
    source = dict(tables)
    source["entries"] = source["entries"].drop(columns=["quali_position"], errors="ignore")
    source["quali_times"] = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in
        (("event_id", "object"), ("driver_id", "object"), ("segment", "object"), ("time_s", "float64"))})
    result = build(cutoff, tables=source, clean=clean)
    result.catalog.update(likelihood_arm=ARM, evidence_policy=POLICY,
        qualifying_times_used=False, qualifying_circuit_factor_used=False)
    # Keep one canonical, response-free array representation for fit/forecast IDs.
    q = result.qualifying.arrays()
    expected = without_qualifying(q)
    if any(not np.array_equal(q[key], expected[key]) for key in q):
        raise ValueError("Race-only design retained qualifying evidence")
    return result


def race_only_model(q, laps, outcomes, *, generate=False):
    """Same prior family and race factors; no qualifying observation density.

    The latent component formerly anchored by qualifying is learned only through
    race observations. Retain its uncertainty and learned link rather than fixing
    either after looking at a result. Segment/noise-only priors remain prior-only;
    the empty observation arrays supply no fictitious qualifying measurements.
    """
    return joint_model(without_qualifying(q), laps, outcomes, generate=generate)
