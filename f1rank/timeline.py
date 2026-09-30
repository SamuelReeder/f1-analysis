"""Race event timeline with uncertain causes (stage 2, docs/racing_approach.md).

    python -m f1rank.timeline

One row per thing that happened in a race: its laps, the evidence for it and, where a
cause is at stake, probabilities over causes. Retirements and other outcomes cover 2010
onward (Jolpica; statuses are coded through 2022). Everything else needs FastF1 timing
and race-control messages, so it covers 2018 onward (all races but the 2018 Italian GP). Rules turn sources into evidence,
not verdicts. Each later model derives its own exclusions from this table and samples
uncertain causes from the probabilities.

Kinds
  safety_car, vsc, red_flag, yellow   neutralisations (race-wide), from track status
  weather_change                      rain starting or stopping (race-wide)
  retirement                          a car that stopped racing, with cause probabilities
  did_not_start, disqualified, withdrew   administrative outcomes (a car Jolpica marks
                                      withdrawn with no laps did not start, whatever its status)
  stoppage                            race control: "CAR n STOPPED" (cause unknown by itself)
  off_track                           race control: spun / off track / track limits
  incident_noted                      race control: an incident naming cars, with the
                                      stewards' outcome and reason
  penalty                             race control: a penalty or warning for a car, with reason
  pit_anomaly                         a pit stop 5+ s longer than the race's median stop
                                      (not while a red flag or a stop-go penalty explains it)
  suspected_damage                    a pit stop within 2 laps after an incident naming the car
  possible_team_order                 teammates within 2 s swapping places on green laps without
                                      pitting (for sensitivity analyses only)

Cause probabilities (p_mechanical, p_own_error, p_other_driver, p_external, p_team_ops,
p_unknown) are given for retirements:
- Coded status (Jolpica codes causes through 2022, and a few later): the status fixes the
  class (mechanical, incident, other).
- Status "Retired" (almost every retirement from 2023; also "Withdrew" after racing laps):
  class probabilities come from a multinomial logistic model of the class given
  race-control and lap evidence, fitted on
  the coded retirements of races with FastF1 evidence (2018-2022, and the few coded
  later) and checked season by season (audit).
- The incident class is split between own error, the other driver, external and unknown by
  the ATTRIBUTION table: stated assumptions, not estimates, to be varied in sensitivity
  analyses. Stewards judge infringements, not necessarily who caused the outcome, so a
  penalty shifts attribution without settling it.

Reviewed rows in data/overrides/timeline_overrides.csv replace inferred probabilities and
are never re-inferred. Writes data/processed/timeline.parquet, and an audit (counts,
cause-model validation, retirements, changes since the previous build) to outputs/timeline/.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OVERRIDES = ROOT / "data" / "overrides" / "timeline_overrides.csv"
AUDIT = ROOT / "outputs" / "timeline"
VERSION = "timeline-v1"
FIRST_SEASON = 2010  # outcomes (Jolpica); FastF1 evidence starts in 2018

CAUSES = ["mechanical", "own_error", "other_driver", "external", "team_ops", "unknown"]
P_COLS = [f"p_{c}" for c in CAUSES]
CLASSES = ["mechanical", "incident", "other"]

FINISHED = re.compile(r"^(Finished|\+\d+ Laps?|Lapped)$")
ADMIN = {"Did not start": "did_not_start", "Disqualified": "disqualified", "Excluded": "disqualified",
         "Withdrew": "withdrew", "Did not qualify": "did_not_start", "Illness": "withdrew", "Injury": "withdrew"}
INCIDENT_STATUS = {"Collision", "Collision damage", "Accident", "Spun off", "Damage"}
OTHER_STATUS = {"Wheel", "Wheel nut", "Puncture", "Tyre", "Debris", "Out of fuel", "Fuel"}
UNCODED = {"Retired", "Withdrew"}  # retirements whose status gives no cause
# every other non-finishing status Jolpica uses is a component (mechanical)

# How the incident class is split, by the evidence about who was involved. Assumptions.
ATTRIBUTION = {
    "single_car": {"own_error": 0.7, "external": 0.1, "unknown": 0.2},
    "penalised": {"own_error": 0.7, "other_driver": 0.1, "unknown": 0.2},
    "other_penalised": {"own_error": 0.1, "other_driver": 0.7, "unknown": 0.2},
    "multi_car": {"own_error": 0.3, "other_driver": 0.3, "unknown": 0.4},
}
# the "other" class (wheel, tyre, debris, fuel) by status; unknown when not coded
OTHER_SPLIT = {"Wheel": {"team_ops": 0.5, "mechanical": 0.5}, "Wheel nut": {"team_ops": 1.0},
               "Out of fuel": {"team_ops": 1.0}, "Fuel": {"team_ops": 1.0},
               "Puncture": {"external": 0.5, "unknown": 0.5}, "Tyre": {"external": 0.5, "unknown": 0.5},
               "Debris": {"external": 1.0}}
CODED_CONFIDENCE = 0.95  # a coded status can be wrong: the rest goes to unknown

CAR = re.compile(r"\b(\d{1,2}) \(([A-Z]{3})\)")
PENALTY = re.compile(r"(\d+ SECOND TIME PENALTY|DRIVE THROUGH PENALTY|STOP AND GO PENALTY|STOP/GO PENALTY|"
                     r"\d+ SECOND STOP AND GO PENALTY|\d+ PLACE GRID PENALTY|GRID PENALTY|REPRIMAND|"
                     r"DISQUALIFIED|BLACK AND WHITE FLAG)", re.I)
INCIDENT = re.compile(r"INCIDENT(?:S)? INVOLVING CARS?", re.I)
OUTCOME = re.compile(r"(NOTED|UNDER INVESTIGATION|WILL BE INVESTIGATED AFTER THE RACE|"
                     r"NO FURTHER (?:ACTION|INVESTIGATION)|NO INVESTIGATION NECESSARY|REVIEWED)", re.I)
# a contact incident names 2+ cars or gives a contact reason (not track limits, pit speeding, ...)
CONTACT = re.compile(r"COLLISION|CONTACT|CRASH|FORCING ANOTHER DRIVER", re.I)
SPUN = re.compile(r"SPUN|OFF TRACK AND CONTINUED", re.I)
OFF_TRACK = re.compile(r"(SPUN AND CONTINUED|OFF TRACK AND CONTINUED|MISSED THE APEX|TRACK LIMITS|"
                       r"TIME .* DELETED|LAP DELETED)", re.I)
TRACK_STATUS = {"2": "yellow", "4": "safety_car", "5": "red_flag", "6": "vsc"}
PIT_ANOMALY_S = 5.0
TEAM_ORDER_GAP_S = 2.0


def load() -> dict[str, pd.DataFrame]:
    names = ["race_laps", "race_messages", "race_track_status", "race_weather", "race_classification", "race"]
    t = {n: pd.read_parquet(PROCESSED / f"{n}.parquet") for n in names}
    t["sources"] = json.loads((PROCESSED / "race_sources.json").read_text())
    return t


def _row(event_id, driver_id, lap_start, lap_end, kind, detail, evidence, involved=(), reason="",
         car_lap=None, **p) -> dict:
    row = {"event_id": event_id, "driver_id": driver_id, "lap_start": lap_start, "lap_end": lap_end,
           "car_lap": car_lap, "kind": kind, "detail": detail, "reason": reason,
           "involved": ",".join(involved), "evidence": json.dumps(evidence)}
    row.update({c: p.get(c, np.nan) for c in P_COLS})
    return row


def leader_lap(laps: pd.DataFrame):
    """Race lap in progress at a session time (from the leader's lap end times)."""
    ends = np.sort(laps.groupby("lap_number").time.min().dropna().to_numpy())
    return lambda t: int(np.searchsorted(ends, t, side="right") + 1)


# --------------------------------------------------------------------------- race-wide

def neutralisations(event_id: str, ts: pd.DataFrame, lap_at) -> list[dict]:
    rows, open_ = [], {}
    for r in ts.sort_values("time").itertuples():
        statuses = set(str(r.status))
        # a new status ends the previous non-green ones it replaces
        for code in list(open_):
            if code not in statuses:
                start, t0 = open_.pop(code)
                rows.append(_row(event_id, None, start, lap_at(r.time), TRACK_STATUS[code],
                                 f"{TRACK_STATUS[code]} from {t0:.0f}s to {r.time:.0f}s",
                                 [{"source": "fastf1 track status", "status": code}]))
        for code in statuses & set(TRACK_STATUS):
            if code not in open_:
                open_[code] = (lap_at(r.time), r.time)
    for code, (start, t0) in open_.items():
        rows.append(_row(event_id, None, start, None, TRACK_STATUS[code], f"{TRACK_STATUS[code]} from {t0:.0f}s",
                         [{"source": "fastf1 track status", "status": code}]))
    return rows


def weather_changes(event_id: str, w: pd.DataFrame, lap_at) -> list[dict]:
    w = w.sort_values("time")
    rain = w.rainfall.astype(bool).to_numpy()
    rows = []
    for i in np.flatnonzero(rain[1:] != rain[:-1]) + 1:
        rows.append(_row(event_id, None, lap_at(w.time.iloc[i]), None, "weather_change",
                         "rain started" if rain[i] else "rain stopped",
                         [{"source": "fastf1 weather", "rainfall": bool(rain[i])}]))
    return rows


# --------------------------------------------------------------------------- race control

def message_events(event_id: str, msgs: pd.DataFrame, number_to_driver: dict) -> list[dict]:
    """Per-car events from race-control messages. Messages about one incident (noted,
    investigated, decided) are grouped by the cars named and the reason."""
    rows, incidents = [], {}
    for m in msgs.sort_values("time").itertuples():
        text = str(m.message)
        cars = [number_to_driver.get(n) for n, _ in CAR.findall(text)]
        cars = [c for c in cars if c]
        lap = int(m.lap) if pd.notna(m.lap) else None
        ev = {"source": "race control", "text": text, "lap": lap}
        reason = text.split(" - ", 1)[1].strip() if " - " in text else ""
        if not cars:
            continue
        pen = PENALTY.search(text)
        if pen and not INCIDENT.search(text):
            for c in cars:
                rows.append(_row(event_id, c, lap, lap, "penalty", pen.group(1).upper(), [ev], reason=reason))
        elif INCIDENT.search(text):
            key = (tuple(sorted(cars)), reason.split(" - ")[0] if reason else "")
            outcome = OUTCOME.search(text)
            inc = incidents.setdefault(key, {"lap": lap, "evidence": [], "outcome": "", "reason": key[1]})
            inc["evidence"].append(ev)
            if outcome:
                inc["outcome"] = outcome.group(1).upper()
        elif "STOPPED" in text:
            rows += [_row(event_id, c, lap, lap, "stoppage", "stopped on track (cause unknown)", [ev]) for c in cars]
        elif OFF_TRACK.search(text):
            rows += [_row(event_id, c, lap, lap, "off_track", OFF_TRACK.search(text).group(1).lower(), [ev])
                     for c in cars]
    for (cars, reason), inc in incidents.items():
        for c in cars:
            others = [x for x in cars if x != c]
            kind = "contact" if others or CONTACT.search(reason) else "other"
            detail = f"{kind}; {inc['outcome'].lower() or 'no outcome given'}"
            rows.append(_row(event_id, c, inc["lap"], inc["lap"], "incident_noted", detail, inc["evidence"],
                             involved=others, reason=reason))
    return rows


# --------------------------------------------------------------------------- laps

def pit_events(event_id: str, laps: pd.DataFrame, rows_so_far: list[dict]) -> list[dict]:
    """Pit anomalies and suspected damage, from pit-lane times."""
    L = laps.sort_values(["driver_id", "lap_number"])
    nxt = L.groupby("driver_id")[["pit_out_time", "lap_number"]].shift(-1)
    stops = L[L.pit_in_time.notna() & nxt.pit_out_time.notna()].assign(
        pit_lane_s=nxt.pit_out_time - L.pit_in_time, out_lap=nxt.lap_number)
    stops = stops[stops.pit_lane_s > 0]
    out = []
    if not len(stops):
        return out
    median = stops.pit_lane_s.median()
    red = {(r["lap_start"], r["lap_end"]) for r in rows_so_far if r["kind"] == "red_flag"}
    served = {(r["driver_id"], r["lap_start"]) for r in rows_so_far if r["kind"] == "penalty"
              and re.search("STOP AND GO|STOP/GO|DRIVE THROUGH", r["detail"])}
    incidents = [(r["driver_id"], r["lap_start"]) for r in rows_so_far if r["kind"] == "incident_noted"
                 and r["lap_start"] is not None and r["detail"].startswith("contact")]
    lap_at = leader_lap(laps)
    for s in stops.itertuples():
        car_lap = int(s.lap_number)
        lap = lap_at(s.pit_in_time)
        under_red = any(a is not None and a <= lap <= (b or 10 ** 6) for a, b in red)
        penalty = any(d == s.driver_id and la is not None and abs(la - lap) <= 3 for d, la in served)
        if s.pit_lane_s > median + PIT_ANOMALY_S and not under_red and not penalty:
            out.append(_row(event_id, s.driver_id, lap, lap, "pit_anomaly",
                            f"pit lane {s.pit_lane_s:.1f}s vs race median {median:.1f}s",
                            [{"source": "fastf1 laps", "pit_lane_s": round(float(s.pit_lane_s), 2),
                              "race_median_s": round(float(median), 2)}], car_lap=car_lap))
        near = [la for d, la in incidents if d == s.driver_id and 0 <= lap - la <= 2]
        if near:
            out.append(_row(event_id, s.driver_id, lap, lap, "suspected_damage",
                            f"pit stop {lap - near[0]} lap(s) after a contact incident naming the car",
                            [{"source": "fastf1 laps + race control", "incident_lap": near[0], "pit_lap": lap}],
                            car_lap=car_lap))
    return out


def team_orders(event_id: str, laps: pd.DataFrame) -> list[dict]:
    lap_at = leader_lap(laps)
    L = laps[laps.position.notna()].copy()
    L["green"] = L.track_status.astype(str) == "1"
    L["pit"] = L.pit_in_time.notna() | L.pit_out_time.notna()
    rows = []
    for team, g in L.groupby("team"):
        ids = g.driver_id.dropna().unique()
        if len(ids) != 2:
            continue
        a, b = sorted(ids)
        w = g.pivot_table(index="lap_number", columns="driver_id", values=["position", "time", "green", "pit"],
                          aggfunc="first").dropna()
        if not len(w):
            continue
        ahead = (w["position"][a] < w["position"][b]).to_numpy()
        close = (w["time"][a] - w["time"][b]).abs().to_numpy() < TEAM_ORDER_GAP_S
        ok = w["green"].all(axis=1).to_numpy().astype(bool) & ~w["pit"].any(axis=1).to_numpy().astype(bool)
        laps_ = w.index.to_numpy()
        for i in range(1, len(w)):
            if ahead[i] != ahead[i - 1] and close[i] and close[i - 1] and ok[i] and ok[i - 1] \
                    and laps_[i] == laps_[i - 1] + 1:
                passed, passer = (a, b) if ahead[i - 1] else (b, a)
                lap = lap_at(float(w["time"][passer].iloc[i]))
                rows.append(_row(event_id, passer, lap, lap, "possible_team_order",
                                 f"passed teammate within {TEAM_ORDER_GAP_S:.0f}s", [{"source": "fastf1 laps"}],
                                 involved=[passed], car_lap=int(laps_[i])))
    return rows


# --------------------------------------------------------------------------- retirements

def admin_kind(r) -> str | None:
    """Administrative outcome, or None for a car that raced and stopped. Jolpica's position
    text "W" (withdrawn) with no laps marks a car that did not start, even when the status
    names a component (e.g. "Engine" for a failure before the start). "Withdrew" after
    racing laps is a retirement whose cause is not coded."""
    if r.status in ("Disqualified", "Excluded"):
        return "disqualified"
    if r.laps == 0 and (r.position_text == "W" or r.status in ADMIN):
        return ADMIN.get(r.status, "did_not_start")
    if r.status in ADMIN and r.status != "Withdrew":
        return ADMIN[r.status]
    return None


def status_class(status: str) -> str | None:
    if FINISHED.match(status):
        return None
    if status in INCIDENT_STATUS:
        return "incident"
    if status in OTHER_STATUS or status in UNCODED:
        return "other"
    return "mechanical"


def race_lap_of_retirement(r, laps: pd.DataFrame, lap_at) -> int:
    """Race lap (the leader's) in which a car stopped: the lap in progress when the car
    completed its last lap, or 1 if it completed none. Completed laps are Jolpica's count:
    FastF1's row for the unfinished lap can carry a much later time, and FastF1 sometimes
    lacks a car's last laps, which are then added on."""
    mine = laps[(laps.driver_id == r.driver_id) & laps.time.notna() & (laps.lap_number <= int(r.laps))]
    if not len(mine):
        return 1
    last = mine.loc[mine.lap_number.idxmax()]
    return lap_at(float(last.time)) + int(r.laps) - int(last.lap_number)


def retirement_features(r, lap: int, laps: pd.DataFrame, rows: list[dict], retired_laps: dict) -> dict:
    """Evidence about one retirement in race lap `lap`, for the cause model. Only evidence
    whose meaning is stable across seasons feeds the cause model ("CAR n STOPPED" messages
    almost vanish after 2019, "spun" messages after 2023; track-limits messages surge from
    2020). spun_msg is kept as recorded evidence only."""
    near = lambda x: x["lap_start"] is not None and lap - 2 <= x["lap_start"] <= lap + 1  # noqa: E731
    mine = [x for x in rows if x["driver_id"] == r.driver_id]
    contact = [x for x in mine if x["kind"] == "incident_noted" and x["detail"].startswith("contact") and near(x)]
    involved = {o for x in contact for o in x["involved"].split(",") if o}
    after = lambda x: x["lap_start"] is not None and x["lap_start"] >= lap - 2  # noqa: E731
    collision_penalty = lambda x: x["kind"] == "penalty" and CONTACT.search(x["reason"]) and after(x)  # noqa: E731
    L = laps[laps.driver_id == r.driver_id].sort_values("lap_number")
    lt = L.lap_time.dropna()
    return {
        "lap1": lap == 1,
        "contact_msg": bool(contact),
        "multi_car": bool(involved) or any(d != r.driver_id and abs(la - lap) <= 1 for d, la in retired_laps.items()),
        "spun_msg": any(x["kind"] == "off_track" and SPUN.search(x["detail"]) and near(x) for x in mine),
        "neutralised": any(x["kind"] in ("safety_car", "vsc", "red_flag") and x["lap_start"] is not None
                           and lap - 1 <= x["lap_start"] <= lap + 1 for x in rows if x["driver_id"] is None),
        "in_pits": bool(len(L)) and pd.notna(L.pit_in_time.iloc[-1]),
        "slow_before": bool(len(lt) >= 6 and lt.iloc[-3:].median() > 1.03 * lt.iloc[:-3].median()),
        # attribution evidence (not model features)
        "penalised": any(collision_penalty(x) for x in mine),
        "other_penalised": any(collision_penalty(x) for x in rows if x["driver_id"] in involved),
    }


# evidence the cause model uses; "spun" messages (like "STOPPED") are not used because they
# stop appearing in 2024 while their causes do not
FEATURES = ["lap1", "contact_msg", "multi_car", "neutralised", "in_pits", "slow_before"]


def fit_cause_model(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Multinomial logistic regression (intercept + features), L2 on the slopes."""
    k, p = len(CLASSES), X.shape[1] + 1
    Xi = np.hstack([np.ones((len(X), 1)), X])
    Y = np.eye(k)[y]

    def loss(w):
        W = w.reshape(p, k)
        z = Xi @ W
        z -= z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        g = Xi.T @ (np.exp(logp) - Y)
        g[1:] += l2 * W[1:]
        return -(Y * logp).sum() + 0.5 * l2 * (W[1:] ** 2).sum(), g.ravel()

    return minimize(loss, np.zeros(p * k), jac=True, method="L-BFGS-B").x.reshape(p, k)


def predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = np.hstack([np.ones((len(X), 1)), X]) @ W
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def validate_cause_model(R: pd.DataFrame) -> dict:
    """Leave-one-season-out on coded retirements: log loss vs base rates, calibration."""
    coded = R[R.cls.notna() & (~R.status.isin(UNCODED)) & R.fastf1]
    X, y = coded[FEATURES].to_numpy(float), coded.cls.map(CLASSES.index).to_numpy()
    seasons = coded.season.to_numpy()
    probs, base = np.zeros((len(coded), len(CLASSES))), np.zeros((len(coded), len(CLASSES)))
    for s in np.unique(seasons):
        tr, te = seasons != s, seasons == s
        probs[te] = predict(fit_cause_model(X[tr], y[tr]), X[te])
        base[te] = np.bincount(y[tr], minlength=len(CLASSES)) / tr.sum()
    ll = lambda P: float(-np.mean(np.log(P[np.arange(len(y)), y])))  # noqa: E731
    bins = pd.cut(probs[:, 0], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    calib = pd.DataFrame({"bin": bins, "p": probs[:, 0], "mech": y == 0}).groupby("bin", observed=True).agg(
        n=("p", "size"), mean_predicted=("p", "mean"), observed=("mech", "mean"))
    by_season = {int(s): {"n": int((seasons == s).sum()),
                          "log_loss_model": float(-np.mean(np.log(probs[seasons == s][
                              np.arange((seasons == s).sum()), y[seasons == s]]))),
                          "log_loss_base_rates": float(-np.mean(np.log(base[seasons == s][
                              np.arange((seasons == s).sum()), y[seasons == s]])))}
                 for s in np.unique(seasons)}
    return {"n": int(len(coded)), "classes": CLASSES,
            "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
            "log_loss_model": ll(probs), "log_loss_base_rates": ll(base),
            "accuracy_model": float((probs.argmax(1) == y).mean()),
            "accuracy_base_rates": float((base.argmax(1) == y).mean()),
            "by_season": by_season,
            "calibration_p_mechanical": calib.reset_index().astype({"bin": str}).to_dict("records")}


def attribution(f) -> dict:
    # a "Collision" status means another car was involved (not a model feature: it is the label)
    if not (f.multi_car or f.status in ("Collision", "Collision damage")):
        return ATTRIBUTION["single_car"]
    if f.penalised:
        return ATTRIBUTION["penalised"]
    return ATTRIBUTION["other_penalised"] if f.other_penalised else ATTRIBUTION["multi_car"]


def retirements(race: pd.DataFrame, per_race: dict, laps_by_race: dict) -> tuple[list[dict], pd.DataFrame]:
    rows, feats = [], []
    no_laps = pd.DataFrame(columns=["driver_id", "lap_number", "pit_in_time", "lap_time", "time"])
    for event_id, g in race.groupby("event_id"):
        rr = per_race.get(event_id, [])
        laps = laps_by_race.get(event_id, no_laps)
        lap_at = leader_lap(laps) if len(laps) else None
        out = g[~g.status.str.match(FINISHED.pattern)]
        race_laps = {r.driver_id: (race_lap_of_retirement(r, laps, lap_at) if lap_at else int(r.laps) + 1)
                     for r in out.itertuples()}
        retired = {r.driver_id: race_laps[r.driver_id] for r in out.itertuples() if admin_kind(r) is None}
        for r in out.itertuples():
            lap = race_laps[r.driver_id]
            kind = admin_kind(r)
            if kind:
                rows.append(_row(event_id, r.driver_id, lap if r.laps else None, None, kind, r.status,
                                 [{"source": "jolpica status", "status": r.status,
                                   "position_text": r.position_text}]))
                continue
            f = retirement_features(r, lap, laps, rr, retired)
            feats.append({"event_id": event_id, "driver_id": r.driver_id, "status": r.status,
                          "season": int(event_id[:4]), "lap": lap, "car_lap": int(r.laps) + 1,
                          "fastf1": bool(len(laps)), "cls": status_class(r.status), **f})
    return rows, pd.DataFrame(feats)


def cause_probabilities(R: pd.DataFrame, W: np.ndarray, base_rates: np.ndarray) -> pd.DataFrame:
    """Cause probabilities per retirement: from the status when coded, else from the model
    (or, without FastF1 evidence, the class base rates)."""
    P = predict(W, R[FEATURES].to_numpy(float))
    P[~R.fastf1.to_numpy()] = base_rates
    out = []
    for i, r in enumerate(R.itertuples()):
        p = dict.fromkeys(CAUSES, 0.0)
        coded = r.status not in UNCODED
        cls_p = dict(zip(CLASSES, np.eye(3)[CLASSES.index(r.cls)] if coded else P[i]))
        scale = CODED_CONFIDENCE if coded else 1.0
        p["unknown"] += 1 - scale
        p["mechanical"] += scale * cls_p["mechanical"]
        for c, v in attribution(r).items():
            p[c] += scale * cls_p["incident"] * v
        for c, v in OTHER_SPLIT.get(r.status, {"unknown": 1.0}).items():
            p[c] += scale * cls_p["other"] * v
        basis = "status code" if coded else ("inferred from evidence" if r.fastf1 else "class base rates")
        out.append({"event_id": r.event_id, "driver_id": r.driver_id, "basis": basis,
                    **{f"p_{k}_class": v for k, v in cls_p.items()}, **{f"p_{c}": v for c, v in p.items()}})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- build

def build(t: dict) -> tuple[pd.DataFrame, dict]:
    race = t["race"][t["race"].event_id.str[:4].astype(int) >= FIRST_SEASON]
    rows, per_race, laps_by_race = [], {}, {}
    for event_id, laps in t["race_laps"].groupby("event_id"):
        laps_by_race[event_id] = laps
        lap_at = leader_lap(laps)
        cls = t["race_classification"][t["race_classification"].event_id == event_id]
        number_to_driver = dict(zip(cls.driver_number.astype(str), cls.driver_id))
        rr = neutralisations(event_id, t["race_track_status"][t["race_track_status"].event_id == event_id], lap_at)
        rr += weather_changes(event_id, t["race_weather"][t["race_weather"].event_id == event_id], lap_at)
        rr += message_events(event_id, t["race_messages"][t["race_messages"].event_id == event_id],
                             number_to_driver)
        rr += pit_events(event_id, laps, rr)
        rr += team_orders(event_id, laps)
        per_race[event_id] = rr
        rows += rr
    admin_rows, R = retirements(race, per_race, laps_by_race)
    validation = validate_cause_model(R)
    coded = R[R.cls.notna() & (~R.status.isin(UNCODED)) & R.fastf1]
    W = fit_cause_model(coded[FEATURES].to_numpy(float), coded.cls.map(CLASSES.index).to_numpy())
    base = np.bincount(coded.cls.map(CLASSES.index), minlength=len(CLASSES)) / len(coded)
    C = cause_probabilities(R, W, base)
    for r in R.merge(C, on=["event_id", "driver_id"]).itertuples():
        f = {k: bool(getattr(r, k)) for k in [*FEATURES, "spun_msg", "penalised", "other_penalised"]}
        rows.append(_row(r.event_id, r.driver_id, int(r.lap), None, "retirement", r.status,
                         [{"source": "jolpica status", "status": r.status, "cause_basis": r.basis,
                           "evidence": [k for k, v in f.items() if v]}], car_lap=int(r.car_lap),
                         **{c: getattr(r, c) for c in P_COLS}))
    rows += admin_rows
    T = pd.DataFrame(rows)
    T["reviewed"] = False
    T["cause_basis"] = np.where(T.kind == "retirement",
                                T.evidence.map(lambda e: json.loads(e)[0].get("cause_basis", "")), "")
    src = t["sources"]["races"]
    T["source"] = T.event_id.map(lambda e: f"fastf1 {src[e]['fastf1_version']} retrieved {src[e]['retrieved_utc'][:10]}"
                                 if e in src else "jolpica only")
    T["version"] = VERSION
    T = apply_overrides(T)
    T = T.sort_values(["event_id", "lap_start", "kind", "driver_id"], na_position="first", ignore_index=True)
    model = {"features": FEATURES, "classes": CLASSES,
             "coefficients": pd.DataFrame(W, index=["intercept", *FEATURES], columns=CLASSES).round(3).to_dict(),
             "validation": validation}
    return T, {"retirements": R.merge(C, on=["event_id", "driver_id"]), "cause_model": model}


def apply_overrides(T: pd.DataFrame) -> pd.DataFrame:
    if not OVERRIDES.exists():
        return T
    ov = pd.read_csv(OVERRIDES, comment="#")
    key = ["event_id", "driver_id", "kind", "lap_start"]
    for o in ov.itertuples():
        m = (T.event_id == o.event_id) & (T.driver_id == o.driver_id) & (T.kind == o.kind) & (T.lap_start == o.lap_start)
        if not m.any():
            raise ValueError(f"override matches no timeline row: {dict(zip(key, (o.event_id, o.driver_id, o.kind, o.lap_start)))}")
        for c in P_COLS:
            T.loc[m, c] = getattr(o, c)
        T.loc[m, ["reviewed", "cause_basis"]] = [True, "reviewed"]
        T.loc[m, "evidence"] = T.loc[m, "evidence"].map(
            lambda e: json.dumps(json.loads(e) + [{"source": "review", "note": o.note, "reviewer": o.reviewer,
                                                  "date": o.reviewed_on}]))
    return T


def changes(old: pd.DataFrame | None, new: pd.DataFrame) -> dict:
    if old is None:
        return {"previous_build": None}
    key = ["event_id", "driver_id", "kind", "lap_start", "detail"]
    ident = lambda d: d[key].map(str).agg("|".join, axis=1)  # noqa: E731
    o = old.assign(k=ident(old)).drop_duplicates("k").set_index("k")
    n = new.assign(k=ident(new)).drop_duplicates("k").set_index("k")
    o = o.reindex(columns=n.columns)
    both = o.index.intersection(n.index)
    moved = (o.loc[both, P_COLS].fillna(0) - n.loc[both, P_COLS].fillna(0)).abs().max(axis=1)
    return {"previous_build": str(old.version.iloc[0]) if "version" in old else None,
            "added": int(len(n.index.difference(o.index))), "removed": int(len(o.index.difference(n.index))),
            "cause_changed_by_0.05_plus": moved[moved >= 0.05].index.tolist()[:200]}


def audit(T: pd.DataFrame, extra: dict, diff: dict) -> dict:
    T = T.assign(season=T.event_id.str[:4].astype(int))
    R = extra["retirements"]
    ret = T[T.kind == "retirement"]
    summary = {
        "version": VERSION, "rows": int(len(T)), "races": int(T.event_id.nunique()),
        "by_kind_and_season": T.groupby(["kind", "season"]).size().unstack(fill_value=0).to_dict("index"),
        "retirements": {
            "n": int(len(ret)), "reviewed": int(ret.reviewed.sum()),
            "by_basis": ret.cause_basis.value_counts().to_dict(),
            "mean_cause_probabilities_by_basis": ret.groupby("cause_basis")[P_COLS].mean().round(3).to_dict("index"),
            "feature_rates_by_season": R[R.fastf1].groupby("season")[FEATURES].mean().round(3).to_dict("index"),
        },
        "cause_model": extra["cause_model"], "attribution_assumptions": ATTRIBUTION,
        "changes_since_previous_build": diff,
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "audit.json").write_text(json.dumps(summary, indent=1, default=str))
    R.to_csv(AUDIT / "retirements.csv", index=False)
    (AUDIT / "AUDIT.md").write_text(audit_markdown(T, summary))
    return summary


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def audit_markdown(T: pd.DataFrame, s: dict) -> str:
    v = s["cause_model"]["validation"]
    ret = s["retirements"]
    L = [f"# Race event timeline: audit ({VERSION})\n",
         f"{s['rows']:,} rows over {s['races']} races (outcomes from {FIRST_SEASON}, race-control and lap "
         "evidence from 2018). Built by `python -m f1rank.timeline` from "
         "`data/processed/race_*.parquet` (FastF1) and Jolpica results. Rules produce evidence, not verdicts.\n",
         "## Rows by kind and season\n",
         _md(pd.DataFrame(s["by_kind_and_season"]).T.rename_axis("kind").reset_index()), "",
         "## Retirement causes\n",
         f"{ret['n']} retirements; cause basis: " + ", ".join(f"{k} {n}" for k, n in ret["by_basis"].items())
         + f"; reviewed: {ret['reviewed']}.\n",
         "Mean cause probabilities by basis:\n",
         _md(pd.DataFrame(ret["mean_cause_probabilities_by_basis"]).T.rename_axis("basis").reset_index()), "",
         "### Cause model (status \"Retired\")\n",
         f"Multinomial logistic model of the class ({', '.join(CLASSES)}) given evidence "
         f"({', '.join(FEATURES)}), fitted on {v['n']} retirements with coded statuses "
         f"({', '.join(f'{k} {n}' for k, n in v['class_counts'].items())}). Leave-one-season-out: log loss "
         f"{v['log_loss_model']:.3f} vs {v['log_loss_base_rates']:.3f} for the training base rates; accuracy "
         f"{v['accuracy_model']:.2f} vs {v['accuracy_base_rates']:.2f}. It is applied to retirements coded only "
         "as \"Retired\" (almost all from 2023), whose evidence rates are compared below: the model assumes "
         "the evidence means the same in those seasons.\n",
         _md(pd.DataFrame(v["by_season"]).T.astype({"n": int}).rename_axis("held-out season").reset_index()), "",
         "Calibration of P(mechanical), held-out seasons:\n",
         _md(pd.DataFrame(v["calibration_p_mechanical"])), "",
         "Evidence rates among retirements, by season:\n",
         _md(pd.DataFrame(ret["feature_rates_by_season"]).T.rename_axis("season").reset_index()), "",
         "### Attribution of the incident class (assumptions, not estimates)\n",
         _md(pd.DataFrame(s["attribution_assumptions"]).T.fillna(0.0).rename_axis("evidence").reset_index()), "",
         "`penalised`: a collision-related penalty for the car in race control's messages from two laps "
         "before the retirement on. Penalties decided after the race are not in the messages, so their "
         "cases fall under `multi_car`.\n",
         "## Changes since the previous build\n",
         "```", json.dumps(s["changes_since_previous_build"], indent=1)[:4000], "```", ""]
    return "\n".join(L)


def main() -> None:
    t = load()
    path = PROCESSED / "timeline.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    T, extra = build(t)
    diff = changes(old, T)
    T.to_parquet(path, index=False)
    s = audit(T, extra, diff)
    v = s["cause_model"]["validation"]
    print(f"{len(T)} rows over {s['races']} races; retirements {s['retirements']['n']} "
          f"({s['retirements']['by_basis']})")
    print(f"cause model, leave-one-season-out on {v['n']} coded retirements: log loss {v['log_loss_model']:.3f} "
          f"vs base rates {v['log_loss_base_rates']:.3f}; accuracy {v['accuracy_model']:.2f} vs "
          f"{v['accuracy_base_rates']:.2f}")
    print(json.dumps(diff)[:300])


if __name__ == "__main__":
    main()
