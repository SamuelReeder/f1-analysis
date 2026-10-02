"""Upgrades declared to the FIA ("Car Presentation Submissions"), 2024 onward, for the
pre-registered race feature tests (docs/race_features.md).

    .venv-fastf1/bin/python extract/fia_upgrades.py     (needs pdfplumber in that environment)

For each season page of the FIA's Formula One documents it lists every event's documents
(the request the page itself makes for one event), downloads the event's "Car
Presentation Submissions" PDF into data/raw/fia/ (a rerun reuses the file) and reads each
team's table with pdfplumber: every declared component with its primary reason (a reason
cell merged over several components applies to each). Per team it counts the components
whose reason is "Performance" (including a Performance subcategory written without its
prefix: "Local Load", "Flow Conditioning", "Drag Reduction") and those whose reason is
"Circuit specific". The FIA published no such document before 2024 round 2.

Writes data/supplements/fia_upgrades.json (read by f1rank.race_features).
"""

import datetime as dt
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "fia"
SUPPLEMENT = ROOT / "data" / "supplements" / "fia_upgrades.json"
SITE = "https://www.fia.com"
SEASONS = {
    2024: "/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043",
    2025: "/documents/championships/fia-formula-one-world-championship-14/season/season-2025-2071",
    2026: "/documents/championships/fia-formula-one-world-championship-14/season/season-2026-2072",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (f1-analysis research; FIA upgrade documents)"}
# Team lineages (entries.team). Racing Bulls is tested before Red Bull.
TEAMS = [
    (r"RACING BULLS|VISA CASH APP|\bRB F1\b|\bRB FORMULA", "faenza"),
    (r"RED BULL", "red_bull"),
    (r"MERCEDES", "brackley"),
    (r"FERRARI", "ferrari"),
    (r"MCLAREN", "mclaren"),
    (r"ASTON MARTIN", "silverstone"),
    (r"ALPINE", "enstone"),
    (r"WILLIAMS", "williams"),
    (r"HAAS", "haas"),
    (r"SAUBER|STAKE|\bKICK\b|AUDI", "hinwil"),
    (r"CADILLAC", "cadillac"),
]
# The template's primary reasons: "Performance - Local Load / Flow Conditioning / Drag
# Reduction", "Circuit specific - Balance / Cooling / Drag Range", "Reliability", ...
# Some teams write a Performance subcategory without its prefix.
PERFORMANCE = re.compile(r"^(performance|local load|flow conditioning|drag reduction)", re.I)
CIRCUIT = re.compile(r"^circuit", re.I)
REASON = re.compile(r"^(performance|circuit|reliab|local load|flow cond|drag|cooling|balance|safety|other|"
                    r"structural|homologation|regulat|weight|cost)", re.I)
STRONG = re.compile(r"^(performance|circuit|reliab|other|structural)", re.I)
REASON_START = re.compile(r"\b(Performance|Circuit|Reliab|Local Load|Flow Cond|Drag|Cooling|Balance|Safety|"
                          r"Other|Structural)", re.I)
HEADER = {"updated", "component", "updated component", "primary reason", "for update", "primary reason for update"}
NONE = re.compile(r"No updates? (?:were |have been )?(?:submitted|for this event)", re.I)


def get(url: str, method: str = "GET", extra_headers: dict | None = None) -> requests.Response:
    for attempt in range(5):
        r = requests.request(method, url, headers={**HEADERS, **(extra_headers or {})}, timeout=60)
        if r.status_code < 500:
            r.raise_for_status()
            time.sleep(1)
            return r
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    return r


def event_documents(event_id: int) -> list[str]:
    r = get(f"{SITE}/decision-document-list/ajax/{event_id}", "POST", {"X-Requested-With": "XMLHttpRequest"})
    html = "".join(c.get("data") or "" for c in r.json() if isinstance(c, dict) and isinstance(c.get("data"), str))
    return re.findall(r'href="([^"]*\.pdf)"', html)


# FIA event names that differ from the calendar's (normalised)
ALIASES = {(2026, "barcelona catalunya grand prix"): "barcelona grand prix"}


def normal(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]+", " ", text.lower()).strip()


def team_of(text: str) -> str | None:
    for pattern, team in TEAMS:
        if re.search(pattern, text, re.I):
            return team
    return None


def is_heading(line: str) -> bool:
    words = line.strip("* ").split()
    return (team_of(line) is not None and len(words) <= 8 and not words[0][0].isdigit()
            and not line.rstrip().endswith((".", ",")))


def cell(c) -> str | None:
    """A table cell's text; None for the continuation of a merged cell."""
    return None if c is None else " ".join(c.split())


def components(table: list[list]) -> list[dict]:
    """Declared components of one table with their primary reasons.

    pdfplumber splits a table row into sub-rows by text line: a cell boundary reads '',
    the continuation of a merged cell None. A component is a row with its number, or a
    row whose own cell (a boundary in the first column) names a component. Its reason is
    the reason column's text in its band of sub-rows (halfway to the neighbouring
    components), which may be wrapped above and below it; a component whose band has no
    reason text and whose reason cell continues a merged cell takes the reason above.
    """
    rows = [[cell(c) for c in row] for row in table]
    width = max(len(r) for r in rows)
    if width < 3 or not any(r and r[0] and r[0].isdigit() for r in rows):
        return []  # not a component table (a page fragment)
    rows = [r + [None] * (width - len(r)) for r in rows]
    header = [any(c and c.lower() in HEADER for c in r) for r in rows]
    # reason column: the one that most often holds text reading as a reason (unambiguous
    # reason words count more than ones that also begin component names, such as
    # "Cooling Louvres"); the component is the text between it and the number
    score = {}
    for r in rows:
        for i, c in enumerate(r):
            if i > 1 and c and REASON.match(c):
                score[i] = score.get(i, 0) + (10 if STRONG.match(c) else 1)
    r_idx = max(score, key=lambda i: (score[i], -i)) if score else None
    def name(r):  # the component: text between the number and the reason columns
        return " ".join(c for c in r[1:r_idx or width] if c)
    at = [k for k, r in enumerate(rows) if not header[k]
          and ((r[0] or "").isdigit() or (r[0] == "" and name(r)))]
    out, reason = [], None
    for i, k in enumerate(at):
        lo = 0 if i == 0 else (at[i - 1] + k) // 2 + 1
        hi = len(rows) - 1 if i == len(at) - 1 else (k + at[i + 1]) // 2
        band = [j for j in range(lo, hi + 1) if not header[j]]
        text = " ".join(rows[j][r_idx] for j in band if rows[j][r_idx]) if r_idx is not None else ""
        if (m := REASON_START.search(text)):
            reason = text[m.start():]  # wrapped text of a neighbour's reason may precede it
        elif r_idx is None or rows[k][r_idx] is not None:
            reason = None
        # the name, wrapped onto continuation rows when the number sits between them
        names = [name(rows[j]) for j in band if name(rows[j]) and (j == k or rows[j][0] is None)]
        if names:  # a numbered template row left blank is not a component
            out.append({"component": " ".join(names), "reason": reason})
    return out


def parse(path: Path) -> dict:
    import pdfplumber
    teams, title, team = {}, None, None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            lines = [line.strip() for line in (page.extract_text() or "").splitlines() if line.strip()]
            if title is None:
                title = lines[0]
                continue  # the cover page
            # A team's first page starts with its name, after a "Car Presentation" heading
            # (2025 on) or alone (2024); a continuation page starts with table text.
            heading = lines[1] if len(lines) > 1 and lines[0].startswith("Car Presentation") else \
                lines[0] if lines and is_heading(lines[0]) else None
            if heading is not None:
                team = team_of(heading)
                if team is None:
                    raise ValueError(f"{path.name}: unknown team heading {heading!r}")
                teams.setdefault(team, {"components": [], "no_updates": False})
            if team is None:
                continue
            if any(NONE.search(line) for line in lines):
                teams[team]["no_updates"] = True
            for table in page.extract_tables():
                teams[team]["components"] += components(table)
                # numbers of the rows with any text on the team's pages (the FIA component
                # list's numbers, so they may skip), to check the count
                teams[team].setdefault("row_numbers", set()).update(
                    int(r[0]) for r in table if r and r[0] and r[0].strip().isdigit() and any(r[1:]))
    for t in teams.values():
        t["row_numbers"] = sorted(t.get("row_numbers", ()))
        reasons = [c["reason"] or "" for c in t["components"]]
        t["performance"] = sum(bool(PERFORMANCE.match(r)) for r in reasons)
        t["circuit_specific"] = sum(bool(CIRCUIT.match(r)) for r in reasons)
        t["no_reason"] = sum(not r for r in reasons)
    return {"title": title, "teams": teams}


def main() -> None:
    events = pd.read_parquet(ROOT / "data" / "processed" / "events.parquet")
    events = events[events.season.isin(SEASONS)]
    by_name = {(int(r.season), normal(r.race_name)): r.event_id for r in events.itertuples()}
    out, problems = {}, []
    for season, page in SEASONS.items():
        html = get(SITE + page).text
        for fia_id in sorted(set(re.findall(r"decision-document-list/nojs/(\d+)", html)), key=int):
            pdfs = [p for p in event_documents(int(fia_id))
                    if re.search(r"car[ _]presentation[ _]submissions", p, re.I)]
            if not pdfs:
                continue
            url = SITE + pdfs[0].replace(" ", "%20")
            path = RAW / str(season) / Path(pdfs[0]).name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(get(url).content)
            doc = parse(path)
            name = normal(re.sub(r"^\d{4}\s+", "", doc["title"]))
            name = ALIASES.get((season, name), name)
            event_id = by_name.get((season, name))
            if event_id is None:
                problems.append(f"{season} FIA event {fia_id}: no event named {doc['title']!r}")
                continue
            for team, t in doc["teams"].items():
                if ((t["no_updates"] and t["components"]) or t["no_reason"]
                        or len(t["row_numbers"]) > len(t["components"])):
                    problems.append(f"{event_id} {team}: no_updates={t['no_updates']}, "
                                    f"{len(t['components'])} components, row numbers {t['row_numbers']}, "
                                    f"{t['no_reason']} without a reason")
            out[event_id] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "title": doc["title"], "teams": doc["teams"]}
            print(f"{event_id}: {len(doc['teams'])} teams, "
                  f"{sum(t['performance'] for t in doc['teams'].values())} performance components", flush=True)
    SUPPLEMENT.parent.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT.write_text(json.dumps({
        "source": "FIA Formula One documents, 'Car Presentation Submissions' per event; tables parsed "
                  "with pdfplumber (extract/fia_upgrades.py)",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "problems": problems,
        "events": dict(sorted(out.items())),
    }, indent=1, ensure_ascii=False))
    print(f"{len(out)} events; {len(problems)} problems")
    for p in problems:
        print("  " + p)


if __name__ == "__main__":
    main()
