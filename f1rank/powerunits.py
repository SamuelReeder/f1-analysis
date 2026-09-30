"""Power-unit supplier for each team-season, from each season's Wikipedia entry list.

    python -m f1rank.powerunits

Neither Jolpica nor FastF1 records engines, and the reliability model (docs/racing_approach.md:
mechanical risk by team-season, power-unit supplier and era) needs them. For each season the
rendered article "<season> Formula One World Championship" is downloaded once to
data/raw/wikipedia/ (retrieval time and sha256 in manifest.json), the entries table (the one
with a "Power unit" or "Engine" column) is parsed with its merged cells expanded, and each
constructor is mapped onto a team lineage (lineage.py) and each power unit onto a supplier:

    Renault (including TAG Heuer badging), Ferrari, Mercedes, Honda (including the
    Honda-built units badged RBPT, 2022-2025), Cosworth, Red Bull Ford, Audi

The constructor names printed there are "chassis-engine"; the chassis part identifies the
lineage. Writes data/reference/power_units.csv (season, team, constructor and power unit as
printed, supplier, and a note where a badge names no supplier: BADGES). Fails if a team-season in data/processed/race.parquet has no supplier,
or has more than one.
"""

import datetime as dt
import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "wikipedia"
OUT = ROOT / "data" / "reference" / "power_units.csv"
URL = "https://en.wikipedia.org/api/rest_v1/page/html/{season}_Formula_One_World_Championship"
HEADERS = {"User-Agent": "f1-analysis research script (python requests)"}

SUPPLIERS = [  # first match wins
    (r"red bull ford|ford", "Red Bull Ford"),
    (r"honda|rbpt", "Honda"),
    (r"tag heuer|renault", "Renault"),
    (r"ferrari", "Ferrari"),
    (r"mercedes", "Mercedes"),
    (r"cosworth", "Cosworth"),
    (r"audi", "Audi"),
]
TEAMS = [  # chassis part of the constructor as printed -> lineage (first match wins)
    (r"toro rosso|alphatauri|racing bulls|^rb$|visa cash app", "faenza"),
    (r"red bull", "red_bull"),
    (r"force india|racing point|aston martin", "silverstone"),
    (r"sauber|alfa romeo|audi", "hinwil"),
    (r"^renault|alpine|lotus f1", "enstone"),
    (r"caterham", "leafield"),
    (r"virgin|marussia|manor|^mrt$", "banbury"),
    (r"^hrt|hispania", "hrt"),
    (r"mercedes", "brackley"),
    (r"ferrari", "ferrari"),
    (r"mclaren", "mclaren"),
    (r"williams", "williams"),
    (r"haas", "haas"),
    (r"cadillac", "cadillac"),
]


def team_of(constructor: str, season: int) -> str | None:
    """Lineage from the chassis part (before the engine name). "Lotus" was the Leafield team
    (Lotus Racing, Team Lotus) in 2010-2011 and the Enstone team from 2012."""
    chassis = constructor.split("-")[0].strip()
    if re.fullmatch(r"(team )?lotus( racing)?", chassis.lower()):
        return "leafield" if season <= 2011 else "enstone"
    return match(chassis, TEAMS)


# units printed under a badge that names no supplier: (season, lineage) -> (supplier, note)
BADGES = {(2017, "faenza"): ("Renault", "Renault units badged 'Toro Rosso' in 2017")}


class Tables(HTMLParser):
    """Every table as a grid of cell texts, with rowspan and colspan expanded."""

    def __init__(self):
        super().__init__()
        self.tables, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.stack.append({"rows": [], "row": None, "cell": None})
        elif not self.stack:
            return
        elif tag == "tr":
            self.stack[-1]["row"] = []
        elif tag in ("td", "th"):
            span = lambda k: int(re.sub(r"\D", "", a.get(k, "1")) or 1)  # noqa: E731
            self.stack[-1]["cell"] = {"text": "", "rows": span("rowspan"), "cols": span("colspan")}
        elif tag == "br" and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"]["text"] += " "
        elif tag in ("sup", "style"):
            self.stack[-1]["skip"] = self.stack[-1].get("skip", 0) + 1

    def handle_endtag(self, tag):
        if not self.stack:
            return
        t = self.stack[-1]
        if tag in ("sup", "style"):
            t["skip"] = max(0, t.get("skip", 0) - 1)
        elif tag in ("td", "th") and t["cell"] is not None and t["row"] is not None:
            t["row"].append(t["cell"])
            t["cell"] = None
        elif tag == "tr" and t["row"] is not None:
            t["rows"].append(t["row"])
            t["row"] = None
        elif tag == "table":
            self.tables.append(grid(self.stack.pop()["rows"]))

    def handle_data(self, data):
        if self.stack and self.stack[-1]["cell"] is not None and not self.stack[-1].get("skip"):
            self.stack[-1]["cell"]["text"] += data


def grid(rows: list) -> list[list[str]]:
    out, pending = [], {}  # pending[(row, col)] = text carried down by a rowspan
    for i, row in enumerate(rows):
        line, j = [], 0
        cells = iter(row)
        while True:
            while (i, j) in pending:
                line.append(pending.pop((i, j)))
                j += 1
            c = next(cells, None)
            if c is None:
                break
            text = " ".join(c["text"].replace("\xa0", " ").split())
            for _ in range(c["cols"]):
                for k in range(1, c["rows"]):
                    pending[(i + k, j)] = text
                line.append(text)
                j += 1
        out.append(line)
    return out


def page(season: int) -> str:
    path = RAW / f"{season}_championship.html"
    if path.exists():
        return path.read_text()
    for attempt in range(6):
        resp = requests.get(URL.format(season=season), headers=HEADERS, timeout=60)
        if resp.status_code != 429:
            break
        time.sleep(30 * 2 ** attempt)
    resp.raise_for_status()
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text)
    man = RAW / "manifest.json"
    m = json.loads(man.read_text()) if man.exists() else {}
    m[path.name] = {"url": URL.format(season=season),
                    "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "sha256": hashlib.sha256(resp.text.encode()).hexdigest()}
    man.write_text(json.dumps(m, indent=1, sort_keys=True))
    time.sleep(5)
    return resp.text


def match(text: str, rules) -> str | None:
    t = text.lower().strip()
    return next((v for pat, v in rules if re.search(pat, t)), None)


def entries(season: int) -> pd.DataFrame:
    p = Tables()
    p.feed(page(season))
    for g in p.tables:
        head = [h.lower() for h in g[0]] if g else []
        pu = next((i for i, h in enumerate(head) if h.startswith("power unit") or h == "engine"), None)
        con = next((i for i, h in enumerate(head) if h.startswith("constructor")), None)
        if pu is None or con is None:
            continue
        rows = [(r[con], r[pu]) for r in g[1:] if len(r) > max(pu, con)]
        df = pd.DataFrame(rows, columns=["constructor", "power_unit"]).drop_duplicates()
        df = df[df.constructor.str.len() > 0]
        df["team"] = df.constructor.map(lambda c: team_of(c, season))
        df["supplier"] = df.power_unit.map(lambda s: match(s, SUPPLIERS))
        df["note"] = ""
        for (s_, team), (supplier, note) in BADGES.items():
            m = (season == s_) & (df.team == team) & df.supplier.isna()
            df.loc[m, ["supplier", "note"]] = supplier, note
        return df.assign(season=season)
    raise ValueError(f"{season}: no entries table with a power unit column")


def main() -> None:
    race = pd.read_parquet(ROOT / "data" / "processed" / "race.parquet")
    race["season"] = race.event_id.str[:4].astype(int)
    need = race[race.season >= 2010][["season", "team"]].drop_duplicates()
    E = pd.concat([entries(s) for s in sorted(need.season.unique())], ignore_index=True)
    E = E.dropna(subset=["team", "supplier"]).drop_duplicates(["season", "team", "supplier"])
    multi = E.groupby(["season", "team"]).supplier.nunique()
    if (multi > 1).any():
        raise ValueError(f"more than one supplier:\n{E.set_index(['season', 'team']).loc[multi[multi > 1].index]}")
    missing = need.merge(E, on=["season", "team"], how="left")
    if missing.supplier.isna().any():
        raise ValueError(f"no supplier for:\n{missing[missing.supplier.isna()]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = need.merge(E, on=["season", "team"]).sort_values(["season", "team"])
    out[["season", "team", "constructor", "power_unit", "supplier", "note"]].to_csv(OUT, index=False)
    print(out.groupby("season").supplier.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
