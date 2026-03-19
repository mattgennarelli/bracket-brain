"""
Fetch current NCAA men's basketball head coaches into data/coaches_YYYY.csv.

Default source is the Wikipedia table of current Division I men's coaches.
This is enough to activate current-season coach mapping in the model; it does
not create historical coach-by-season data for recalibration.

Usage:
  python scripts/fetch_coaches.py
  python scripts/fetch_coaches.py 2026
  python scripts/fetch_coaches.py 2026 --source wikipedia
"""
import csv
import os
import re
import sys
from collections import Counter

import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from scripts.fetch_data import _find_torvik_key, load_json

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_current_NCAA_Division_I_men%27s_basketball_coaches"
USER_AGENT = "BracketBrain/1.0 (+local coach fetch)"
TEAM_ALIASES = {
    "California Baptist Lancers": "Cal Baptist",
    "California Baptist": "Cal Baptist",
    "LIU Sharks": "Long Island",
    "LIU": "Long Island",
    "Miami RedHawks": "Miami OH",
    "Miami Hurricanes": "Miami FL",
    "UIC Flames": "Illinois Chicago",
    "UIC": "Illinois Chicago",
    "Kansas City Roos": "Missouri-Kansas City",
    "Kansas City": "Missouri-Kansas City",
    "UT Martin Skyhawks": "Tennessee Martin",
    "UT Martin": "Tennessee Martin",
    "UTRGV Vaqueros": "UT Rio Grande Valley",
    "UTRGV": "UT Rio Grande Valley",
}


def _clean_text(value):
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_coach_name(value):
    text = _clean_text(value)
    if not text:
        return ""
    if text.lower() == "vacant":
        return ""
    text = re.sub(r"\s*\((?:interim|acting|temporary)\)\s*$", "", text, flags=re.I)
    return text.strip()


def _map_team_name(team_name, merged_keys):
    alias = TEAM_ALIASES.get(_clean_text(team_name))
    if alias:
        mapped = _find_torvik_key(alias, merged_keys)
        if mapped:
            return mapped
    mapped = _find_torvik_key(team_name, merged_keys)
    if mapped:
        return mapped
    tokens = _clean_text(team_name).split()
    for n in range(len(tokens) - 1, 0, -1):
        candidate = " ".join(tokens[:n])
        alias = TEAM_ALIASES.get(candidate)
        if alias:
            mapped = _find_torvik_key(alias, merged_keys)
            if mapped:
                return mapped
        mapped = _find_torvik_key(candidate, merged_keys)
        if mapped:
            return mapped
    return None


def _extract_wikipedia_rows(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    for table in soup.find_all("table"):
        header_cells = table.find("tr")
        if not header_cells:
            continue
        headers = [_clean_text(th.get_text(" ", strip=True)).lower() for th in header_cells.find_all(["th", "td"])]
        if "team" not in headers or "current coach" not in headers:
            continue
        team_idx = headers.index("team")
        coach_idx = headers.index("current coach")
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) <= max(team_idx, coach_idx):
                continue
            rows.append({
                "team": _clean_text(cells[team_idx].get_text(" ", strip=True)),
                "coach": _clean_text(cells[coach_idx].get_text(" ", strip=True)),
            })
        if rows:
            return rows
    raise ValueError("Could not find coaches table with Team / Current coach columns")


def _rows_to_records(rows, teams_merged=None):
    merged_keys = set(teams_merged or [])
    out = []
    matched = 0
    for row in rows:
        team = _clean_text(row.get("team"))
        coach = _clean_coach_name(row.get("coach"))
        if not team or not coach:
            continue
        mapped_team = _map_team_name(team, merged_keys) if merged_keys else None
        if mapped_team:
            matched += 1
        out.append({
            "team": mapped_team or team,
            "coach": coach,
            "source": "wikipedia_current_coaches",
        })
    return out, matched


def fetch_wikipedia_rows(url=WIKIPEDIA_URL):
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return _extract_wikipedia_rows(response.text)


def load_merged_team_names(year):
    path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
    data = load_json(path)
    if not data:
        return []
    rows = data.values() if isinstance(data, dict) else data
    return [row.get("team") for row in rows if row.get("team")]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["team", "coach", "source"])
        writer.writeheader()
        writer.writerows(rows)


def find_duplicate_teams(rows):
    counts = Counter(row["team"] for row in rows)
    return {team: count for team, count in counts.items() if count > 1}


def main():
    year = 2026
    source = "wikipedia"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--source" and i + 1 < len(args):
            source = args[i + 1].strip().lower()
            i += 2
            continue
        try:
            year = int(a)
        except ValueError:
            pass
        i += 1

    if source != "wikipedia":
        print(f"Unsupported source: {source}")
        return 1

    teams_merged = load_merged_team_names(year)
    source_rows = fetch_wikipedia_rows()
    rows, matched = _rows_to_records(source_rows, teams_merged=teams_merged)
    out_path = os.path.join(DATA_DIR, f"coaches_{year}.csv")
    write_csv(out_path, rows)
    print(f"Wrote {len(rows)} coaches to {out_path}")
    if teams_merged:
        print(f"Matched {matched}/{len(rows)} rows directly to teams_merged_{year}.json")
    duplicates = find_duplicate_teams(rows)
    if duplicates:
        print(f"WARNING: duplicate mapped teams detected: {duplicates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
