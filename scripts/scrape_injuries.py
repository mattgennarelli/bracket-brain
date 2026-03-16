"""
Scrape injury data from Action Network NCAAB injury report.

Output: data/injuries_YYYY.json in same format as fetch_injuries.py
  {"Duke": [{"player": "Cameron Boozer", "status": "out", "bpr_share": 0.32, "importance": 1.0}]}

Uses EvanMiya players CSV to enrich bpr_share when available.

Usage:
  python scripts/scrape_injuries.py 2026
  python scripts/scrape_injuries.py 2026 --filter-tournament  # only teams in bracket/teams_merged

Dependencies: beautifulsoup4, httpx (or requests)
"""
import json
import logging
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from engine import _normalize_team_for_match

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("scrape_injuries")

URL = "https://www.actionnetwork.com/ncaab/injury-report"

# Status mapping: Action Network -> engine
STATUS_MAP = {
    "out": "out",
    "out for season": "out",
    "out indefinitely": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "probable": "probable",
}


def _normalize_status(status_str):
    """Map Action Network status to engine status."""
    if not status_str:
        return "out"
    s = str(status_str).strip().lower()
    for k, v in STATUS_MAP.items():
        if k in s:
            return v
    if "?" in s or "questionable" in s:
        return "questionable"
    if "probable" in s:
        return "probable"
    if "doubtful" in s:
        return "doubtful"
    return "out"


def _normalize_team_for_torvik(name):
    """Convert Action Network team name to Torvik-style (e.g. Duke Blue Devils -> Duke)."""
    if not name:
        return ""
    n = " ".join(name.strip().split())
    # Common suffixes to strip
    suffixes = [
        " Blue Devils", " Tar Heels", " Wildcats", " Crimson Tide", " Tigers",
        " Bulldogs", " Huskies", " Hoosiers", " Spartans", " Wolverines",
        " Fighting Irish", " Bears", " Cougars", " Aggies", " Volunteers",
        " Razorbacks", " Rebels", " Volunteers", " Gamecocks", " Commodores",
        " Jayhawks", " Longhorns", " Sooners", " Mountaineers", " Seminoles",
        " Hurricanes", " Gators", " Cardinals", " Hoyas", " Friars",
        " Musketeers", " Pirates", " Red Storm", " Golden Eagles", " Flyers",
        " Bonnies", " Billikens", " Gaels", " Zags", " Antelopes",
    ]
    for suf in suffixes:
        if n.endswith(suf):
            return n[: -len(suf)].strip()
    return n


def scrape_action_network():
    """Scrape Action Network injury page. Returns dict team -> list of injury dicts."""
    try:
        import requests
    except ImportError:
        _log.error("Need requests. pip install requests")
        return {}

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        _log.error("Need beautifulsoup4. pip install beautifulsoup4")
        return {}

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        resp = requests.get(URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        _log.error("Failed to fetch %s: %s", URL, e)
        return {}

    injuries_by_team = {}
    current_team = None

    def _is_position(s):
        if not s:
            return False
        s = s.upper()
        return s in ("G", "F", "C") or "/" in s

    tables = soup.find_all("table")
    for table in tables:
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            texts = [(c.get_text() or "").strip() for c in cells]
            first = texts[0] if texts else ""
            if not first or first == "Name":
                continue

            # Player row: Name | Pos (G/F/C) | Status | ...
            if len(texts) >= 4 and _is_position(texts[1]):
                status = _normalize_status(texts[2])
                if current_team:
                    injuries_by_team.setdefault(current_team, []).append({
                        "player": first,
                        "status": status,
                        "bpr_share": 0.5,
                        "importance": 0.5,
                    })
                continue

            # Team header row: single cell or team name in first col
            if len(cells) <= 2 or not texts[1]:
                if len(first) > 2:
                    current_team = _normalize_team_for_torvik(first)
                    if current_team:
                        injuries_by_team.setdefault(current_team, [])

    return injuries_by_team


def fetch_tournament_teams(year):
    """Load team names from bracket or teams_merged for --filter-tournament."""
    teams = set()
    for fname in ["bracket_{}.json", "teams_merged_{}.json"]:
        path = os.path.join(DATA_DIR, fname.format(year))
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    t = row.get("team") or row.get("Team")
                    if t:
                        teams.add(_normalize_team_for_match(t))
            elif isinstance(data, dict):
                for rname, region in data.get("regions", {}).items():
                    for team_obj in (region.values() if isinstance(region, dict) else region):
                        t = team_obj.get("team") if isinstance(team_obj, dict) else None
                        if t:
                            teams.add(_normalize_team_for_match(t))
        except Exception:
            pass
    return teams


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape injuries from Action Network")
    parser.add_argument("year", type=int, nargs="?", default=2026)
    parser.add_argument("--filter-tournament", action="store_true",
                       help="Only include teams in bracket/teams_merged")
    parser.add_argument("--enrich", action="store_true",
                       help="Enrich bpr_share from EvanMiya (requires fetch_injuries)")
    args = parser.parse_args()
    year = args.year

    _log.info("Scraping %s ...", URL)
    injuries = scrape_action_network()
    if not injuries:
        _log.warning("No injuries parsed. Action Network page structure may have changed.")
        sys.exit(1)

    if args.filter_tournament:
        tournament_teams = fetch_tournament_teams(year)
        if tournament_teams:
            filtered = {}
            for team, inj_list in injuries.items():
                key = _normalize_team_for_match(team)
                if not key:
                    continue
                matched = False
                for t in tournament_teams:
                    if key == t or (key in t or t in key):
                        matched = True
                        break
                if matched:
                    filtered[team] = inj_list
            injuries = filtered
            _log.info("Filtered to %d tournament teams", len(injuries))

    if args.enrich:
        try:
            sys.path.insert(0, SCRIPT_DIR)
            from fetch_injuries import enrich_injuries_with_bpr
            enrich_injuries_with_bpr(injuries, year)
        except Exception as e:
            _log.warning("Enrichment failed: %s", e)

    out_path = os.path.join(DATA_DIR, f"injuries_{year}.json")
    with open(out_path, "w") as f:
        json.dump(injuries, f, indent=2)
    print(f"Wrote {len(injuries)} teams with injuries -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
