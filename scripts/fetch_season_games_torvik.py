"""
fetch_season_games_torvik.py — Scrape game-by-game results from Bart Torvik for H2H analysis.

Fetches the complete game log for every D1 team for a given year (or range of years),
deduplicates (each game appears in both teams' logs), and writes:
  data/season_games_YYYY.json

These files power the head-to-head feature in engine.get_head_to_head().

Usage:
  python scripts/fetch_season_games_torvik.py 2026
  python scripts/fetch_season_games_torvik.py 2008 2026       # all years 2008-2026
  python scripts/fetch_season_games_torvik.py 2026 --refresh  # re-fetch even if file exists

Torvik game log URL:
  https://barttorvik.com/trankings.php?year=YYYY&type=all
  or per-team:
  https://barttorvik.com/trank.php?year=YYYY&team=TEAM_NAME&conlimit=All#

We use the team-level CSV export which gives individual game results:
  https://barttorvik.com/getgamestats.php?year=YYYY&team=TEAM&type=all&begin=20230101&end=20230401&venue=All

Politeness: 0.25s delay between requests. ~380 teams × 0.25s = ~95s per year.
Estimated total for 2008-2026 (18 years): ~28 minutes. Run once and cache.
"""
import csv
import io
import json
import os
import re
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, ROOT)

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

from engine import _normalize_team_for_match

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TORVIK_GAME_STATS_URL = (
    "https://barttorvik.com/getgamestats.php"
    "?year={year}&team={team}&type=all&venue=All"
)
TORVIK_TEAM_LIST_URL = (
    "https://barttorvik.com/trankings.php?year={year}&type=all"
)
REQUEST_DELAY = 0.3   # seconds between requests — be polite

# ---------------------------------------------------------------------------
# Team list from Torvik
# ---------------------------------------------------------------------------

def fetch_team_list(year: int) -> list[str]:
    """Fetch list of all D1 teams for a given year from Torvik rankings."""
    # First try: load from teams_merged file (we already have this)
    path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        teams = [v.get("team", k) for k, v in data.items() if isinstance(v, dict)]
        print(f"  Using {len(teams)} teams from teams_merged_{year}.json")
        return teams

    # Fallback: parse the Torvik rankings page
    try:
        url = TORVIK_TEAM_LIST_URL.format(year=year)
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "BracketBrain-Research/1.0"})
        r.raise_for_status()
        # Extract team names from HTML (they appear in a specific pattern)
        teams = re.findall(r'team=([^&"]+)', r.text)
        teams = list({t.replace("+", " ").replace("%27", "'") for t in teams if len(t) > 2})
        print(f"  Found {len(teams)} teams from Torvik rankings page")
        return sorted(teams)
    except Exception as e:
        print(f"  WARN: failed to fetch team list for {year}: {e}")
        return []


# ---------------------------------------------------------------------------
# Per-team game log
# ---------------------------------------------------------------------------

def _parse_torvik_game_row(row: dict, team_name: str, year: int) -> dict | None:
    """Parse one row from Torvik game stats CSV into a game dict."""
    # Torvik columns (approximate — may vary by year):
    # Date, Opponent, Location (H/A/N), Pts, Opp Pts, Result (W/L), ...
    try:
        date_raw = row.get("Date") or row.get("date") or ""
        opp = (row.get("Opp") or row.get("Opponent") or row.get("opp") or "").strip()
        loc = (row.get("Loc") or row.get("Location") or row.get("loc") or "N").strip().upper()
        pts_raw = row.get("Pts") or row.get("pts") or row.get("PTS") or "0"
        opp_pts_raw = row.get("OppPts") or row.get("Opp Pts") or row.get("opp_pts") or "0"

        if not opp or not date_raw:
            return None

        try:
            pts = int(str(pts_raw).strip())
            opp_pts = int(str(opp_pts_raw).strip())
        except (ValueError, TypeError):
            return None

        # Determine home/away — Torvik uses H=home, A=away, N=neutral
        if loc == "H":
            home_team, away_team = team_name, opp
            home_score, away_score = pts, opp_pts
        elif loc == "A":
            home_team, away_team = opp, team_name
            home_score, away_score = opp_pts, pts
        else:  # neutral
            home_team, away_team = team_name, opp
            home_score, away_score = pts, opp_pts

        winner = home_team if home_score > away_score else away_team
        margin = abs(home_score - away_score)

        # Normalize date to YYYY-MM-DD
        date_str = str(date_raw).strip()
        # Torvik format is usually MM/DD/YYYY or YYYYMMDD
        if re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
            m, d, y = date_str.split("/")
            date_str = f"{y}-{int(m):02d}-{int(d):02d}"
        elif re.match(r"\d{8}", date_str):
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        return {
            "date": date_str,
            "team_a": home_team,   # home (or team_name at neutral)
            "team_b": away_team,
            "score_a": home_score,
            "score_b": away_score,
            "winner": winner,
            "margin": margin,
            "location": loc,
            "game_type": "regular_season",
            "year": year,
        }
    except Exception:
        return None


def fetch_game_log(team: str, year: int) -> list[dict]:
    """Fetch game-by-game log for one team from Torvik."""
    team_encoded = team.replace(" ", "+").replace("'", "%27")
    url = TORVIK_GAME_STATS_URL.format(year=year, team=team_encoded)
    try:
        r = requests.get(url, timeout=12,
                         headers={"User-Agent": "BracketBrain-Research/1.0"})
        r.raise_for_status()
        content = r.text.strip()
        if not content:
            return []
        reader = csv.DictReader(io.StringIO(content))
        games = []
        for row in reader:
            g = _parse_torvik_game_row(row, team, year)
            if g:
                games.append(g)
        return games
    except Exception as e:
        return []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _game_key(g: dict) -> frozenset:
    """Stable key for deduplication — same game from both teams' logs."""
    return frozenset([
        g.get("date", ""),
        _normalize_team_for_match(g.get("team_a", "")),
        _normalize_team_for_match(g.get("team_b", "")),
    ])


def deduplicate(games: list[dict]) -> list[dict]:
    """Remove duplicate game records (same game scraped from both teams)."""
    seen = set()
    out = []
    for g in games:
        k = _game_key(g)
        if k not in seen:
            seen.add(k)
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_season_games(year: int, refresh: bool = False) -> bool:
    """
    Fetch all regular-season games for a given year and write to
    data/season_games_YYYY.json. Returns True on success.
    """
    out_path = os.path.join(DATA_DIR, f"season_games_{year}.json")
    if os.path.isfile(out_path) and not refresh:
        with open(out_path) as f:
            existing = json.load(f)
        print(f"  {year}: {len(existing)} games already in {os.path.basename(out_path)} (use --refresh to re-fetch)")
        return True

    print(f"\n{'='*60}")
    print(f"  Fetching season games for {year}...")
    print(f"{'='*60}")

    teams = fetch_team_list(year)
    if not teams:
        print(f"  ERROR: no teams found for {year}")
        return False

    all_games: list[dict] = []
    failed = 0

    for i, team in enumerate(teams, 1):
        games = fetch_game_log(team, year)
        all_games.extend(games)
        if games:
            print(f"  [{i:3d}/{len(teams)}] {team}: {len(games)} games")
        else:
            failed += 1
        time.sleep(REQUEST_DELAY)

    # Dedup and sort by date
    unique = deduplicate(all_games)
    unique.sort(key=lambda g: g.get("date", ""))

    with open(out_path, "w") as f:
        json.dump(unique, f, indent=2)

    print(f"\n  {year}: {len(unique)} unique games written ({failed} teams had no data)")
    print(f"  → {out_path}")
    return len(unique) > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Torvik season game logs for H2H analysis")
    parser.add_argument("years", nargs="+", type=int,
                        help="Year(s) to fetch. Pass two years for a range: 2008 2026")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch even if output file exists")
    args = parser.parse_args()

    if len(args.years) == 2 and args.years[0] < args.years[1]:
        year_list = list(range(args.years[0], args.years[1] + 1))
        year_list = [y for y in year_list if y != 2020]  # no 2020 tournament
    else:
        year_list = args.years

    print(f"Fetching season games for years: {year_list}")
    success = 0
    for year in year_list:
        ok = fetch_season_games(year, refresh=args.refresh)
        if ok:
            success += 1

    print(f"\nDone: {success}/{len(year_list)} years fetched successfully.")


if __name__ == "__main__":
    main()
