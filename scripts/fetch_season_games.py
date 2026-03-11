"""
Fetch or load regular season game-by-game results for head-to-head matchup display.

Supports:
  - Manual CSV: data/season_games_YYYY.csv with columns:
    date, team_a, team_b, score_a, score_b, location (optional)

Output: data/season_games_YYYY.json

Usage:
  python scripts/fetch_season_games.py 2026
  python scripts/fetch_season_games.py --from-csv data/season_games_2026.csv 2026
"""
import csv
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _find_col(headers, candidates):
    """Find first matching column (case-insensitive)."""
    lower = {str(h).strip().lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_from_csv(path, year):
    """Load season games from CSV. Returns list of game dicts."""
    if not os.path.isfile(path):
        return []
    games = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        date_col = _find_col(headers, ["date", "Date", "game_date"])
        team_a_col = _find_col(headers, ["team_a", "Team A", "home", "Home"])
        team_b_col = _find_col(headers, ["team_b", "Team B", "away", "Away", "opponent", "Opponent"])
        score_a_col = _find_col(headers, ["score_a", "Score A", "home_score", "pts", "score"])
        score_b_col = _find_col(headers, ["score_b", "Score B", "away_score", "opp_score"])
        loc_col = _find_col(headers, ["location", "Location", "venue", "Venue", "site"])

        for row in reader:
            team_a = (row.get(team_a_col or "team_a") or "").strip()
            team_b = (row.get(team_b_col or "team_b") or "").strip()
            if not team_a or not team_b:
                continue
            try:
                score_a = int(row.get(score_a_col or "score_a") or 0)
            except (ValueError, TypeError):
                score_a = 0
            try:
                score_b = int(row.get(score_b_col or "score_b") or 0)
            except (ValueError, TypeError):
                score_b = 0
            winner = team_a if score_a > score_b else team_b
            games.append({
                "date": (row.get(date_col or "date") or "").strip(),
                "team_a": team_a,
                "team_b": team_b,
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner,
                "margin": abs(score_a - score_b),
                "location": (row.get(loc_col or "location") or "").strip(),
                "game_type": "regular_season",
            })
    return games


def fetch_season_games(year=2026, from_csv=None):
    """Load or fetch season games. Writes data/season_games_YYYY.json."""
    out_path = os.path.join(DATA_DIR, f"season_games_{year}.json")
    games = []

    if from_csv:
        path = os.path.abspath(from_csv)
        print(f"Loading from CSV: {path}")
        games = load_from_csv(path, year)
    else:
        csv_path = os.path.join(DATA_DIR, f"season_games_{year}.csv")
        if os.path.isfile(csv_path):
            print(f"Loading from {csv_path}")
            games = load_from_csv(csv_path, year)

    if not games:
        if not from_csv:
            print(f"No season games found. To add current-season head-to-head:")
            print(f"  1. Create data/season_games_{year}.csv with columns: date, team_a, team_b, score_a, score_b, location")
            print(f"  2. Run: python scripts/fetch_season_games.py {year}")
        return False

    with open(out_path, "w") as f:
        json.dump(games, f, indent=2)
    print(f"Wrote {len(games)} games to {out_path}")
    return True


def main():
    year = 2026
    from_csv = None
    args = sys.argv[1:]
    if "--from-csv" in args:
        i = args.index("--from-csv")
        if i + 1 < len(args):
            from_csv = args[i + 1]
            args = args[:i] + args[i + 2:]
    for a in args:
        try:
            year = int(a)
            break
        except ValueError:
            pass
    ok = fetch_season_games(year, from_csv=from_csv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
