"""
Compute momentum (recent form) from season game-by-game results.

Uses simplified win-rate differential: momentum = (win_pct_last_N - win_pct_season) * 2
clamped to [-1, 1]. Positive = hot streak, negative = cold.

Input:
  - data/season_games_YYYY.json
  - data/teams_merged_YYYY.json (for season win_pct)

Output:
  - data/momentum_YYYY.json: {team: {momentum, adj_o_recent, adj_d_recent, games_used}}

Usage:
  python scripts/compute_momentum.py 2026
  python scripts/compute_momentum.py --games 10 2026
"""
import json
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")

sys.path.insert(0, ROOT)
from engine import _normalize_team_for_match


def _normalize(name):
    if not name or not isinstance(name, str):
        return ""
    return " ".join(name.strip().split())


def load_season_games(year):
    """Load season games. Returns list of game dicts sorted by date."""
    path = os.path.join(DATA_DIR, f"season_games_{year}.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        data = json.load(f)
    games = data if isinstance(data, list) else data.get("games", [])
    games = [g for g in games if g.get("team_a") and g.get("team_b")]
    games.sort(key=lambda x: x.get("date", ""))
    return games


def load_teams_merged(year):
    """Load teams_merged. Returns dict keyed by normalized name."""
    path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for r in data:
        team = r.get("team", "")
        if not team:
            continue
        key = _normalize_team_for_match(team)
        if key:
            out[key] = dict(r)
    return out


def compute_momentum(year, last_n=10):
    """Compute momentum for each team. Returns dict team -> {momentum, games_used}."""
    games = load_season_games(year)
    teams = load_teams_merged(year)
    if not games:
        return {}

    # Build team -> list of (date, won) for each game
    team_games = defaultdict(list)
    for g in games:
        date = g.get("date", "")
        t_a = _normalize_team_for_match(g.get("team_a", ""))
        t_b = _normalize_team_for_match(g.get("team_b", ""))
        score_a = g.get("score_a", 0) or 0
        score_b = g.get("score_b", 0) or 0
        if not t_a or not t_b:
            continue
        team_games[t_a].append((date, score_a > score_b))
        team_games[t_b].append((date, score_b > score_a))

    result = {}
    for team_key, game_list in team_games.items():
        game_list.sort(key=lambda x: x[0])
        last = game_list[-last_n:] if len(game_list) >= last_n else game_list
        if not last:
            continue
        wins_last = sum(1 for _, w in last if w)
        win_pct_last = wins_last / len(last)

        # Season win_pct from teams_merged
        team_row = teams.get(team_key)
        if not team_row:
            continue
        wins = team_row.get("wins")
        games_count = team_row.get("games")
        win_pct_season = team_row.get("win_pct")
        if win_pct_season is None and wins is not None and games_count and games_count > 0:
            win_pct_season = wins / games_count
        if win_pct_season is None:
            win_pct_season = 0.5

        momentum = (win_pct_last - win_pct_season) * 2
        momentum = max(-1.0, min(1.0, momentum))

        # Resolve display name from teams_merged
        display_name = team_row.get("team", team_key)
        result[display_name] = {
            "momentum": round(momentum, 4),
            "games_used": len(last),
            "win_pct_last": round(win_pct_last, 4),
            "win_pct_season": round(win_pct_season, 4),
        }
    return result


def main():
    year = 2026
    last_n = 10
    args = sys.argv[1:]
    if "--games" in args:
        i = args.index("--games")
        if i + 1 < len(args):
            try:
                last_n = int(args[i + 1])
                args = args[:i] + args[i + 2:]
            except ValueError:
                args = args[:i] + args[i + 1:]
    for a in args:
        if a.startswith("-"):
            continue
        try:
            year = int(a)
            break
        except ValueError:
            pass

    print(f"Computing momentum for {year} (last {last_n} games)...")
    momentum_data = compute_momentum(year, last_n=last_n)
    if not momentum_data:
        print("No momentum data (missing season_games or teams_merged). Writing empty file.")
        momentum_data = {}

    out_path = os.path.join(DATA_DIR, f"momentum_{year}.json")
    with open(out_path, "w") as f:
        json.dump(momentum_data, f, indent=2)
    print(f"Wrote {len(momentum_data)} teams -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
