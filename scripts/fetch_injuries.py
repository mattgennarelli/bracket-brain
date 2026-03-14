"""
Fetch or load injury data for tournament teams.

Supports manual CSV: data/injuries_YYYY.csv with columns:
  team, player, status, bpr_share (or importance)

Status values: out, doubtful, questionable, probable
bpr_share: 0-1, player's share of team BPR (from EvanMiya). If missing, importance is used.
importance: 0-1, manual importance weight when bpr_share not available.

Output: data/injuries_YYYY.json
  {"Duke": [{"player": "Cameron Boozer", "status": "out", "bpr_share": 0.32, "importance": 1.0}]}

Usage:
  python scripts/fetch_injuries.py 2026
  python scripts/fetch_injuries.py --from-csv data/injuries_2026.csv 2026
"""
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from engine import _normalize_team_for_match


def _normalize(name):
    if not name or not isinstance(name, str):
        return ""
    return " ".join(name.strip().split())


def _find_col(headers, candidates):
    """Find first matching column (case-insensitive)."""
    lower = {str(h).strip().lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _parse_float(val, default=None):
    if val is None or val == "":
        return default
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return default


def load_from_csv(path, year):
    """Load injuries from CSV. Returns dict team -> list of injury dicts."""
    if not os.path.isfile(path):
        return {}
    injuries_by_team = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        team_col = _find_col(headers, ["team", "Team", "school", "School"])
        player_col = _find_col(headers, ["player", "Player", "name", "Name"])
        status_col = _find_col(headers, ["status", "Status", "injury_status"])
        bpr_col = _find_col(headers, ["bpr_share", "BPR Share", "bpr"])
        importance_col = _find_col(headers, ["importance", "Importance", "weight"])

        for row in reader:
            team = (row.get(team_col or "team") or "").strip()
            player = (row.get(player_col or "player") or "").strip()
            status = (row.get(status_col or "status") or "out").strip().lower()
            if not team or not player:
                continue
            bpr_share = _parse_float(row.get(bpr_col or "bpr_share"))
            importance = _parse_float(row.get(importance_col or "importance"), 0.5)
            if bpr_share is None:
                bpr_share = importance
            elif importance is None:
                importance = bpr_share
            injuries_by_team.setdefault(team, []).append({
                "player": player,
                "status": status if status in ("out", "doubtful", "questionable", "probable") else "out",
                "bpr_share": round(bpr_share, 4),
                "importance": round(importance, 4),
            })
    return injuries_by_team


def fetch_injuries(year=2026, from_csv=None):
    """Load or fetch injuries. Writes data/injuries_YYYY.json."""
    out_path = os.path.join(DATA_DIR, f"injuries_{year}.json")
    injuries = {}

    if from_csv:
        path = os.path.abspath(from_csv)
        print(f"Loading from CSV: {path}")
        injuries = load_from_csv(path, year)
    else:
        csv_path = os.path.join(DATA_DIR, f"injuries_{year}.csv")
        if os.path.isfile(csv_path):
            print(f"Loading from {csv_path}")
            injuries = load_from_csv(csv_path, year)

    if not injuries:
        if not from_csv:
            print(f"No injury data found. To add injuries before Selection Sunday:")
            print(f"  1. Create data/injuries_{year}.csv with columns: team, player, status, bpr_share (or importance)")
            print(f"  2. Run: python scripts/fetch_injuries.py {year}")
        # Write empty file so merge doesn't fail
        injuries = {}

    with open(out_path, "w") as f:
        json.dump(injuries, f, indent=2)
    print(f"Wrote {len(injuries)} teams with injuries -> {out_path}")
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
        if a.startswith("-"):
            continue
        try:
            year = int(a)
            break
        except ValueError:
            pass
    ok = fetch_injuries(year, from_csv=from_csv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
