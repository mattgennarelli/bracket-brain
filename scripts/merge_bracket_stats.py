"""
Merge team stats from teams_merged_YYYY.json into bracket_YYYY.json.
Use when you have a bracket with team names but placeholder stats (e.g. from
fetch_brackets before fetch_torvik/fetch_data was run).

Usage:
  python scripts/merge_bracket_stats.py 2017
  python scripts/merge_bracket_stats.py 2026 data/bracket_2026.json

Output: Overwrites the bracket file in place with stats from teams_merged.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")

sys.path.insert(0, ROOT)
from engine import load_teams_merged, enrich_bracket_with_teams


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("Usage: python scripts/merge_bracket_stats.py YEAR [bracket_path]")
        print("  YEAR: e.g. 2017, 2026")
        print("  bracket_path: default data/bracket_YYYY.json")
        sys.exit(1)

    year = int(args[0])
    bracket_path = args[1] if len(args) > 1 else os.path.join(DATA_DIR, f"bracket_{year}.json")

    if not os.path.isfile(bracket_path):
        print(f"ERROR: Bracket not found: {bracket_path}")
        sys.exit(1)

    merged = load_teams_merged(DATA_DIR, year)
    if not merged:
        print(f"ERROR: No teams_merged_{year}.json or torvik_{year}.json found.")
        print("Run: python scripts/fetch_torvik.py", year)
        print("     python scripts/fetch_data.py", year)
        sys.exit(1)

    with open(bracket_path) as f:
        data = json.load(f)

    bracket = {}
    for rname, teams in data.get("regions", {}).items():
        bracket[rname] = {t["seed"]: t for t in teams}

    n = enrich_bracket_with_teams(bracket, merged)
    print(f"Enriched {n} teams from teams_merged_{year}.json")

    # Write back: convert bracket dict back to regions list format
    out = {"regions": {}, "final_four_matchups": data.get("final_four_matchups", [[0, 1], [2, 3]])}
    if data.get("quadrant_order"):
        out["quadrant_order"] = data["quadrant_order"]
    for rname, by_seed in bracket.items():
        out["regions"][rname] = [by_seed[s] for s in sorted(by_seed.keys())]

    with open(bracket_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {bracket_path}")


if __name__ == "__main__":
    main()
