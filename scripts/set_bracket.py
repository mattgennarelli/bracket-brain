"""
Create or edit the tournament bracket file (data/bracket_YYYY.json).
  --template: Write an empty template with placeholder slots (TEAM_1 .. TEAM_16 per region).
  --interactive: Prompt for region, seed, team name and optionally fill stats from merged data.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
BRACKET_PATH = os.path.join(DATA_DIR, "bracket_2026.json")
REGIONS = ["South", "East", "Midwest", "West"]
SEEDS = list(range(1, 17))


def normalize_team(name):
    if not name or not isinstance(name, str):
        return ""
    return " ".join(name.strip().split())


def load_merged_teams(year=2026):
    path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
    if not os.path.isfile(path):
        path = os.path.join(DATA_DIR, f"torvik_{year}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {normalize_team(r["team"]): r for r in data if r.get("team") and not str(r.get("team", "")).startswith("Unknown")}
    return {normalize_team(k): v for k, v in data.items() if isinstance(v, dict) and ("adj_o" in v or "team" in v) and not str(v.get("team", "")).startswith("Unknown")}


def template_bracket(year=2026):
    """Return bracket structure with placeholder teams."""
    placeholder = lambda s, r: {
        "team": f"TEAM_{s}",
        "seed": s,
        "adj_o": 100,
        "adj_d": 100,
        "adj_tempo": 67.5,
        "barthag": 0.5,
    }
    regions = {}
    for region in REGIONS:
        regions[region] = [placeholder(seed, region) for seed in SEEDS]
    return {
        "regions": regions,
        "final_four_matchups": [[0, 1], [2, 3]],
    }


def write_template(path=None, year=2026):
    path = path or os.path.join(DATA_DIR, f"bracket_{year}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = template_bracket(year)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote template to {path}")
    print("Edit the file to replace TEAM_* with actual team names and stats, or run with --interactive.")


def lookup_team(merged, team_name):
    """Return a team dict with stats from merged data, or None."""
    key = normalize_team(team_name)
    if key in merged:
        r = dict(merged[key])
        r["team"] = r.get("team") or team_name
        r.setdefault("seed", 8)
        return r
    # Fuzzy: first team that contains the query
    key_lower = key.lower()
    for k, v in merged.items():
        if key_lower in k.lower() or k.lower() in key_lower:
            r = dict(v)
            r["team"] = team_name.strip() or k
            r.setdefault("seed", 8)
            return r
    return None


def interactive_edit(path=None, year=2026):
    path = path or os.path.join(DATA_DIR, f"bracket_{year}.json")
    if not os.path.isfile(path):
        write_template(path, year)
    with open(path) as f:
        data = json.load(f)
    merged = load_merged_teams(year)
    if merged:
        print(f"Loaded {len(merged)} teams for auto-fill (from teams_merged or torvik).")
    else:
        print("No merged/torvik data found. You will enter stats manually.")

    regions = data.setdefault("regions", {})
    for r in REGIONS:
        if r not in regions:
            regions[r] = [{"team": f"TEAM_{s}", "seed": s, "adj_o": 100, "adj_d": 100, "adj_tempo": 67.5, "barthag": 0.5} for s in SEEDS]

    while True:
        region = input("\nRegion (South/East/Midwest/West) or Enter to finish: ").strip()
        if not region:
            break
        if region not in regions:
            print("Invalid region.")
            continue
        try:
            seed = int(input("Seed (1-16): ").strip())
        except ValueError:
            print("Invalid seed.")
            continue
        if seed not in SEEDS:
            print("Seed must be 1-16.")
            continue
        team_name = input("Team name: ").strip()
        if not team_name:
            print("Skipped.")
            continue
        # Find existing slot by seed in this region
        team_list = regions[region]
        slot = next((i for i, t in enumerate(team_list) if t.get("seed") == seed), None)
        if slot is None:
            team_list.append({"team": team_name, "seed": seed, "adj_o": 100, "adj_d": 100, "adj_tempo": 67.5, "barthag": 0.5})
            slot = len(team_list) - 1
        row = team_list[slot]
        row["team"] = team_name
        filled = lookup_team(merged, team_name)
        if filled:
            row["adj_o"] = filled.get("adj_o", 100)
            row["adj_d"] = filled.get("adj_d", 100)
            row["adj_tempo"] = filled.get("adj_tempo", 67.5)
            row["barthag"] = filled.get("barthag", 0.5)
            if "luck" in filled:
                row["luck"] = filled["luck"]
            if "star_score" in filled:
                row["star_score"] = filled["star_score"]
            print(f"  Auto-filled stats: adj_o={row['adj_o']}, adj_d={row['adj_d']}")
        else:
            try:
                row["adj_o"] = float(input("  adj_o (or Enter for 100): ").strip() or 100)
                row["adj_d"] = float(input("  adj_d (or Enter for 100): ").strip() or 100)
                row["adj_tempo"] = float(input("  adj_tempo (or Enter for 67.5): ").strip() or 67.5)
                row["barthag"] = float(input("  barthag (or Enter for 0.5): ").strip() or 0.5)
            except ValueError:
                pass
        print(f"  Set {region} seed {seed} = {team_name}")

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")


def main():
    year = 2026
    argv = sys.argv[1:]
    if argv and argv[0].isdigit():
        year = int(argv[0])
        argv = argv[1:]
    if "--template" in argv:
        write_template(os.path.join(DATA_DIR, f"bracket_{year}.json"), year)
        return
    if "--interactive" in argv:
        interactive_edit(os.path.join(DATA_DIR, f"bracket_{year}.json"), year)
        return
    # Default: write template
    print("Usage: python scripts/set_bracket.py [year] --template | --interactive")
    write_template(os.path.join(DATA_DIR, f"bracket_{year}.json"), year)


if __name__ == "__main__":
    main()
