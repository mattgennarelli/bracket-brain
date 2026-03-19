#!/usr/bin/env python3
"""
Manual injury sync: apply overrides from data/injury_overrides.csv and re-merge.

Usage:
    python scripts/sync_injuries.py          # Apply overrides and re-merge 2026 data
    python scripts/sync_injuries.py --show   # Show current injury status for all tourney teams

CSV format (data/injury_overrides.csv):
    team,player,status,return_round[,source_url,source_type,verified_at,confidence,notes]
    Duke,Patrick Ngongba,out,Sweet 16
    Duke,Caleb Foster,healthy,
    Texas Tech,JT Toppin,out,

Status values: out, doubtful, questionable, probable, day-to-day, healthy
Return round values: Round of 64, Round of 32, Sweet 16, Elite 8, Final Four, Championship
  - If set, injury penalty is zeroed from that round onward
  - Leave blank if unknown
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from engine import _normalize_team_for_match

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OVERRIDES_PATH = os.path.join(DATA_DIR, "injury_overrides.csv")
INJURIES_PATH = os.path.join(DATA_DIR, "injuries_2026.json")
MERGED_PATH = os.path.join(DATA_DIR, "teams_merged_2026.json")

ROUND_ORDER = [
    "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"
]

MASCOT_SUFFIXES = (
    "badgers",
    "blue devils",
    "red raiders",
    "razorbacks",
    "crimson tide",
    "spartans",
    "huskies",
    "wildcats",
    "cardinals",
    "bulldogs",
    "tar heels",
    "wolverines",
    "buckeyes",
    "gaels",
    "red storm",
    "bruins",
)


def load_overrides():
    """Load CSV overrides -> list of dicts."""
    if not os.path.isfile(OVERRIDES_PATH):
        print(f"No overrides file at {OVERRIDES_PATH}")
        return []
    overrides = []
    with open(OVERRIDES_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row.get("team", "").strip()
            player = row.get("player", "").strip()
            status = row.get("status", "").strip().lower()
            return_round = row.get("return_round", "").strip()
            if team and player and status:
                overrides.append({
                    "team": team,
                    "player": player,
                    "status": status,
                    "return_round": return_round if return_round in ROUND_ORDER else "",
                })
    return overrides


def find_matching_team_keys(team_name, injuries):
    """Return all injury keys that normalize to the same team.

    This prevents alias duplicates like "Wisconsin" and "Wisconsin Badgers" from
    reintroducing stale injury rows during the merge step.
    """
    def canonical(name):
        norm = _normalize_team_for_match(name)
        if not norm:
            return ""
        for suffix in MASCOT_SUFFIXES:
            token = f" {suffix}"
            if norm.endswith(token):
                norm = norm[: -len(token)].strip()
                break
        return norm

    target = canonical(team_name)
    if not target:
        return []
    return [key for key in injuries if canonical(key) == target]


def apply_overrides(overrides):
    """Apply overrides to injuries_2026.json."""
    with open(INJURIES_PATH) as f:
        injuries = json.load(f)

    changes = 0
    for ov in overrides:
        team = ov["team"]
        team_keys = find_matching_team_keys(team, injuries)
        if not team_keys:
            injuries[team] = []
            team_keys = [team]

        player_lower = ov["player"].lower()
        for team_key in team_keys:
            if isinstance(injuries[team_key], dict):
                inj_list = injuries[team_key].get("injuries", [])
            else:
                inj_list = injuries[team_key]

            found = False
            for inj in list(inj_list):
                pname = inj.get("player", "").lower()
                stripped = pname
                for suffix in (" jr.", " jr", " sr.", " sr"):
                    if stripped.endswith(suffix):
                        stripped = stripped[:-len(suffix)].strip()
                if pname == player_lower or stripped == player_lower or player_lower == stripped:
                    old_status = inj.get("status", "unknown")
                    if ov["status"] == "healthy":
                        inj_list.remove(inj)
                        print(f"  ✓ {team_key} / {ov['player']}: {old_status} → REMOVED (healthy)")
                    else:
                        inj["status"] = ov["status"]
                        if ov["return_round"]:
                            inj["return_round"] = ov["return_round"]
                        elif "return_round" in inj:
                            del inj["return_round"]
                        print(f"  ✓ {team_key} / {ov['player']}: {old_status} → {ov['status']}"
                              f"{' (back by ' + ov['return_round'] + ')' if ov['return_round'] else ''}")
                    found = True
                    changes += 1
                    break

            if not found and ov["status"] != "healthy":
                new_entry = {"player": ov["player"], "status": ov["status"]}
                if ov["return_round"]:
                    new_entry["return_round"] = ov["return_round"]
                inj_list.append(new_entry)
                print(f"  + {team_key} / {ov['player']}: ADDED as {ov['status']}"
                      f"{' (back by ' + ov['return_round'] + ')' if ov['return_round'] else ''}")
                changes += 1

            if isinstance(injuries[team_key], dict):
                injuries[team_key]["injuries"] = inj_list
            else:
                injuries[team_key] = inj_list

    with open(INJURIES_PATH, "w") as f:
        json.dump(injuries, f, indent=2)

    return changes


def show_injuries():
    """Show all injuries for tournament teams."""
    with open(MERGED_PATH) as f:
        teams = json.load(f)

    # Get bracket teams
    bracket_path = os.path.join(DATA_DIR, "bracket_2026.json")
    bracket_teams = set()
    if os.path.isfile(bracket_path):
        with open(bracket_path) as f:
            bracket = json.load(f)
        regions = bracket.get("regions", {})
        if isinstance(regions, dict):
            for region_name, team_list in regions.items():
                for team in (team_list if isinstance(team_list, list) else []):
                    bracket_teams.add(team.get("team", "") or team.get("name", ""))
        elif isinstance(regions, list):
            for region in regions:
                for team in region.get("teams", []):
                    bracket_teams.add(team.get("team", "") or team.get("name", ""))

    for t in sorted(teams, key=lambda x: x.get("team", "")):
        name = t.get("team", "")
        if bracket_teams and name not in bracket_teams:
            continue
        injuries = t.get("injuries", [])
        if not injuries:
            continue
        impact = t.get("injury_impact", 0)
        print(f"\n{name} (−{impact:.1f} pts):")
        for inj in injuries:
            ret = inj.get("return_round", "")
            ret_str = f" → back by {ret}" if ret else ""
            bpr = inj.get("bpr", 0)
            share = inj.get("bpr_share", 0)
            print(f"  {inj['player']:25s} {inj.get('status','?'):15s} "
                  f"BPR={bpr:+.1f}  share={share:.3f}{ret_str}")


def main():
    if "--show" in sys.argv:
        show_injuries()
        return

    print("Loading injury overrides...")
    overrides = load_overrides()
    if not overrides:
        print("No overrides to apply.")
        return

    print(f"Applying {len(overrides)} overrides to {INJURIES_PATH}...")
    changes = apply_overrides(overrides)
    print(f"\n{changes} changes applied.")

    # Re-merge
    print("\nRe-merging 2026 data...")
    from scripts.fetch_data import build_merged_teams
    build_merged_teams(2026, skip_torvik_fetch=True)

    print("\nDone! Run 'python run.py' to regenerate bracket with updated injuries.")
    print("Or push to deploy: git add data/ && git commit -m 'Update injuries' && git push")


if __name__ == "__main__":
    main()
