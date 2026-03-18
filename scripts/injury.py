#!/usr/bin/env python3
"""
Quick CLI for managing injury entries.

Usage:
  python scripts/injury.py add "Duke" "Patrick Ngongba" out
  python scripts/injury.py add "UCLA" "Donovan Dent" questionable
  python scripts/injury.py remove "UCLA" "Donovan Dent"
  python scripts/injury.py list                          # show all injuries
  python scripts/injury.py list Duke                     # show Duke injuries
  python scripts/injury.py news                          # scrape ESPN news for injuries
  python scripts/injury.py news --update                 # scrape + auto-update file

BPR and poss are auto-populated from EvanMiya data.
"""
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)


def _load_injuries(year):
    path = os.path.join(DATA_DIR, f"injuries_{year}.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_injuries(data, year):
    path = os.path.join(DATA_DIR, f"injuries_{year}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path}")


def _get_bpr(team, player, year):
    """Look up BPR data for a player."""
    from scripts.fetch_injuries_live import _load_bpr_data, _find_bpr_team, FALLBACK_BPR_SHARE
    bpr_data = _load_bpr_data(year)
    team_bpr = _find_bpr_team(team, bpr_data)

    player_match = team_bpr.get(player.lower())
    if player_match is None:
        last = player.lower().split()[-1] if player else ""
        for k, v in team_bpr.items():
            if last and last in k:
                player_match = v
                break
    if player_match is None:
        parts = player.lower().split()
        if len(parts) >= 2:
            for k, v in team_bpr.items():
                if parts[0] in k and parts[-1] in k:
                    player_match = v
                    break

    if player_match:
        return {
            "bpr": round(float(player_match["bpr"]), 4),
            "poss": round(float(player_match.get("poss", 0)), 0),
            "bpr_share": round(float(player_match.get("bpr_share", 0)), 4),
        }
    return {
        "bpr": round(FALLBACK_BPR_SHARE * 10, 4),
        "poss": 0.0,
        "bpr_share": FALLBACK_BPR_SHARE,
    }


def cmd_add(team, player, status, year=2026):
    data = _load_injuries(year)
    if team not in data:
        data[team] = {"injuries": [], "roster": []}
    elif isinstance(data[team], list):
        data[team] = {"injuries": data[team], "roster": []}

    injuries = data[team].get("injuries", [])
    # Remove existing
    injuries = [i for i in injuries if i.get("player", "").lower() != player.lower()]

    bpr_info = _get_bpr(team, player, year)
    injuries.append({
        "player": player,
        "status": status.lower(),
        "bpr": bpr_info["bpr"],
        "poss": bpr_info["poss"],
        "bpr_share": bpr_info["bpr_share"],
        "importance": bpr_info["bpr_share"],
        "source": "manual",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    data[team]["injuries"] = injuries
    _save_injuries(data, year)
    print(f"  ✓ {team} / {player} → {status} (BPR={bpr_info['bpr']}, poss={bpr_info['poss']})")


def cmd_remove(team, player, year=2026):
    data = _load_injuries(year)
    if team not in data:
        print(f"  No injury data for {team}")
        return
    entry = data[team] if isinstance(data[team], dict) else {"injuries": data[team], "roster": []}
    before = len(entry.get("injuries", []))
    entry["injuries"] = [i for i in entry.get("injuries", []) if i.get("player", "").lower() != player.lower()]
    after = len(entry["injuries"])
    data[team] = entry
    _save_injuries(data, year)
    if before > after:
        print(f"  ✓ Removed {player} from {team}")
    else:
        print(f"  {player} not found in {team} injuries")


def cmd_list(team_filter=None, year=2026):
    data = _load_injuries(year)
    found = False
    for team in sorted(data.keys()):
        if team_filter and team_filter.lower() not in team.lower():
            continue
        v = data[team]
        injuries = v.get("injuries", []) if isinstance(v, dict) else v
        if not injuries:
            continue
        found = True
        print(f"\n  {team}:")
        for i in injuries:
            player = i.get("player", "?")
            status = i.get("status", "?")
            bpr = i.get("bpr", 0)
            poss = i.get("poss", 0)
            src = i.get("source", "?")
            flag = " ⚠️ " if bpr == 1.0 and poss == 0 else "  "
            print(f"  {flag}{player:25s} {status:15s} BPR={bpr:6.2f}  poss={poss:5.0f}  [{src}]")
    if not found:
        print("  No injuries found." + (f" (filter: {team_filter})" if team_filter else ""))


def cmd_news(year=2026):
    """Scrape ESPN news for injury info and merge into injuries file."""
    from scripts.fetch_injuries_live import fetch_injuries_from_news
    fetch_injuries_from_news(year)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manage injury entries")
    parser.add_argument("--year", type=int, default=2026)
    sub = parser.add_subparsers(dest="cmd")

    p_add = sub.add_parser("add", help="Add/update injury")
    p_add.add_argument("team")
    p_add.add_argument("player")
    p_add.add_argument("status", choices=["out", "doubtful", "questionable", "probable", "day-to-day"])

    p_rm = sub.add_parser("remove", help="Remove injury")
    p_rm.add_argument("team")
    p_rm.add_argument("player")

    p_list = sub.add_parser("list", help="List injuries")
    p_list.add_argument("team", nargs="?", default=None)

    p_news = sub.add_parser("news", help="Scrape ESPN news for injuries")

    args = parser.parse_args()

    if args.cmd == "add":
        cmd_add(args.team, args.player, args.status, args.year)
    elif args.cmd == "remove":
        cmd_remove(args.team, args.player, args.year)
    elif args.cmd == "list":
        cmd_list(args.team, args.year)
    elif args.cmd == "news":
        cmd_news(args.year)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
