"""
save_bets.py — Fetch today's best bets and save them to disk.

Run once each morning before games start:
  python scripts/save_bets.py

Saves:
  data/bets_YYYY-MM-DD.json   — today's picks (snapshot)
  data/bets_ledger.json       — running history; picks start as pending, get
                                 settled by settle_bets.py after games finish

Usage:
  export ODDS_API_KEY=your_key
  python scripts/save_bets.py
  python scripts/save_bets.py --year 2026
  python scripts/save_bets.py --ml-edge 0.05 --spread-edge 4.0
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, SCRIPT_DIR)

from best_bets import get_best_bets_json, DEFAULT_ML_EDGE, DEFAULT_SPREAD_EDGE, DEFAULT_TOTAL_EDGE
from engine import is_ncaa_tournament_game

LEDGER_PATH = os.path.join(DATA_DIR, "bets_ledger.json")
ET_TZ = ZoneInfo("America/New_York")


def _pick_id(pick):
    """Stable dedup key for a single pick."""
    side = pick.get("bet_side") or pick.get("bet_team", "")
    return f"{pick['date']}|{pick['home_team']}|{pick['away_team']}|{pick['bet_type']}|{side}"


def load_ledger():
    if os.path.isfile(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {"picks": []}


def save_ledger(ledger):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def _game_date_et(commence_time: str, fallback: str) -> str:
    if commence_time:
        try:
            dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
            return dt.astimezone(ET_TZ).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return fallback


def main():
    parser = argparse.ArgumentParser(description="Save today's best bets to disk")
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY", ""))
    parser.add_argument("--year", type=int, default=datetime.now().year)
    parser.add_argument("--ml-edge", type=float, default=DEFAULT_ML_EDGE)
    parser.add_argument("--spread-edge", type=float, default=DEFAULT_SPREAD_EDGE)
    parser.add_argument("--total-edge", type=float, default=DEFAULT_TOTAL_EDGE)
    parser.add_argument("--no-totals", action="store_true",
                        help="Skip total bets (use during conference tournaments; model is calibrated for NCAA tourney only)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key. Set ODDS_API_KEY or pass --api-key.")
        sys.exit(1)

    today = datetime.now(ET_TZ).strftime("%Y-%m-%d")
    print(f"Fetching picks for {today}...")

    total_min = 9999 if args.no_totals else args.total_edge
    if args.no_totals:
        print("  Totals disabled (--no-totals): model is calibrated for NCAA tournament only.")
    bets = get_best_bets_json(
        args.api_key,
        year=args.year,
        ml_min=args.ml_edge,
        spread_min=args.spread_edge,
        total_min=total_min,
    )

    if not bets:
        print("No qualifying picks today.")
        # Still save an empty file so the API knows we ran
        daily_path = os.path.join(DATA_DIR, f"bets_{today}.json")
        with open(daily_path, "w") as f:
            json.dump({"date": today, "picks": [], "fetched_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
        return

    if not bets:
        print("No qualifying picks today.")
        daily_path = os.path.join(DATA_DIR, f"bets_{today}.json")
        with open(daily_path, "w") as f:
            json.dump({"date": today, "picks": [], "fetched_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
        return

    # Annotate each pick with date + pending result
    now_iso = datetime.now(timezone.utc).isoformat()
    for pick in bets:
        pick["date"] = _game_date_et(pick.get("commence_time", ""), today)
        pick["generated_at"] = now_iso   # when this pick was saved (for model epoch filtering)
        pick["result"] = None          # null = pending
        pick["actual_score_home"] = None
        pick["actual_score_away"] = None
        pick["settled_at"] = None
        pick["ncaa_tournament"] = is_ncaa_tournament_game(
            pick.get("home_team", ""), pick.get("away_team", ""), year=args.year
        )

    # Save daily snapshot
    daily_path = os.path.join(DATA_DIR, f"bets_{today}.json")
    with open(daily_path, "w") as f:
        json.dump({
            "date": today,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "picks": bets,
        }, f, indent=2)
    print(f"  Saved {len(bets)} pick(s) to {os.path.basename(daily_path)}")

    # Append new picks to ledger (skip duplicates)
    ledger = load_ledger()
    existing_ids = {_pick_id(p) for p in ledger["picks"]}
    added = 0
    for pick in bets:
        pid = _pick_id(pick)
        if pid not in existing_ids:
            ledger["picks"].append(pick)
            existing_ids.add(pid)
            added += 1

    save_ledger(ledger)
    print(f"  Added {added} new pick(s) to ledger ({len(ledger['picks'])} total)")

    # Print summary
    print(f"\n  Today's picks:")
    for p in bets:
        bt = p["bet_type"].upper()
        side = p.get("bet_side") or p.get("bet_team", "?")
        odds = p.get("bet_odds")
        odds_str = f"{'+' if odds and odds > 0 else ''}{int(odds)}" if odds else "N/A"
        stars = p.get("stars", "")
        edge = p.get("edge", 0)
        print(f"    {stars:4s} [{bt}] {side} ({odds_str})  edge: {edge:+.1%}")

    print(f"\nDone. Run settle_bets.py after games finish to record results.")


if __name__ == "__main__":
    main()
