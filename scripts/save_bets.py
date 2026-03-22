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
    commence = pick.get("commence_time") or pick.get("date", "")
    return f"{commence}|{pick['home_team']}|{pick['away_team']}|{pick['bet_type']}|{side}"


def _market_id(pick):
    game_date = _game_date_et(pick.get("commence_time", ""), pick.get("date", ""))
    return f"{game_date}|{pick.get('home_team','')}|{pick.get('away_team','')}|{pick.get('bet_type','')}"


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


def _pick_has_started(pick) -> bool:
    commence_time = pick.get("commence_time")
    if not commence_time:
        return False
    try:
        dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
    except ValueError:
        return False
    return dt <= datetime.now(timezone.utc)


def _pick_is_locked(pick) -> bool:
    return (
        pick.get("result") in {"W", "L", "P"}
        or (pick.get("actual_score_home") is not None and pick.get("actual_score_away") is not None)
    )


def _pick_generated_before_start(pick) -> bool:
    generated_at = pick.get("generated_at")
    commence_time = pick.get("commence_time")
    if not generated_at or not commence_time:
        return False
    try:
        generated_dt = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
        commence_dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
    except ValueError:
        return False
    return generated_dt <= commence_dt


def _official_pick_preference_key(pick):
    return (
        1 if _pick_generated_before_start(pick) else 0,
        1 if _pick_is_locked(pick) else 0,
        1 if _pick_has_started(pick) else 0,
        str(pick.get("generated_at") or ""),
        str(pick.get("settled_at") or ""),
        str(pick.get("commence_time") or ""),
        _game_date_et(pick.get("commence_time", ""), pick.get("date", "")),
    )


def _merge_market_picks(existing_picks, new_picks):
    merged = {}
    for pick in existing_picks:
        key = _market_id(pick)
        current = merged.get(key)
        if current is None or _official_pick_preference_key(pick) > _official_pick_preference_key(current):
            merged[key] = pick

    for pick in new_picks:
        key = _market_id(pick)
        current = merged.get(key)
        if current is not None and (_pick_is_locked(current) or _pick_has_started(current)):
            continue
        merged[key] = pick

    return list(merged.values())


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

    # Replace superseded same-market picks; preserve started/settled markets.
    ledger = load_ledger()
    before_count = len(ledger["picks"])
    ledger["picks"] = _merge_market_picks(ledger["picks"], bets)
    save_ledger(ledger)
    delta = len(ledger["picks"]) - before_count
    if delta >= 0:
        print(f"  Added {delta} new pick(s) to ledger ({len(ledger['picks'])} total)")
    else:
        print(f"  Replaced {-delta} superseded pick(s) in ledger ({len(ledger['picks'])} total)")

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
