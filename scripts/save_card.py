"""
save_card.py — Save today's full game card (all matchups, 1-3 picks each).

Run once each morning before games start:
  python scripts/save_card.py

Uses ODDS_PROVIDER (odds_api or betstack). Set ODDS_API_KEY or BETSTACK_API_KEY.

Usage:
  export ODDS_API_KEY=your_key
  python scripts/save_card.py

  export ODDS_PROVIDER=betstack
  export BETSTACK_API_KEY=your_key
  python scripts/save_card.py --year 2026
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

from best_bets import get_full_card_json
from odds_provider import get_api_key
from settle_bets import compute_stats  # reuse stats computation

CARD_LEDGER_PATH = os.path.join(DATA_DIR, "card_ledger.json")
ET_TZ = ZoneInfo("America/New_York")


def _pick_id(pick):
    """Stable dedup key for a single card pick."""
    side = pick.get("bet_side") or pick.get("bet_team", "")
    commence = pick.get("commence_time") or pick.get("date", "")
    return f"{commence}|{pick['home_team']}|{pick['away_team']}|{pick['bet_type']}|{side}"


def _market_id(pick):
    game_date = _game_date_et(pick.get("commence_time", ""), pick.get("date", ""))
    return f"{game_date}|{pick.get('home_team','')}|{pick.get('away_team','')}|{pick.get('bet_type','')}"


def load_ledger():
    if os.path.isfile(CARD_LEDGER_PATH):
        with open(CARD_LEDGER_PATH) as f:
            return json.load(f)
    return {"picks": []}


def save_ledger(ledger):
    with open(CARD_LEDGER_PATH, "w") as f:
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
    parser = argparse.ArgumentParser(description="Save today's full game card to disk")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: ODDS_API_KEY or BETSTACK_API_KEY per ODDS_PROVIDER)")
    parser.add_argument("--year", type=int, default=datetime.now().year)
    args = parser.parse_args()

    api_key = args.api_key or get_api_key()
    if not api_key:
        provider = (os.environ.get("ODDS_PROVIDER") or "odds_api").strip().lower()
        print("ERROR: No API key. Set ODDS_API_KEY or BETSTACK_API_KEY (per ODDS_PROVIDER), or pass --api-key.")
        sys.exit(1)

    today = datetime.now(ET_TZ).strftime("%Y-%m-%d")
    print(f"Fetching full card for {today}...")

    games = get_full_card_json(api_key, year=args.year)

    if not games:
        print("No games today (or no odds available).")
        return

    # Flatten games → picks; attach date + pending result
    all_picks = []
    for game in games:
        for pick in game.get("picks", []):
            pick["home_team"] = game["home_team"]
            pick["away_team"] = game["away_team"]
            pick["commence_time"] = game.get("commence_time", "")
            pick["date"] = _game_date_et(pick["commence_time"], today)
            pick["result"] = None
            pick["actual_score_home"] = None
            pick["actual_score_away"] = None
            pick["settled_at"] = None
            all_picks.append(pick)

    print(f"  {len(games)} games, {len(all_picks)} picks total")

    # Save daily snapshot
    daily_path = os.path.join(DATA_DIR, f"card_{today}.json")
    with open(daily_path, "w") as f:
        json.dump({
            "date": today,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "games": games,
        }, f, indent=2)
    print(f"  Saved daily snapshot → {os.path.basename(daily_path)}")

    # Replace superseded same-market picks; preserve started/settled markets.
    ledger = load_ledger()
    before_count = len(ledger["picks"])
    ledger["picks"] = _merge_market_picks(ledger["picks"], all_picks)
    ledger["stats"] = compute_stats(ledger["picks"])
    save_ledger(ledger)
    delta = len(ledger["picks"]) - before_count
    if delta >= 0:
        print(f"  Added {delta} new pick(s) to card ledger ({len(ledger['picks'])} total)")
    else:
        print(f"  Replaced {-delta} superseded pick(s) in card ledger ({len(ledger['picks'])} total)")

    # Summary by game
    for game in games:
        data_flag = "" if game.get("data_available", True) else " [no data]"
        picks_str = ", ".join(
            f"{p['bet_type'].upper()}: {p.get('bet_side') or p.get('bet_team')} {p.get('stars','')}"
            for p in game.get("picks", [])
        )
        print(f"  {game['away_team']} @ {game['home_team']}{data_flag}  →  {picks_str or 'no picks'}")

    print(f"\nDone. Run settle_card.py after games finish.")


if __name__ == "__main__":
    main()
