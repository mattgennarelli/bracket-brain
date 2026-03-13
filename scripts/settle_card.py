"""
settle_card.py — Settle pending card picks with actual game scores.

Run after each day's games finish (or via GitHub Actions):
  python scripts/settle_card.py

Mirrors settle_bets.py but operates on data/card_ledger.json instead
of data/bets_ledger.json.

Usage:
  python scripts/settle_card.py
  python scripts/settle_card.py --days 3
  python scripts/settle_card.py --debug
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT)

# Reuse all settle logic from settle_bets
from settle_bets import (
    fetch_scores_espn,
    match_score,
    get_scores_from_record,
    settle_pick,
    compute_stats,
)

CARD_LEDGER_PATH = os.path.join(DATA_DIR, "card_ledger.json")


def main():
    parser = argparse.ArgumentParser(description="Settle pending card picks with actual scores")
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY", ""))
    parser.add_argument("--days", type=int, default=3, help="Days back to fetch scores (default 3)")
    parser.add_argument("--debug", action="store_true", help="Log unmatched picks")
    args = parser.parse_args()

    if not os.path.isfile(CARD_LEDGER_PATH):
        print("No card ledger found. Run save_card.py first.")
        return

    with open(CARD_LEDGER_PATH) as f:
        ledger = json.load(f)

    pending = [p for p in ledger["picks"] if p.get("result") is None]
    if not pending:
        print("No pending card picks to settle.")
        return

    print(f"Settling {len(pending)} pending card pick(s)...")

    # Fetch scores via ESPN (free)
    scores_by_key = fetch_scores_espn(pending, days=args.days)
    print(f"  Loaded {len(scores_by_key)} completed game score(s) from ESPN.")

    settled = 0
    for pick in ledger["picks"]:
        if pick.get("result") is not None:
            continue
        score_rec = match_score(pick, scores_by_key)
        if score_rec is None:
            if args.debug:
                print(f"  [unmatched] {pick['away_team']} @ {pick['home_team']}")
            continue
        home_score, away_score = get_scores_from_record(score_rec)
        if home_score is None or away_score is None:
            continue
        result = settle_pick(pick, home_score, away_score)
        if result is None:
            continue
        pick["result"] = result
        pick["actual_score_home"] = home_score
        pick["actual_score_away"] = away_score
        pick["settled_at"] = datetime.now(timezone.utc).isoformat()
        settled += 1
        sign = "✓" if result == "W" else ("=" if result == "P" else "✗")
        bt = pick["bet_type"].upper()
        side = pick.get("bet_side") or pick.get("bet_team", "")
        print(f"  {sign} [{bt}] {side} — {result}  ({pick['home_team']} {int(home_score)}-{int(away_score)} {pick['away_team']})")

    if settled:
        ledger["stats"] = compute_stats(ledger["picks"])
        with open(CARD_LEDGER_PATH, "w") as f:
            json.dump(ledger, f, indent=2)
        stats = ledger["stats"]
        print(f"\nSettled {settled} pick(s). Card record: {stats.get('wins',0)}W-{stats.get('losses',0)}L "
              f"({stats.get('hit_rate',0)*100:.0f}% hit rate)")
    else:
        print("No new picks settled (games may still be in progress).")


if __name__ == "__main__":
    main()
