"""
manage_bets.py — View and edit the betting picks ledger.

Commands:
  list              Show all picks (pending and settled)
  list pending      Show only unsettled picks
  settle <N> W|L|P  Manually set result for pick N (W=win, L=loss, P=push)
  delete <N>        Delete pick number N (from list output)
  clear-pending     Remove all unsettled picks
  clear-date YYYY-MM-DD   Remove all picks from a specific date
  clear-all         Wipe the entire ledger

After editing, commit and push so Render picks it up:
  git add data/bets_ledger.json && git commit -m "update ledger" && git push
"""

import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_PATH = os.path.join(ROOT, "data", "bets_ledger.json")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load():
    if not os.path.isfile(LEDGER_PATH):
        return {"picks": []}
    with open(LEDGER_PATH) as f:
        return json.load(f)


def save(ledger):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)
    print(f"Saved. To sync with Render:")
    print(f"  git add data/bets_ledger.json && git commit -m 'update ledger' && git push")


def result_str(r):
    return r if r else "PENDING"


def list_picks(ledger, filter_pending=False):
    picks = ledger["picks"]
    if filter_pending:
        picks = [p for p in picks if p.get("result") is None]
    if not picks:
        print("No picks found.")
        return
    print(f"\n{'#':>3}  {'Date':10}  {'Type':6}  {'Pick':35}  {'Odds':6}  {'Edge':6}  {'Result':8}")
    print("-" * 85)
    source = ledger["picks"]
    for i, p in enumerate(ledger["picks"]):
        if filter_pending and p.get("result") is not None:
            continue
        bt = p.get("bet_type", "?").upper()
        side = p.get("bet_side") or p.get("bet_team", "?")
        if bt == "SPREAD" and p.get("bet_spread") is not None:
            side += f" {p['bet_spread']:+.1f}"
        if bt == "TOTAL":
            side = f"{p.get('bet_side','?')} {p.get('vegas_total','?')}"
        odds = p.get("bet_odds")
        odds_str = f"{'+' if odds and odds > 0 else ''}{int(odds)}" if odds else "N/A"
        edge = p.get("edge", 0)
        result = result_str(p.get("result"))
        print(f"{i:>3}  {p.get('date','?'):10}  {bt:6}  {side[:35]:35}  {odds_str:6}  {edge:+.1%}  {result}")
    print()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0].lower()
    ledger = load()

    if cmd == "list":
        filter_pending = len(args) > 1 and args[1].lower() == "pending"
        list_picks(ledger, filter_pending=filter_pending)

    elif cmd == "settle":
        if len(args) < 3:
            print("Usage: manage_bets.py settle <N> W|L|P")
            return
        try:
            idx = int(args[1])
        except ValueError:
            print(f"Invalid index: {args[1]}")
            return
        result = args[2].upper()
        if result not in ("W", "L", "P"):
            print(f"Result must be W, L, or P (got {args[2]})")
            return
        picks = ledger["picks"]
        if idx < 0 or idx >= len(picks):
            print(f"Index {idx} out of range (0–{len(picks)-1})")
            return
        pick = picks[idx]
        if pick.get("result") is not None:
            print(f"Pick #{idx} already settled as {pick['result']}. Overwriting.")
        pick["result"] = result
        pick["settled_at"] = datetime.now(timezone.utc).isoformat()
        pick["settled_by"] = "manual"
        from settle_bets import compute_stats
        ledger["stats"] = compute_stats(ledger["picks"])
        save(ledger)
        side = pick.get("bet_side") or pick.get("bet_team", "?")
        print(f"Settled #{idx}: [{pick.get('bet_type','?').upper()}] {side} -> {result}")

    elif cmd == "delete":
        if len(args) < 2:
            print("Usage: manage_bets.py delete <N>")
            return
        try:
            idx = int(args[1])
        except ValueError:
            print(f"Invalid index: {args[1]}")
            return
        picks = ledger["picks"]
        if idx < 0 or idx >= len(picks):
            print(f"Index {idx} out of range (0–{len(picks)-1})")
            return
        removed = picks.pop(idx)
        side = removed.get("bet_side") or removed.get("bet_team", "?")
        print(f"Deleted #{idx}: [{removed.get('bet_type','?').upper()}] {side} ({removed.get('date','?')})")
        ledger["picks"] = picks
        save(ledger)

    elif cmd == "clear-pending":
        before = len(ledger["picks"])
        ledger["picks"] = [p for p in ledger["picks"] if p.get("result") is not None]
        removed = before - len(ledger["picks"])
        print(f"Removed {removed} pending pick(s). {len(ledger['picks'])} remain.")
        save(ledger)

    elif cmd == "clear-date":
        if len(args) < 2:
            print("Usage: manage_bets.py clear-date YYYY-MM-DD")
            return
        date = args[1]
        before = len(ledger["picks"])
        ledger["picks"] = [p for p in ledger["picks"] if p.get("date") != date]
        removed = before - len(ledger["picks"])
        print(f"Removed {removed} pick(s) from {date}. {len(ledger['picks'])} remain.")
        save(ledger)

    elif cmd == "clear-all":
        confirm = input(f"Delete all {len(ledger['picks'])} picks? [y/N] ")
        if confirm.lower() == "y":
            ledger["picks"] = []
            ledger.pop("stats", None)
            print("Ledger cleared.")
            save(ledger)
        else:
            print("Cancelled.")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
