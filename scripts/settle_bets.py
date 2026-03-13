"""
settle_bets.py — Fetch completed game scores and settle pending picks.

Run after each day's games are complete (or each morning to settle yesterday):
  python scripts/settle_bets.py

Uses ESPN first (free), then Odds API as fallback for unmatched picks (mid-majors).
Matches against pending picks in data/bets_ledger.json and marks each as W / L / P (push).
Updates the ledger and prints a running hit rate.

Usage:
  python scripts/settle_bets.py                    # ESPN + Odds API fallback (if ODDS_API_KEY set)
  python scripts/settle_bets.py --source odds       # Odds API only (requires ODDS_API_KEY)
  python scripts/settle_bets.py --days 3           # check up to 3 days back
  python scripts/settle_bets.py --debug            # log unmatched picks
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT)

try:
    import requests
except ImportError:
    requests = None

from best_bets import _normalize_team_for_match, _american_to_decimal

LEDGER_PATH = os.path.join(DATA_DIR, "bets_ledger.json")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_ncaab"


# ---------------------------------------------------------------------------
# Scores fetch
# ---------------------------------------------------------------------------

def fetch_scores_odds(api_key, days_from=1):
    """Fetch completed NCAAB scores from Odds API. days_from: 0=today, 1=yesterday, etc (max 3)."""
    if not requests:
        return []
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/scores/"
    params = {"apiKey": api_key, "daysFrom": days_from, "dateFormat": "iso"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return [g for g in resp.json() if g.get("completed")]


def fetch_scores_espn(picks, days=3):
    """Fetch completed NCAAB scores from ESPN (free, no API key)."""
    from espn_scores import fetch_espn_scoreboard, build_scores_by_key

    dates_set = set()
    for p in picks:
        d = p.get("date")
        if d:
            dates_set.add(d.replace("-", ""))
    now = datetime.now(timezone.utc)
    for i in range(days + 1):
        dt = now - timedelta(days=i)
        dates_set.add(dt.strftime("%Y%m%d"))
    dates = sorted(dates_set)
    games = fetch_espn_scoreboard(dates)
    completed = [g for g in games if g.get("completed")]
    return build_scores_by_key(completed)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _norm(name):
    return _normalize_team_for_match(name)


def _scores_key(home, away):
    return f"{_norm(home)}|{_norm(away)}"


def match_score(pick, scores_by_key):
    """Return the score record for a pick, or None if not found / not completed."""
    key = _scores_key(pick["home_team"], pick["away_team"])
    if key in scores_by_key:
        return scores_by_key[key]
    # Flipped order fallback (rare but Odds API sometimes reverses)
    key_flipped = _scores_key(pick["away_team"], pick["home_team"])
    if key_flipped in scores_by_key:
        rec = scores_by_key[key_flipped]
        # Flip scores so home/away align with pick
        return {**rec, "home_team": rec["away_team"], "away_team": rec["home_team"],
                "home_score": rec.get("away_score"), "away_score": rec.get("home_score"),
                "_scores_flipped": True}
    return None


def parse_scores_odds(score_rec):
    """Return (home_score, away_score) as floats, or (None, None)."""
    scores = score_rec.get("scores") or []
    home_name = score_rec["home_team"]
    away_name = score_rec["away_team"]
    flipped = score_rec.get("_scores_flipped", False)

    home_score = away_score = None
    for s in scores:
        try:
            val = float(s["score"])
        except (KeyError, ValueError, TypeError):
            continue
        n = s.get("name", "")
        # match_score already swapped home_team/away_team when flipped,
        # so always match by the (already-corrected) home_name/away_name
        if _norm(n) == _norm(home_name):
            home_score = val
        elif _norm(n) == _norm(away_name):
            away_score = val

    # Fallback: just assign by position if names didn't match
    if home_score is None and away_score is None and len(scores) == 2:
        try:
            home_score = float(scores[0]["score"])
            away_score = float(scores[1]["score"])
        except Exception:
            pass

    return home_score, away_score


def get_scores_from_record(score_rec):
    """Extract (home_score, away_score) from score record. Handles ESPN and Odds API formats."""
    if score_rec.get("home_score") is not None and score_rec.get("away_score") is not None:
        return (float(score_rec["home_score"]), float(score_rec["away_score"]))
    return parse_scores_odds(score_rec)


# ---------------------------------------------------------------------------
# Settle logic
# ---------------------------------------------------------------------------

def settle_pick(pick, home_score, away_score):
    """Return 'W', 'L', or 'P' based on pick type and actual scores."""
    bt = pick["bet_type"]
    actual_margin = home_score - away_score  # positive = home won

    if bt == "ml":
        side = pick.get("bet_side", "")
        winner = pick["home_team"] if actual_margin > 0 else pick["away_team"]
        if actual_margin == 0:
            return "P"
        return "W" if _norm(side) == _norm(winner) else "L"

    elif bt == "spread":
        # vegas_spread = home team's spread (e.g. -7.5 means home favored)
        vegas_spread_home = pick.get("vegas_spread")
        bet_team = pick.get("bet_team", "")
        if vegas_spread_home is None:
            return None
        if _norm(bet_team) == _norm(pick["home_team"]):
            cover = actual_margin + vegas_spread_home
        else:
            cover = -(actual_margin + vegas_spread_home)
        if cover > 0:
            return "W"
        if cover < 0:
            return "L"
        return "P"

    elif bt == "total":
        vegas_total = pick.get("vegas_total")
        side = pick.get("bet_side", "OVER")
        if vegas_total is None:
            return None
        actual_total = home_score + away_score
        diff = actual_total - vegas_total
        if diff == 0:
            return "P"
        if side == "OVER":
            return "W" if diff > 0 else "L"
        else:  # UNDER
            return "W" if diff < 0 else "L"

    return None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _american_to_decimal_safe(odds):
    """Convert American odds to decimal format, defaulting to -110 on failure."""
    try:
        return _american_to_decimal(float(odds))
    except (TypeError, ValueError):
        return 1.909  # default -110


def compute_stats(picks):
    """Compute hit rate and ROI stats from a list of pick dicts."""
    by_type = {t: {"picks": 0, "wins": 0, "losses": 0, "pushes": 0}
               for t in ("ml", "spread", "total")}
    total_settled = wins = losses = pushes = 0
    total_wagered = 0.0
    net_units = 0.0

    for p in picks:
        r = p.get("result")
        if r not in ("W", "L", "P"):
            continue
        bt = p.get("bet_type", "ml")
        if bt not in by_type:
            by_type[bt] = {"picks": 0, "wins": 0, "losses": 0, "pushes": 0}
        by_type[bt]["picks"] += 1
        total_settled += 1
        if r == "W":
            by_type[bt]["wins"] += 1
            wins += 1
        elif r == "L":
            by_type[bt]["losses"] += 1
            losses += 1
        elif r == "P":
            by_type[bt]["pushes"] += 1
            pushes += 1

        # Kelly-weighted ROI tracking
        kelly_u = p.get("kelly_units", 1.0)  # default 1 unit if no kelly sizing
        decimal_odds = _american_to_decimal_safe(p.get("bet_odds"))
        if r == "W":
            net_units += kelly_u * (decimal_odds - 1)
            total_wagered += kelly_u
        elif r == "L":
            net_units -= kelly_u
            total_wagered += kelly_u
        # Push: no change to net_units, but still counts as wagered
        elif r == "P":
            total_wagered += kelly_u

    def hit_rate(d):
        decided = d["wins"] + d["losses"]
        return round(d["wins"] / decided, 4) if decided else None

    for bt in by_type:
        by_type[bt]["hit_rate"] = hit_rate(by_type[bt])

    return {
        "total_picks": len([p for p in picks if p.get("result") is not None or True]),
        "settled": total_settled,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": round(wins / (wins + losses), 4) if (wins + losses) else None,
        "by_type": by_type,
        "total_wagered": round(total_wagered, 2),
        "net_units": round(net_units, 2),
        "roi_pct": round(net_units / total_wagered * 100, 2) if total_wagered > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Settle pending bets with actual scores")
    parser.add_argument("--source", choices=["espn", "odds"], default="espn",
                        help="Score source: espn (free) or odds (requires API key). Default uses ESPN first, then Odds API fallback.")
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY", ""))
    parser.add_argument("--days", type=int, default=3,
                        help="How many days back to fetch scores (default 3)")
    parser.add_argument("--debug", action="store_true",
                        help="Log unmatched picks")
    args = parser.parse_args()

    if args.source == "odds" and not args.api_key:
        print("ERROR: Odds API requires ODDS_API_KEY. Use --source espn for free ESPN scores.")
        sys.exit(1)

    if not os.path.isfile(LEDGER_PATH):
        print("No ledger found. Run save_bets.py first.")
        return

    with open(LEDGER_PATH) as f:
        ledger = json.load(f)

    pending = [p for p in ledger["picks"] if p.get("result") is None]
    if not pending:
        print("No pending picks to settle.")
        _print_stats(compute_stats(ledger["picks"]))
        return

    settled_count = 0
    now = datetime.now(timezone.utc).isoformat()
    settle_log_entries = []

    if args.source == "odds":
        # Odds API only
        print(f"Found {len(pending)} pending pick(s). Fetching scores from Odds API...")
        all_scores = []
        for d in range(0, min(args.days + 1, 4)):
            try:
                scores = fetch_scores_odds(args.api_key, days_from=d)
                all_scores.extend(scores)
            except Exception as e:
                print(f"  Warning: could not fetch scores for daysFrom={d}: {e}")
        scores_by_key = {}
        for g in all_scores:
            k = _scores_key(g["home_team"], g["away_team"])
            scores_by_key[k] = g
        print(f"  {len(all_scores)} completed game(s) found")
    else:
        # Phase 1: ESPN (free)
        print(f"Found {len(pending)} pending pick(s). Fetching scores from ESPN...")
        scores_by_key = fetch_scores_espn(pending, days=args.days)
        print(f"  {len(scores_by_key)} completed game(s) from ESPN")

    for pick in ledger["picks"]:
        if pick.get("result") is not None:
            continue
        score_rec = match_score(pick, scores_by_key)
        if not score_rec:
            if args.debug:
                print(f"  [UNMATCHED] {pick['away_team']} @ {pick['home_team']} (no ESPN game)")
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
        pick["settled_at"] = now
        settled_count += 1

        settle_log_entries.append({
            "pick_idx": ledger["picks"].index(pick),
            "home_team": pick["home_team"],
            "away_team": pick["away_team"],
            "bet_type": pick["bet_type"],
            "bet_side": pick.get("bet_side") or pick.get("bet_team"),
            "result": result,
            "home_score": home_score,
            "away_score": away_score,
            "settled_at": now,
        })

        side = pick.get("bet_side") or pick.get("bet_team", "?")
        print(f"  [{result}] {pick['bet_type'].upper()} {side}  "
              f"({pick['home_team']} {int(home_score)}-{int(away_score)} {pick['away_team']})")

    # Phase 2: Odds API fallback for still-pending (mid-major games ESPN may not have)
    still_pending = [p for p in ledger["picks"] if p.get("result") is None]
    if still_pending and args.api_key and args.source == "espn":
        print(f"\n  {len(still_pending)} pick(s) still pending. Trying Odds API fallback...")
        all_odds = []
        for d in range(0, min(args.days + 1, 4)):
            try:
                scores = fetch_scores_odds(args.api_key, days_from=d)
                all_odds.extend(scores)
            except Exception as e:
                print(f"  Warning: could not fetch Odds API for daysFrom={d}: {e}")
        for g in all_odds:
            k = _scores_key(g["home_team"], g["away_team"])
            if k not in scores_by_key:
                scores_by_key[k] = g
        print(f"  {len(all_odds)} game(s) from Odds API")

        for pick in ledger["picks"]:
            if pick.get("result") is not None:
                continue
            score_rec = match_score(pick, scores_by_key)
            if not score_rec:
                if args.debug:
                    print(f"  [UNMATCHED] {pick['away_team']} @ {pick['home_team']} (no Odds API game)")
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
            pick["settled_at"] = now
            settled_count += 1

            settle_log_entries.append({
                "pick_idx": ledger["picks"].index(pick),
                "home_team": pick["home_team"],
                "away_team": pick["away_team"],
                "bet_type": pick["bet_type"],
                "bet_side": pick.get("bet_side") or pick.get("bet_team"),
                "result": result,
                "home_score": home_score,
                "away_score": away_score,
                "settled_at": now,
            })

            side = pick.get("bet_side") or pick.get("bet_team", "?")
            print(f"  [{result}] {pick['bet_type'].upper()} {side}  "
                  f"({pick['home_team']} {int(home_score)}-{int(away_score)} {pick['away_team']})")

    # Write settle log (even if 0 settled, for midnight run audit trail)
    log_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = os.path.join(DATA_DIR, f"settle_log_{log_date}.json")
    with open(log_path, "w") as f:
        json.dump({
            "date": log_date,
            "run_at": now,
            "settled_count": settled_count,
            "decisions": settle_log_entries,
        }, f, indent=2)
    if settled_count > 0:
        print(f"  Logged to {log_path}")

    if settled_count == 0:
        print("  No picks could be settled yet (games may not be complete).")
    else:
        print(f"\n  Settled {settled_count} pick(s).")

    # Recompute stats and save
    ledger["stats"] = compute_stats(ledger["picks"])
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)

    _print_stats(ledger["stats"])


def _print_stats(stats):
    hr = stats.get("hit_rate")
    hr_str = f"{hr:.1%}" if hr is not None else "—"
    decided = stats.get("wins", 0) + stats.get("losses", 0)
    print(f"\n  Overall: {stats['wins']}W-{stats['losses']}L-{stats['pushes']}P  "
          f"({decided} decided)  Hit rate: {hr_str}")
    if stats.get("total_wagered", 0) > 0:
        print(f"  Units wagered: {stats['total_wagered']:.1f}  "
              f"Net: {stats['net_units']:+.1f}  ROI: {stats['roi_pct']:+.1f}%")
    for bt, s in stats.get("by_type", {}).items():
        if s["picks"] == 0:
            continue
        bhr = s.get("hit_rate")
        bhr_str = f"{bhr:.1%}" if bhr is not None else "—"
        print(f"    {bt.upper():8s}: {s['wins']}W-{s['losses']}L-{s['pushes']}P  {bhr_str}")


if __name__ == "__main__":
    main()
