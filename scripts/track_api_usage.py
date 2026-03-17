"""
track_api_usage.py — Track Odds API monthly call usage to stay within the 500/month free tier.

Reads the remaining-requests header returned by The Odds API on every call and persists
a running log to data/api_usage.json. Warns when approaching the monthly limit.

Budget:
  - Free tier: 500 requests/month
  - Warning at:  100 remaining  (80% used)
  - Critical at:  50 remaining  (90% used)
  - Hard stop at: 10 remaining  (98% used) — skip non-essential calls

Usage (called automatically from best_bets.py / save_bets.py):
  from track_api_usage import record_usage, get_remaining, should_skip_call

Or standalone:
  python scripts/track_api_usage.py          # print current status
  python scripts/track_api_usage.py --reset  # reset counter (new billing period)
"""
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USAGE_FILE = os.path.join(DATA_DIR, "api_usage.json")

# Budget thresholds
MONTHLY_BUDGET = 500
WARN_THRESHOLD = 100    # warn: 80% used
CRITICAL_THRESHOLD = 50 # critical: 90% used
HARD_STOP_THRESHOLD = 10 # stop non-essential calls: 98% used


def _load() -> dict:
    if os.path.isfile(USAGE_FILE):
        try:
            with open(USAGE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "requests_remaining": MONTHLY_BUDGET,
        "requests_used": 0,
        "last_updated": None,
        "billing_period_start": datetime.now(timezone.utc).strftime("%Y-%m"),
        "call_log": [],
    }


def _save(data: dict):
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def record_usage(remaining: int | None, endpoint: str = "unknown"):
    """
    Record an API call. Call this after every Odds API request with the
    x-requests-remaining header value.

    Args:
        remaining: Value of x-requests-remaining header (or None if missing)
        endpoint: Which endpoint was called (for logging)
    """
    data = _load()
    now = datetime.now(timezone.utc).isoformat()

    if remaining is not None:
        prev_remaining = data.get("requests_remaining", MONTHLY_BUDGET)
        data["requests_used"] = data.get("requests_used", 0) + max(0, prev_remaining - remaining)
        data["requests_remaining"] = remaining
    else:
        # No header — count 1 request
        data["requests_used"] = data.get("requests_used", 0) + 1
        data["requests_remaining"] = max(0, data.get("requests_remaining", MONTHLY_BUDGET) - 1)

    data["last_updated"] = now
    data.setdefault("call_log", []).append({
        "ts": now,
        "endpoint": endpoint,
        "remaining": data["requests_remaining"],
    })
    # Keep last 100 log entries
    data["call_log"] = data["call_log"][-100:]

    _save(data)

    # Print status and warnings
    remaining_now = data["requests_remaining"]
    if remaining_now <= HARD_STOP_THRESHOLD:
        print(f"  ⛔ ODDS API CRITICAL: only {remaining_now} requests left this month! Skipping non-essential calls.")
    elif remaining_now <= CRITICAL_THRESHOLD:
        print(f"  🔴 ODDS API WARNING: {remaining_now} requests remaining (critical threshold)")
    elif remaining_now <= WARN_THRESHOLD:
        print(f"  🟡 ODDS API: {remaining_now} requests remaining this month")

    return data


def get_remaining() -> int:
    """Return current remaining request count."""
    return _load().get("requests_remaining", MONTHLY_BUDGET)


def should_skip_call(essential: bool = False) -> bool:
    """
    Return True if this call should be skipped to preserve budget.
    Essential calls (e.g., saving today's picks) are never skipped.
    Non-essential calls (e.g., extra odds refreshes) are skipped below HARD_STOP_THRESHOLD.
    """
    if essential:
        return False
    remaining = get_remaining()
    return remaining <= HARD_STOP_THRESHOLD


def print_status():
    data = _load()
    remaining = data.get("requests_remaining", MONTHLY_BUDGET)
    used = data.get("requests_used", 0)
    pct_used = (used / MONTHLY_BUDGET * 100) if MONTHLY_BUDGET > 0 else 0
    period = data.get("billing_period_start", "unknown")
    updated = data.get("last_updated", "never")

    print(f"\nOdds API Usage — {period}")
    print(f"  Used:      {used:3d} / {MONTHLY_BUDGET}  ({pct_used:.1f}%)")
    print(f"  Remaining: {remaining:3d}")
    print(f"  Updated:   {updated}")

    if remaining <= HARD_STOP_THRESHOLD:
        print(f"  Status:    ⛔ CRITICAL — pausing non-essential calls")
    elif remaining <= CRITICAL_THRESHOLD:
        print(f"  Status:    🔴 WARNING")
    elif remaining <= WARN_THRESHOLD:
        print(f"  Status:    🟡 CAUTION")
    else:
        print(f"  Status:    ✅ OK")

    recent = data.get("call_log", [])[-5:]
    if recent:
        print(f"\n  Recent calls:")
        for entry in recent:
            print(f"    {entry.get('ts','')[:19]}  {entry.get('endpoint','?'):30s}  {entry.get('remaining','?')} left")


def reset(new_period: str | None = None):
    """Reset counter for a new billing period."""
    period = new_period or datetime.now(timezone.utc).strftime("%Y-%m")
    data = {
        "requests_remaining": MONTHLY_BUDGET,
        "requests_used": 0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "billing_period_start": period,
        "call_log": [],
    }
    _save(data)
    print(f"Reset Odds API usage counter for {period}")


if __name__ == "__main__":
    if "--reset" in sys.argv:
        period = None
        for a in sys.argv[1:]:
            if a.startswith("--period="):
                period = a.split("=", 1)[1]
        reset(period)
    else:
        print_status()
