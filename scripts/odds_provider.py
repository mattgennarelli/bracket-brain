"""
odds_provider.py — Abstract interface for fetching NCAAB odds from multiple sources.

Providers:
  - odds_api: The Odds API (api.the-odds-api.com) — 500 req/month free
  - betstack: BetStack API (api.betstack.dev) — free forever, 1 req/60 sec

Set ODDS_PROVIDER=odds_api|betstack (default: odds_api)
Set ODDS_API_KEY for The Odds API, or BETSTACK_API_KEY for BetStack.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import requests
except ImportError:
    requests = None


def get_provider():
    """Return the active odds provider based on ODDS_PROVIDER env."""
    name = (os.environ.get("ODDS_PROVIDER") or "odds_api").strip().lower()
    if name == "betstack":
        return BetStackProvider()
    return OddsAPIProvider()


def get_api_key(provider_name=None):
    """Return the API key for the given provider (or active provider)."""
    name = provider_name or (os.environ.get("ODDS_PROVIDER") or "odds_api").strip().lower()
    if name == "betstack":
        return os.environ.get("BETSTACK_API_KEY", "")
    return os.environ.get("ODDS_API_KEY", "")


# ---------------------------------------------------------------------------
# Common output shape (used by best_bets.parse_game consumers)
# ---------------------------------------------------------------------------

def parse_game_common(raw, provider):
    """
    Parse raw game from any provider into common dict:
      home_team, away_team, commence_time,
      ml_home, ml_away, spread_home, spread_line, total_line
    """
    return provider.parse_game(raw)


# ---------------------------------------------------------------------------
# OddsAPIProvider — The Odds API
# ---------------------------------------------------------------------------

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_ncaab"


class OddsAPIProvider:
    """The Odds API (api.the-odds-api.com)."""

    def fetch_games(self, api_key):
        """Fetch today's NCAAB games with spread, ML, totals. Returns raw API response list."""
        if not requests:
            return []
        url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds/"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 401:
            raise ValueError("Invalid API key. Get a free key at https://the-odds-api.com")
        if resp.status_code == 422:
            return []
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        if hasattr(sys.stdout, "write"):
            print(f"  Odds API: {used} requests used, {remaining} remaining this month")
        return resp.json()

    def parse_game(self, raw):
        """Extract common shape from The Odds API game object."""
        home = raw["home_team"]
        away = raw["away_team"]
        commence = raw.get("commence_time", "")

        ml_home = ml_away = None
        spread_home = spread_line = None
        total_line = None

        h2h_home_odds, h2h_away_odds = [], []
        spread_homes, spread_lines = [], []
        total_lines = []

        for bookmaker in raw.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = market["key"]
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if key == "h2h":
                    if home in outcomes:
                        h2h_home_odds.append(outcomes[home]["price"])
                    if away in outcomes:
                        h2h_away_odds.append(outcomes[away]["price"])
                elif key == "spreads":
                    if home in outcomes:
                        spread_homes.append(outcomes[home].get("point", 0))
                        spread_lines.append(outcomes[home]["price"])
                elif key == "totals":
                    over = outcomes.get("Over")
                    if over:
                        total_lines.append(over.get("point", 0))

        def _median(lst):
            if not lst:
                return None
            s = sorted(lst)
            return s[len(s) // 2]

        ml_home = _median(h2h_home_odds)
        ml_away = _median(h2h_away_odds)
        spread_home = _median(spread_homes)
        spread_line = _median(spread_lines)
        total_line = _median(total_lines)

        return {
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "spread_home": spread_home,
            "spread_line": spread_line,
            "total_line": total_line,
        }


# ---------------------------------------------------------------------------
# BetStackProvider — BetStack API
# ---------------------------------------------------------------------------

BETSTACK_BASE = "https://api.betstack.dev/api/v1"
BETSTACK_LEAGUE = "basketball_ncaab"


class BetStackProvider:
    """BetStack API (api.betstack.dev). Free tier: 1 req/60 sec."""

    def fetch_games(self, api_key):
        """Fetch NCAAB events with lines. Returns list of event dicts (with lines embedded or separate)."""
        if not requests:
            return []
        url = f"{BETSTACK_BASE}/events"
        params = {"league": BETSTACK_LEAGUE, "include_lines": "true"}
        headers = {"X-API-Key": api_key}
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 401:
            raise ValueError("Invalid BetStack API key. Get a free key at https://betstack.dev")
        if resp.status_code in (404, 422, 429):
            return []
        resp.raise_for_status()
        events = resp.json() or []
        if not isinstance(events, list):
            events = [events] if events else []
        return events

    def parse_game(self, raw):
        """Extract common shape from BetStack event. Handles embedded lines or line fields on event."""
        home = raw.get("home_team", "")
        away = raw.get("away_team", "")
        commence = raw.get("commence_time", "")

        # Lines may be: embedded "lines" array, or line fields directly on event
        lines = raw.get("lines")
        if isinstance(lines, list) and lines:
            lines = lines[0]
        elif lines is None:
            lines = raw

        ml_home = lines.get("money_line_home")
        ml_away = lines.get("money_line_away")
        spread_home = lines.get("point_spread_home")
        spread_line = lines.get("point_spread_home_line") or lines.get("point_spread_away_line")
        if spread_line is None:
            spread_line = -110
        total_line = lines.get("total_number")

        return {
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "spread_home": spread_home,
            "spread_line": spread_line,
            "total_line": total_line,
        }


# ---------------------------------------------------------------------------
# Scores (for settle_bets) — BetStack includes result in events
# ---------------------------------------------------------------------------

def fetch_scores_betstack(api_key, days_back=3):
    """
    Fetch completed NCAAB scores from BetStack. Returns list of score records
    compatible with settle_bets match_score/get_scores_from_record.
    """
    if not requests:
        return []
    url = f"{BETSTACK_BASE}/events"
    params = {"league": BETSTACK_LEAGUE, "completed": "true"}
    headers = {"X-API-Key": api_key}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        events = resp.json() or []
        if not isinstance(events, list):
            events = [events] if events else []
        all_scores = []
        for ev in events:
            res = ev.get("result") or {}
            if res.get("final") and res.get("home_score") is not None and res.get("away_score") is not None:
                all_scores.append({
                    "home_team": ev.get("home_team", ""),
                    "away_team": ev.get("away_team", ""),
                    "home_score": res["home_score"],
                    "away_score": res["away_score"],
                    "completed": True,
                })
        return all_scores
    except Exception:
        return []
