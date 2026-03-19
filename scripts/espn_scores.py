"""
espn_scores.py — Fetch NCAAB scores from ESPN's free API.

Used for live score display and settlement. No API key required.
ESPN API: http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard

Team name matching: ESPN uses displayName ("Seton Hall Pirates") and shortDisplayName ("Seton Hall").
We normalize with engine._normalize_team_for_match and build keys for matching to Odds API picks.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import requests
except ImportError:
    requests = None

from engine import _normalize_team_for_match, _strip_mascot

ESPN_SCOREBOARD_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"

# ESPN name variants that don't match Odds API after normalization
# Maps ESPN normalized name -> canonical form used in picks
_ESPN_ALIASES = {
    "ohio st": "ohio state",  # ESPN shortDisplayName "Ohio St." vs Odds "Ohio State"
    "ole miss": "mississippi",  # ESPN uses "Ole Miss"
}


def _norm(name):
    """Normalize team name for matching."""
    if not name or not isinstance(name, str):
        return ""
    n = _normalize_team_for_match(name)
    return _ESPN_ALIASES.get(n, n)


def _scores_key(home, away):
    """Build lookup key for home|away pair."""
    return f"{_norm(home)}|{_norm(away)}"


def _team_name_variants(team):
    """Return short/canonical aliases for an ESPN team payload."""
    if not isinstance(team, dict):
        return []

    names = []
    for key in ("shortDisplayName", "displayName", "name", "location", "abbreviation"):
        value = team.get(key)
        if not value:
            continue
        value = " ".join(str(value).split()).strip()
        if value:
            names.append(value)
        stripped = _strip_mascot(value)
        if stripped and stripped != value:
            names.append(stripped)

    seen = set()
    out = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def fetch_espn_scoreboard(dates):
    """
    Fetch scoreboard for given dates. dates: list of 'YYYYMMDD' strings.
    Returns list of game dicts with: home_team, away_team, home_score, away_score,
    completed, status_detail, display_clock, period.
    """
    if not requests:
        return []

    all_games = []
    for d in dates:
        try:
            resp = requests.get(
                ESPN_SCOREBOARD_URL,
                params={"dates": d, "limit": 100},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        for event in data.get("events", []):
            game = _parse_espn_event(event)
            if game:
                all_games.append(game)

    return all_games


def _parse_espn_event(event):
    """Parse ESPN event into our game format."""
    comps = event.get("competitions", [])
    if not comps:
        return None

    comp = comps[0]
    competitors = comp.get("competitors", [])
    if len(competitors) < 2:
        return None

    home = away = None
    for c in competitors:
        team = c.get("team", {})
        variants = _team_name_variants(team)
        name = variants[0] if variants else ""
        ha = c.get("homeAway", "")
        try:
            score = float(c.get("score", 0))
        except (TypeError, ValueError):
            score = None

        if ha == "home":
            home = {"name": name, "aliases": variants, "score": score}
        elif ha == "away":
            away = {"name": name, "aliases": variants, "score": score}

    if not home or not away:
        return None

    status = comp.get("status", {}) or {}
    status_type = status.get("type", {}) or {}
    state = status_type.get("state", "")
    completed = status_type.get("completed", False) or (state == "post")
    detail = status_type.get("detail", "") or ""
    display_clock = status.get("displayClock", "") or ""
    period = status.get("period", 0)

    return {
        "home_team": home["name"],
        "away_team": away["name"],
        "home_aliases": home.get("aliases", []),
        "away_aliases": away.get("aliases", []),
        "home_score": home["score"],
        "away_score": away["score"],
        "completed": completed,
        "status_detail": detail,
        "display_clock": display_clock,
        "period": period,
    }


def build_scores_by_key(games):
    """
    Build dict keyed by scores_key (normalized home|away) for matching to picks.
    Adds both displayName and shortDisplayName variants so we match regardless
    of which format the pick uses.
    """
    out = {}
    for g in games:
        home_names = [g["home_team"], *g.get("home_aliases", [])]
        away_names = [g["away_team"], *g.get("away_aliases", [])]
        for home in home_names:
            for away in away_names:
                key = _scores_key(home, away)
                out[key] = g
                key_flip = _scores_key(away, home)
                if key_flip not in out:
                    out[key_flip] = {
                        **g,
                        "home_team": away,
                        "away_team": home,
                        "home_score": g["away_score"],
                        "away_score": g["home_score"],
                    }
    return out


def fetch_scores_for_picks(picks, days=2):
    """
    Fetch ESPN scores for dates covering the given picks.
    picks: list of pick dicts with home_team, away_team, date.
    Returns dict: { scores_key: { home_score, away_score, completed, ... } }
    """
    if not picks:
        return {}

    # Collect unique dates from picks
    dates_set = set()
    for p in picks:
        d = p.get("date")
        if d:
            dates_set.add(d.replace("-", ""))  # YYYY-MM-DD -> YYYYMMDD

    # Also include today and yesterday for live games
    now = datetime.now(timezone.utc)
    for i in range(days + 1):
        dt = now - timedelta(days=i)
        dates_set.add(dt.strftime("%Y%m%d"))

    dates = sorted(dates_set)
    games = fetch_espn_scoreboard(dates)
    return build_scores_by_key(games)
