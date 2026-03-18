"""
fetch_injuries_live.py — Pull real-time injury data from ESPN for tournament teams.

Uses the free, keyless ESPN API to fetch injury status for all active tournament teams,
then merges with EvanMiya BPR share data to produce the injuries_YYYY.json format
consumed by engine.py's calc_injury_penalty().

Run this every few hours during the tournament to stay current:
  python scripts/fetch_injuries_live.py 2026
  python scripts/fetch_injuries_live.py 2026 --merge    # merge into existing file, don't replace

ESPN injury statuses are mapped to our four-level system:
  "Out"          -> "out"       (full penalty)
  "Doubtful"     -> "doubtful"  (0.7× penalty)
  "Questionable" -> "questionable" (0.4× penalty)
  "Probable"     -> "probable"  (0.1× penalty — essentially healthy, included for completeness)

BPR share is loaded from data/evanmiya_players_YYYY.csv (same format as fetch_injuries.py).
If no BPR data is available for a player, falls back to a conservative 0.10 estimate,
which keeps them above the _MIN_BPR_SHARE threshold (0.05) but modest enough not to
over-penalize when we're uncertain of the player's true impact.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, ROOT)

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

from engine import _normalize_team_for_match

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams?limit=500"
)
ESPN_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams/{team_id}/injuries"
)
ESPN_NEWS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/news?limit=50"
)
ESPN_TEAM_NEWS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams/{team_id}/news?limit=20"
)

# ESPN status -> our injury severity level
STATUS_MAP = {
    "out": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "probable": "probable",
    "day-to-day": "day-to-day",
    "injured reserve": "out",
    "suspension": "out",
    "not with team": "out",
}

# Minimum BPR share — must match engine._MIN_BPR_SHARE
MIN_BPR_SHARE = 0.05

# Default BPR share when no EvanMiya data is available for a player
FALLBACK_BPR_SHARE = 0.10

# ---------------------------------------------------------------------------
# ESPN team list (cached within a run)
# ---------------------------------------------------------------------------

_espn_teams: dict = {}  # normalised_name -> {"id": ..., "display_name": ...}


def _fetch_espn_teams() -> dict:
    """Fetch all NCAAB teams from ESPN and build normalised-name → id map."""
    global _espn_teams
    if _espn_teams:
        return _espn_teams
    try:
        r = requests.get(ESPN_TEAMS_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  WARN: failed to fetch ESPN team list: {e}")
        return {}

    for team in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        t = team.get("team", {})
        tid = t.get("id")
        display = t.get("displayName", "")
        short = t.get("shortDisplayName", "")
        abbr = t.get("abbreviation", "")
        norm = _normalize_team_for_match(display)
        _espn_teams[norm] = {"id": tid, "display_name": display, "short": short, "abbr": abbr}
        # Also index by short name and abbreviation for broader matching
        for alt in [short, abbr]:
            if alt:
                _espn_teams.setdefault(_normalize_team_for_match(alt), {"id": tid, "display_name": display, "short": short, "abbr": abbr})
    print(f"  Fetched {len(_espn_teams)} ESPN team entries.")
    return _espn_teams


def _find_espn_id(team_name: str) -> str | None:
    """Return ESPN team ID for a team name, or None if not found."""
    teams = _fetch_espn_teams()
    norm = _normalize_team_for_match(team_name)
    if norm in teams:
        return teams[norm]["id"]
    # Try expanding "st" -> "state" for abbreviated names like "Ohio St." -> "ohio st" -> "ohio state"
    import re as _re
    expanded = _re.sub(r'\bst\b', 'state', norm)
    if expanded != norm and expanded in teams:
        return teams[expanded]["id"]
    # Progressive substring match — threshold 0.65 so "ohio st" (7) matches "ohio state" (10)
    for key, val in teams.items():
        if norm in key or key in norm:
            if len(norm) >= len(key) * 0.65:
                return val["id"]
    return None


# ---------------------------------------------------------------------------
# EvanMiya BPR data
# ---------------------------------------------------------------------------

def _load_bpr_data(year: int) -> dict:
    """Load evanmiya_players_YYYY.csv → dict normalised_team -> dict player_lower -> {bpr, bpr_share, poss}.

    Returns absolute BPR in pts/100 possessions (e.g. 14.8 for a star player) and poss
    (season possessions). Both are needed for the weighted on/off injury calculation.
    """
    path = os.path.join(DATA_DIR, f"evanmiya_players_{year}.csv")
    if not os.path.isfile(path):
        return {}
    try:
        import csv
        # First pass: collect raw BPR + poss values per team
        raw: dict = {}  # norm_team -> {player_lower: (bpr_abs, poss)}
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                team = (row.get("team") or row.get("Team") or "").strip()
                player = (row.get("player") or row.get("Player") or row.get("name") or "").strip()
                bpr_str = row.get("bpr") or row.get("BPR") or "0"
                poss_str = row.get("poss") or row.get("Poss") or "0"
                try:
                    bpr_abs = float(str(bpr_str).replace("%", "").strip())
                except (ValueError, TypeError):
                    bpr_abs = 0.0
                try:
                    poss = float(str(poss_str).strip())
                except (ValueError, TypeError):
                    poss = 0.0
                if team and player:
                    norm_team = _normalize_team_for_match(team)
                    raw.setdefault(norm_team, {})[player.lower()] = (bpr_abs, poss)

        # Second pass: compute team BPR totals for share calculation
        result: dict = {}
        for norm_team, players in raw.items():
            team_total = sum(v for v, _ in players.values() if v > 0) or 1.0
            result[norm_team] = {
                name: {
                    "bpr": round(bpr, 4),
                    "bpr_share": round(max(bpr, 0) / team_total, 4),
                    "poss": round(poss, 0),
                }
                for name, (bpr, poss) in players.items()
            }

        print(f"  Loaded BPR data for {len(result)} teams from {os.path.basename(path)}")
        return result
    except Exception as e:
        print(f"  WARN: failed to load BPR data: {e}")
        return {}


# ---------------------------------------------------------------------------
# ESPN injury fetch
# ---------------------------------------------------------------------------

def _find_bpr_team(team_name: str, bpr_data: dict) -> dict:
    """Find BPR data for a team, handling ESPN vs EvanMiya name mismatches.

    ESPN names include mascots ("UCLA Bruins" → "ucla bruins") while EvanMiya
    names are just the school ("UCLA" → "ucla"). This function tries:
    1. Exact match on normalised name
    2. "st" → "state" expansion (e.g., "ohio st" → "ohio state")
    3. Substring match — EvanMiya key contained in ESPN name or vice versa
    4. Strip trailing words (mascots) and re-normalize each prefix
       e.g., "SMU Mustangs" → try normalize("SMU") → "southern methodist"
    """
    norm = _normalize_team_for_match(team_name)

    # 1. Exact match
    if norm in bpr_data:
        return bpr_data[norm]

    # 2. "st" → "state" expansion
    expanded = re.sub(r'\bst\b', 'state', norm)
    if expanded != norm and expanded in bpr_data:
        return bpr_data[expanded]

    # 3. Substring match — prefer longest matching key (most specific)
    best_key = None
    best_len = 0
    for key in bpr_data:
        if key in norm or norm in key:
            if len(key) > best_len:
                best_key = key
                best_len = len(key)
    if best_key and best_len >= 3:  # minimum 3 chars to avoid spurious matches
        return bpr_data[best_key]

    # 4. Strip trailing words (likely mascot) and re-normalize each prefix.
    #    "SMU Mustangs" → try normalize("SMU Mustangs"), normalize("SMU") → "southern methodist"
    #    This catches cases where the school abbreviation maps to a full name via _NAME_ALIASES.
    words = team_name.strip().split()
    for end in range(len(words) - 1, 0, -1):
        prefix = " ".join(words[:end])
        prefix_norm = _normalize_team_for_match(prefix)
        if prefix_norm and prefix_norm != norm and prefix_norm in bpr_data:
            return bpr_data[prefix_norm]

    return {}


def fetch_injuries_for_team(team_name: str, espn_id: str | None, bpr_data: dict) -> list:
    """Fetch injury report from ESPN for one team. Returns list of injury dicts."""
    if not espn_id:
        return []

    try:
        url = ESPN_INJURIES_URL.format(team_id=espn_id)
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    WARN: ESPN injury fetch failed for {team_name}: {e}")
        return []

    injuries = []
    team_bpr = _find_bpr_team(team_name, bpr_data)

    for item in data.get("injuries", []):
        athlete = item.get("athlete", {})
        player_name = athlete.get("displayName", "").strip()
        if not player_name:
            continue

        status_raw = item.get("status", "").strip().lower()
        status = STATUS_MAP.get(status_raw)
        if status is None:
            continue  # unknown status — skip

        # Look up BPR for this player (absolute pts/100 possessions + share of team total)
        player_key = player_name.lower()
        player_data = team_bpr.get(player_key)
        if player_data is None:
            # Try last-name-only match
            last = player_key.split()[-1] if player_key else ""
            for k, v in team_bpr.items():
                if last and last in k:
                    player_data = v
                    break
        if player_data is None:
            # Fallback: conservative values when no EvanMiya data found
            player_data = {"bpr": FALLBACK_BPR_SHARE * 10, "bpr_share": FALLBACK_BPR_SHARE}

        bpr_abs = player_data["bpr"]
        bpr_share = player_data["bpr_share"]

        if bpr_share < MIN_BPR_SHARE and bpr_abs < (MIN_BPR_SHARE * 10):
            continue  # below threshold — skip fringe players

        injuries.append({
            "player": player_name,
            "status": status,
            "bpr": round(float(bpr_abs), 4),           # absolute pts/100 possessions
            "poss": round(float(player_data.get("poss", 0)), 0),  # season possessions
            "bpr_share": round(float(bpr_share), 4),   # fraction of team BPR total
            "importance": round(float(bpr_share), 4),  # legacy field
            "source": "espn",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    return injuries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_roster(team_name: str, bpr_data: dict) -> list:
    """Return full roster as [{player, bpr, poss}, ...] sorted by poss desc.

    Used by calc_injury_penalty to compute the healthy-roster weighted avg BPR
    for the on/off approach (instead of assuming replacement is average D1 player).
    """
    team_bpr = _find_bpr_team(team_name, bpr_data)
    roster = []
    for name, data in team_bpr.items():
        bpr = data.get("bpr", 0.0)
        poss = data.get("poss", 0.0)
        if poss > 0:
            roster.append({"player": name, "bpr": round(bpr, 4), "poss": round(poss, 0)})
    roster.sort(key=lambda x: x["poss"], reverse=True)
    return roster


def fetch_all_injuries(year: int = 2026, team_names: list | None = None, merge: bool = False) -> dict:
    """
    Fetch live injuries for all teams (or a subset).
    Returns dict team -> {"injuries": [...], "roster": [...]}.

    If merge=True, loads the existing injuries_YYYY.json and only updates teams
    for which we find ESPN data (preserving manually entered records for others).
    """
    bpr_data = _load_bpr_data(year)

    # Determine which teams to fetch
    if team_names is None:
        # Load from teams_merged so we only fetch teams we actually have data for
        teams_path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
        if os.path.isfile(teams_path):
            with open(teams_path) as f:
                raw = json.load(f)
            if isinstance(raw, list):
                team_names = [v.get("team") for v in raw if isinstance(v, dict) and v.get("team")]
            else:
                team_names = [v.get("team", k) for k, v in raw.items() if isinstance(v, dict)]
        else:
            print(f"  No teams_merged_{year}.json found — fetching ESPN team list only")
            team_names = list(_fetch_espn_teams().values())

    # Load existing data if merging (handle both old list format and new dict format)
    out_path = os.path.join(DATA_DIR, f"injuries_{year}.json")
    if merge and os.path.isfile(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        # Normalise old format {team: [list]} -> {team: {"injuries": list, "roster": []}}
        result = {}
        for k, v in existing.items():
            if isinstance(v, list):
                result[k] = {"injuries": v, "roster": []}
            else:
                result[k] = v
    else:
        result = {}

    fetched = 0
    skipped = 0
    total = len(team_names)

    print(f"  Fetching ESPN injury data for {total} teams...")
    for i, team in enumerate(team_names, 1):
        espn_id = _find_espn_id(team)
        if not espn_id:
            skipped += 1
            continue
        injuries = fetch_injuries_for_team(team, espn_id, bpr_data)
        roster = _build_roster(team, bpr_data)
        if injuries:
            result[team] = {"injuries": injuries, "roster": roster}
            fetched += 1
            print(f"  [{i}/{total}] {team}: {len(injuries)} injured player(s)")
        elif merge and team in result:
            # Keep existing injury list but refresh roster (BPR data may have updated)
            result[team]["roster"] = roster
        else:
            result[team] = {"injuries": [], "roster": roster}
        # Polite delay — ESPN is free and we want to keep it that way
        time.sleep(0.15)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Done. {fetched} teams with injuries, {skipped} not found in ESPN.")
    print(f"  Wrote {out_path}")
    return result


# ---------------------------------------------------------------------------
# News-based injury parsing (ESPN news API)
# ---------------------------------------------------------------------------

# Patterns that indicate a player is OUT or will miss time
_OUT_PATTERNS = [
    r"\b(?:ruled out|will not play|will miss|won't play|is out)\b",
    r"\bwithout\s+\w+",  # "without Patrick Ngongba"
    r"\b(?:sidelined|shelved|shut down|lost for)\b",
    r"\b(?:out (?:for|of|indefinitely|with))\b",
    r"\b(?:season[- ]ending|torn|fracture[ds]?|surgery)\b",
    r"\b(?:suspended|ineligible)\b",
]

_DOUBTFUL_PATTERNS = [
    r"\b(?:doubtful|unlikely to play|not expected to play|not likely)\b",
    r"\b(?:likely without)\b",
]

_QUESTIONABLE_PATTERNS = [
    r"\b(?:questionable|uncertain|game-time decision|game time decision)\b",
    r"\b(?:may not play|might not play|status unclear|iffy)\b",
]

_DAY_TO_DAY_PATTERNS = [
    r"\b(?:day[- ]to[- ]day|d2d|nursing|dealing with)\b",
]

_CLEARED_PATTERNS = [
    r"\b(?:cleared to play|will play|set to return|expected to play|back in lineup)\b",
    r"\b(?:return[s]? to (?:practice|lineup|action)|good to go)\b",
]

# Compiled pattern sets for status detection
_STATUS_REGEXES = [
    ("out", [re.compile(p, re.IGNORECASE) for p in _OUT_PATTERNS]),
    ("doubtful", [re.compile(p, re.IGNORECASE) for p in _DOUBTFUL_PATTERNS]),
    ("questionable", [re.compile(p, re.IGNORECASE) for p in _QUESTIONABLE_PATTERNS]),
    ("day-to-day", [re.compile(p, re.IGNORECASE) for p in _DAY_TO_DAY_PATTERNS]),
    ("cleared", [re.compile(p, re.IGNORECASE) for p in _CLEARED_PATTERNS]),
]


def _detect_status(text: str) -> str | None:
    """Detect injury status from a text string. Returns status or None."""
    for status, regexes in _STATUS_REGEXES:
        for rx in regexes:
            if rx.search(text):
                return status
    return None


def _extract_player_name_from_text(text: str, roster_names: set) -> str | None:
    """Try to find a known roster player mentioned in the text.

    Strategy:
    1. Check if any full roster name appears in the text (case-insensitive).
    2. Check if first+last name parts both appear nearby (avoids "Brown" matching any Brown).
    Only use full-name matching for news parsing to avoid false positives.
    """
    text_lower = text.lower()

    # 1. Full name match (most reliable) — prefer longest match first
    best_match = None
    best_len = 0
    for name in roster_names:
        if name in text_lower and len(name) > best_len:
            best_match = name
            best_len = len(name)
    if best_match:
        return best_match

    # 2. First + last name both appear, and last name is uncommon (>= 4 chars)
    for name in roster_names:
        parts = name.split()
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            if len(last) >= 4 and first in text_lower and last in text_lower:
                # Verify they're close together (within 30 chars)
                fi = text_lower.find(first)
                li = text_lower.find(last)
                if abs(fi - li) < 30:
                    return name

    return None


def _build_espn_team_lookup() -> dict:
    """Build ESPN display_name -> team_id lookup from the teams cache."""
    teams = _fetch_espn_teams()
    # Also build reverse: id -> display_name for article team matching
    return teams


def _parse_news_articles(articles: list, team_name: str, team_bpr: dict) -> list:
    """Parse a list of ESPN news articles for injury information.

    Each article has 'headline' and optionally 'description'.
    Returns list of injury dicts matching our standard format.
    """
    roster_names = set(team_bpr.keys())  # lowercase player names
    found = []
    seen_players = set()

    for article in articles:
        headline = article.get("headline", "")
        description = article.get("description", "")
        combined = f"{headline} {description}"

        # Detect injury status from text
        status = _detect_status(combined)
        if status is None:
            continue
        if status == "cleared":
            # Player is healthy — we note this to potentially remove them later
            player = _extract_player_name_from_text(combined, roster_names)
            if player and player not in seen_players:
                seen_players.add(player)
                found.append({
                    "_player_key": player,
                    "_cleared": True,
                    "headline": headline,
                })
            continue

        # Find which player the article is about
        player = _extract_player_name_from_text(combined, roster_names)
        if not player:
            continue
        if player in seen_players:
            continue  # already found from an earlier (newer) article
        seen_players.add(player)

        player_data = team_bpr.get(player)
        if player_data is None:
            player_data = {"bpr": FALLBACK_BPR_SHARE * 10, "bpr_share": FALLBACK_BPR_SHARE, "poss": 0}

        bpr_abs = player_data["bpr"]
        bpr_share = player_data["bpr_share"]

        if bpr_share < MIN_BPR_SHARE and bpr_abs < (MIN_BPR_SHARE * 10):
            continue  # below threshold — skip fringe players

        # Capitalize player name for display
        display_name = " ".join(w.capitalize() for w in player.split())

        found.append({
            "_player_key": player,
            "_cleared": False,
            "player": display_name,
            "status": status,
            "bpr": round(float(bpr_abs), 4),
            "poss": round(float(player_data.get("poss", 0)), 0),
            "bpr_share": round(float(bpr_share), 4),
            "importance": round(float(bpr_share), 4),
            "source": "espn_news",
            "headline": headline,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    return found


def fetch_injuries_from_news(year: int = 2026) -> dict:
    """Fetch injury info from ESPN news articles and merge into injuries JSON.

    This complements the structured injury API (which often returns empty for NCAAB)
    by parsing news headlines and descriptions for injury keywords.
    """
    bpr_data = _load_bpr_data(year)

    # Load team list from teams_merged
    teams_path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
    if not os.path.isfile(teams_path):
        print(f"  ERROR: {teams_path} not found.")
        return {}
    with open(teams_path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        team_names = [v.get("team") for v in raw if isinstance(v, dict) and v.get("team")]
    else:
        team_names = [v.get("team", k) for k, v in raw.items() if isinstance(v, dict)]

    # Load existing injuries to merge into
    out_path = os.path.join(DATA_DIR, f"injuries_{year}.json")
    if os.path.isfile(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        result = {}
        for k, v in existing.items():
            if isinstance(v, list):
                result[k] = {"injuries": v, "roster": []}
            else:
                result[k] = v
    else:
        result = {}

    # Step 1: Fetch general NCAAB news
    print("  Fetching ESPN general news feed...")
    general_articles = []
    try:
        r = requests.get(ESPN_NEWS_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
        general_articles = data.get("articles", [])
        print(f"    Got {len(general_articles)} general articles.")
    except Exception as e:
        print(f"    WARN: failed to fetch general news: {e}")

    # Step 2: Fetch team-specific news for each tournament team
    news_found = 0
    cleared_found = 0
    total = len(team_names)

    print(f"  Fetching team-specific ESPN news for {total} teams...")
    for i, team in enumerate(team_names, 1):
        espn_id = _find_espn_id(team)
        if not espn_id:
            continue

        team_bpr = _find_bpr_team(team, bpr_data)
        if not team_bpr:
            continue

        # Collect articles for this team: team-specific + any general articles
        team_articles = []
        try:
            url = ESPN_TEAM_NEWS_URL.format(team_id=espn_id)
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            team_articles = data.get("articles", [])
        except Exception as e:
            print(f"    WARN: news fetch failed for {team} (id={espn_id}): {e}")

        # Also scan general articles for mentions of this team
        team_norm = _normalize_team_for_match(team)
        for art in general_articles:
            headline = (art.get("headline", "") + " " + art.get("description", "")).lower()
            # Check if team name appears in the article
            if team_norm in headline or team.lower() in headline:
                team_articles.append(art)

        if not team_articles:
            time.sleep(0.1)
            continue

        # Parse articles for injury info
        parsed = _parse_news_articles(team_articles, team, team_bpr)

        # Ensure team entry exists
        if team not in result:
            roster = _build_roster(team, bpr_data)
            result[team] = {"injuries": [], "roster": roster}

        existing_injuries = result[team].get("injuries", [])
        existing_players = {inj["player"].lower() for inj in existing_injuries}

        for entry in parsed:
            player_key = entry.pop("_player_key")
            is_cleared = entry.pop("_cleared")
            headline = entry.pop("headline", "")

            if is_cleared:
                # Remove player from injuries if they've been cleared
                before = len(existing_injuries)
                existing_injuries = [
                    inj for inj in existing_injuries
                    if inj["player"].lower() != player_key
                ]
                if len(existing_injuries) < before:
                    cleared_found += 1
                    print(f"    [{i}/{total}] {team}: CLEARED {player_key} (from: {headline[:60]})")
                continue

            if player_key in existing_players:
                # Update status of existing entry
                for inj in existing_injuries:
                    if inj["player"].lower() == player_key:
                        old_status = inj.get("status")
                        inj["status"] = entry["status"]
                        inj["source"] = "espn_news"
                        inj["updated_at"] = entry["updated_at"]
                        if old_status != entry["status"]:
                            print(f"    [{i}/{total}] {team}: UPDATED {entry['player']} "
                                  f"{old_status} -> {entry['status']} (from: {headline[:60]})")
                        break
            else:
                # New injury — add it
                existing_injuries.append(entry)
                existing_players.add(player_key)
                news_found += 1
                print(f"    [{i}/{total}] {team}: NEW {entry['player']} ({entry['status']}) "
                      f"(from: {headline[:60]})")

        result[team]["injuries"] = existing_injuries

        # Polite delay
        time.sleep(0.15)

    # Write merged result
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  News parsing done. {news_found} new injuries found, {cleared_found} cleared.")
    print(f"  Wrote {out_path}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch live ESPN injury data")
    parser.add_argument("year", nargs="?", type=int, default=2026)
    parser.add_argument("--merge", action="store_true",
                        help="Merge into existing injuries JSON instead of replacing")
    parser.add_argument("--news", action="store_true",
                        help="Parse ESPN news articles for injury info (supplements structured API)")
    parser.add_argument("--team", help="Fetch a single team only (for testing)")
    args = parser.parse_args()

    if args.news:
        print(f"Parsing ESPN news for injury info ({args.year})...")
        fetch_injuries_from_news(args.year)
    elif args.team:
        print(f"Fetching live ESPN injuries for {args.year}...")
        bpr = _load_bpr_data(args.year)
        espn_id = _find_espn_id(args.team)
        print(f"  ESPN ID for '{args.team}': {espn_id}")
        injuries = fetch_injuries_for_team(args.team, espn_id, bpr)
        print(json.dumps({args.team: injuries}, indent=2))
    else:
        print(f"Fetching live ESPN injuries for {args.year}...")
        fetch_all_injuries(args.year, merge=args.merge)


if __name__ == "__main__":
    main()
