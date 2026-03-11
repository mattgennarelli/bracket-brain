"""
Fetch and convert historical brackets from danvk/march-madness-data (1985-2017).
Downloads JSON from GitHub, converts to our bracket format, merges with team stats.
"""
import json
import os
import re
import sys

try:
    import requests
except ImportError:
    requests = None

# Fallback region names when year isn't in the lookup table
_DEFAULT_REGION_NAMES = ["South", "East", "Midwest", "West"]

# Actual NCAA region names per year, keyed by danvk region index.
# danvk always orders regions as [#1overall, #4overall, #2overall, #3overall]
# with FF pairings [0,1] and [2,3].
_YEAR_REGION_NAMES = {
    2017: {0: "East",  1: "West",  2: "Midwest", 3: "South"},    # Villanova, Gonzaga, Kansas, UNC
    2018: {0: "South", 1: "West",  2: "East",    3: "Midwest"},   # Virginia, Xavier, Villanova, Kansas
    2019: {0: "East",  1: "West",  2: "South",   3: "Midwest"},   # Duke, Gonzaga, Virginia, UNC
    2021: {0: "West",  1: "East",  2: "South",   3: "Midwest"},   # Gonzaga, Michigan, Baylor, Illinois
    2022: {0: "West",  1: "East",  2: "South",   3: "Midwest"},   # Gonzaga, Baylor, Arizona, Kansas
    2023: {0: "South", 1: "East",  2: "Midwest", 3: "West"},      # Alabama, Purdue, Houston, Kansas
    2024: {0: "East",  1: "West",  2: "South",   3: "Midwest"},   # UConn, UNC, Houston, Purdue
    2025: {0: "South", 1: "West",  2: "Midwest", 3: "East"},      # Auburn, Florida, Houston, Duke
}

# Base URL for raw JSON
DANVK_BASE = "https://raw.githubusercontent.com/danvk/march-madness-data/master/data"


def normalize_team_name(name):
    """Normalize team name for matching (strip, collapse spaces, handle common variants)."""
    if not name or not isinstance(name, str):
        return ""
    s = " ".join(name.strip().split())
    # Remove common suffixes/variants
    s = re.sub(r"\s*\*+$", "", s)
    s = re.sub(r"\s*#+$", "", s)
    return s


def normalize_for_match(name):
    """More aggressive normalization for lookup (lowercase, remove punctuation)."""
    s = normalize_team_name(name).lower()
    s = re.sub(r"['\-\.]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Common name aliases for Torvik/KenPom matching
TEAM_ALIASES = {
    "st johns": "St. John's",
    "st johns ny": "St. John's",
    "st. johns": "St. John's",
    "st bonaventure": "St. Bonaventure",
    "st josephs": "St. Joseph's",
    "uc irvine": "UC Irvine",
    "uc davis": "UC Davis",
    "uc santa barbara": "UC Santa Barbara",
    "unlv": "UNLV",
    "lsu": "LSU",
    "uconn": "UConn",
    "usc": "USC",
    "ucla": "UCLA",
    "smu": "SMU",
    "tcu": "TCU",
    "byu": "BYU",
    "north carolina": "North Carolina",
    "nc state": "NC State",
    "miami fl": "Miami (FL)",
    "miami": "Miami (FL)",
    "florida gulf coast": "Florida Gulf Coast",
    "fgcu": "Florida Gulf Coast",
    "texas am": "Texas A&M",
    "ole miss": "Ole Miss",
    "virginia commonwealth": "VCU",
    "middle tennessee": "Middle Tennessee",
    "mtsu": "Middle Tennessee",
    "new mexico state": "New Mexico State",
    "saint marys": "Saint Mary's",
    "st marys": "Saint Mary's",
    "mount st marys": "Mount St. Mary's",
    "east tennessee state": "East Tennessee State",
    "etsu": "East Tennessee State",
    "unc wilmington": "UNC Wilmington",
    "ole miss": "Ole Miss",
    "mississippi": "Ole Miss",
    "loyola chicago": "Loyola Chicago",
    "loyola-chicago": "Loyola Chicago",
    "wichita state": "Wichita State",
    "oregon state": "Oregon State",
    "ohio state": "Ohio State",
    "michigan state": "Michigan State",
    "penn state": "Penn State",
    "iowa state": "Iowa State",
    "oklahoma state": "Oklahoma State",
    "kansas state": "Kansas State",
    "texas tech": "Texas Tech",
    "west virginia": "West Virginia",
    "south carolina": "South Carolina",
    "george mason": "George Mason",
    "george washington": "George Washington",
}


def extract_teams_from_round64(region_data):
    """Extract (team, seed) pairs from round-of-64 games in one region."""
    teams = []
    if not region_data or len(region_data) < 1:
        return teams
    round64 = region_data[0]
    for game in round64:
        for slot in game:
            if isinstance(slot, dict) and slot.get("round_of") == 64:
                teams.append((slot.get("team", ""), int(slot.get("seed", 0))))
    return teams


def load_teams_merged(data_dir, year):
    """Load teams_merged_YYYY.json or torvik_YYYY.json for stats lookup."""
    for fname in (f"teams_merged_{year}.json", f"torvik_{year}.json"):
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return {normalize_for_match(r.get("team", "")): r for r in data if r.get("team")}
            return {}
    return {}


def lookup_team_stats(team_name, merged, default=None):
    """Find team stats from merged data, trying canonical name and aliases."""
    if default is None:
        default = {"adj_o": 85, "adj_d": 112, "adj_tempo": 64, "barthag": 0.05}
    key = normalize_for_match(team_name)
    if key in merged:
        return dict(merged[key])
    alias = TEAM_ALIASES.get(key)
    if alias:
        alias_key = normalize_for_match(alias)
        if alias_key in merged:
            return dict(merged[alias_key])
    for mkey, mrow in merged.items():
        if key in mkey or mkey in key:
            return dict(mrow)
    return dict(default)


def convert_danvk_to_our_format(danvk_data, teams_merged, year=None):
    """Convert danvk JSON to our bracket format.

    danvk regions are always ordered [#1overall, #4overall, #2overall, #3overall]
    with FF pairings [0,1] and [2,3].  Our engine uses quadrant_order
    [TL, TR, BR, BL] with pairings TL-BL and TR-BR, so:
      TL=region0, BL=region1, TR=region2, BR=region3.
    """
    regions = danvk_data.get("regions", [])
    if len(regions) != 4:
        return None

    rname_map = _YEAR_REGION_NAMES.get(year, {})
    region_names = [rname_map.get(i, _DEFAULT_REGION_NAMES[i]) for i in range(4)]

    output = {"regions": {}, "final_four_matchups": [[0, 1], [2, 3]]}
    # quadrant_order: [TL, TR, BR, BL] = [region0, region2, region3, region1]
    output["quadrant_order"] = [region_names[0], region_names[2], region_names[3], region_names[1]]

    for i, region_data in enumerate(regions):
        teams = extract_teams_from_round64(region_data)
        team_list = []
        for team_name, seed in teams:
            stats = lookup_team_stats(team_name, teams_merged)
            team_list.append({
                "team": team_name.strip() or f"Unknown_{seed}",
                "seed": seed,
                "adj_o": stats.get("adj_o", 100),
                "adj_d": stats.get("adj_d", 100),
                "adj_tempo": stats.get("adj_tempo", 67.5),
                "barthag": stats.get("barthag", 0.5),
            })
        output["regions"][region_names[i]] = team_list

    return output


def fetch_danvk_bracket(year, data_dir, merged=None):
    """
    Fetch danvk bracket for year and convert to our format.
    If merged is None, attempts to load from data_dir.
    Returns (our_format_dict, True) or (None, False) on failure.
    """
    if year < 1985 or year > 2025:
        return None, False
    if requests is None:
        return None, False

    url = f"{DANVK_BASE}/{year}.json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        danvk_data = r.json()
    except Exception as e:
        print(f"  danvk fetch {year}: {e}")
        return None, False

    if merged is None:
        merged = load_teams_merged(data_dir, year)
    our_format = convert_danvk_to_our_format(danvk_data, merged, year=year)
    return our_format, our_format is not None
