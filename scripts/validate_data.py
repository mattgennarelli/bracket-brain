"""
Validate that every team in bracket_YYYY.json has real stats (not placeholder defaults)
in teams_merged_YYYY.json.

Usage:
  python scripts/validate_data.py           # validate 2026
  python scripts/validate_data.py 2025      # validate a specific year
  python scripts/validate_data.py --all     # validate all bracket years

Exit code 0 = all teams matched with real stats.
Exit code 1 = missing or placeholder stats found.
"""
import json
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from engine import _normalize_team_for_match, _strip_mascot, load_teams_merged

# Placeholder stats assigned when a team is not found
_PLACEHOLDER = {"adj_o": 85.0, "adj_d": 112.0, "barthag": 0.05}
_PLACEHOLDER_THRESHOLD = 90.0  # adj_o below this for any tournament team is suspicious
_RICHNESS_MIN_TEAM_COUNT = 370
_RICHNESS_MIN_FIELD_COUNTS = {
    "wins": 350,
    "games": 350,
    "ppg": 350,
    "opp_ppg": 350,
    "three_pt_pct": 350,
    "experience": 350,
    "top_player": 340,
    "top_player_bpr": 340,
    "em_bpr": 340,
}
_RICHNESS_MAX_MISSING_BRACKET_TEAMS = {
    "wins": 4,
    "games": 4,
    "ppg": 4,
    "opp_ppg": 4,
    "three_pt_pct": 4,
    "experience": 4,
    "top_player": 6,
    "top_player_bpr": 6,
    "em_bpr": 6,
}


def _iter_bracket_teams(bracket):
    """Yield (region, seed, team_name) for every real team in a bracket."""
    br = bracket.get("regions", bracket)
    for region, seeds in br.items():
        if not isinstance(seeds, (dict, list)):
            continue
        items = seeds if isinstance(seeds, list) else list(seeds.values())
        for entry in items:
            entries = entry if isinstance(entry, list) else [entry]
            for t in entries:
                if not isinstance(t, dict):
                    continue
                tname = t.get("team", "")
                seed = t.get("seed", "?")
                if not tname or tname.startswith("TEAM_") or tname.startswith("Unknown"):
                    continue
                yield region, seed, tname
    for entry in bracket.get("first_four", []):
        if not isinstance(entry, dict):
            continue
        for team_key, seed_key in (("team_a", "seed_a"), ("team_b", "seed_b")):
            tname = entry.get(team_key, "")
            seed = entry.get(seed_key, "?")
            if not tname or tname.startswith("TEAM_") or tname.startswith("Unknown"):
                continue
            yield "First Four", seed, tname


def _load_merged_rows(year):
    path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _build_team_lookup(rows):
    lookup = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        team = row.get("team", "")
        if not team:
            continue
        for candidate in (team, _strip_mascot(team)):
            key = _normalize_team_for_match(candidate)
            if key:
                lookup[key] = row
    return lookup


def _lookup_row(team_lookup, name):
    keys = []
    for candidate in (name, _strip_mascot(name)):
        key = _normalize_team_for_match(candidate)
        if key and key not in keys:
            keys.append(key)

    for key in keys:
        if key in team_lookup:
            return team_lookup[key]

    for key in keys:
        for team_key, row in team_lookup.items():
            shorter, longer = (key, team_key) if len(key) <= len(team_key) else (team_key, key)
            if len(shorter) >= 6 and shorter in longer and len(shorter) >= len(longer) * 0.80:
                return row
    return None


def _richness_issues(year, bracket):
    rows = _load_merged_rows(year)
    if not rows:
        return [f"teams_merged_{year}.json not found or empty"]

    issues = []
    if year >= 2026 and len(rows) < _RICHNESS_MIN_TEAM_COUNT:
        issues.append(
            f"  TOO FEW TEAMS:  {len(rows)} rows in teams_merged_{year}.json "
            f"(expected >= {_RICHNESS_MIN_TEAM_COUNT})"
        )

    if year < 2026:
        return issues

    for field, min_count in _RICHNESS_MIN_FIELD_COUNTS.items():
        count = sum(1 for row in rows if isinstance(row, dict) and row.get(field) not in (None, "", []))
        if count < min_count:
            issues.append(
                f"  FIELD COVERAGE: {field} present for only {count} teams "
                f"(expected >= {min_count})"
            )

    lookup = _build_team_lookup(rows)
    missing_by_field = defaultdict(list)
    seen = set()
    for _, _, tname in _iter_bracket_teams(bracket):
        if tname in seen:
            continue
        seen.add(tname)
        row = _lookup_row(lookup, tname)
        if row is None:
            continue
        for field, max_missing in _RICHNESS_MAX_MISSING_BRACKET_TEAMS.items():
            if row.get(field) in (None, "", []):
                missing_by_field[field].append(tname)

    for field, teams in missing_by_field.items():
        max_missing = _RICHNESS_MAX_MISSING_BRACKET_TEAMS[field]
        if len(teams) > max_missing:
            sample = ", ".join(teams[:6])
            issues.append(
                f"  TOURNAMENT FIELD: {field} missing for {len(teams)} teams "
                f"(expected <= {max_missing}) — {sample}"
            )

    return issues


def validate_year(year):
    """Check bracket vs teams_merged for one year. Returns (ok, issues) tuple."""
    bracket_path = os.path.join(DATA_DIR, f"bracket_{year}.json")
    if not os.path.isfile(bracket_path):
        return False, [f"bracket_{year}.json not found"]

    teams_merged = load_teams_merged(DATA_DIR, year)
    if not teams_merged:
        return False, [f"teams_merged_{year}.json not found or empty"]

    with open(bracket_path) as f:
        bracket = json.load(f)

    issues = []
    matched = 0
    seen = set()
    tournament_teams = {tname for _, _, tname in _iter_bracket_teams(bracket)}

    for region, seed, tname in _iter_bracket_teams(bracket):
        if tname in seen:
            continue
        seen.add(tname)
        key = _normalize_team_for_match(tname)
        stats = teams_merged.get(key)

        if stats is None:
            # Try substring fallback
            for k, v in teams_merged.items():
                if len(key) <= len(k) and key in k and len(key) >= len(k) * 0.65:
                    stats = v
                    break

        if stats is None:
            issues.append(f"  NO MATCH:      [{seed:>2}] {tname}")
            continue

        adj_o = stats.get("adj_o")
        barthag = stats.get("barthag")

        if adj_o is not None and adj_o <= _PLACEHOLDER_THRESHOLD:
            issues.append(f"  PLACEHOLDER?:  [{seed:>2}] {tname}  adj_o={adj_o:.1f}")
        elif barthag is not None and barthag <= _PLACEHOLDER["barthag"]:
            issues.append(f"  PLACEHOLDER?:  [{seed:>2}] {tname}  barthag={barthag:.3f}")
        else:
            matched += 1

    issues.extend(_richness_issues(year, bracket))
    total = len(tournament_teams)
    return len(issues) == 0, issues, matched, total


def main():
    args = sys.argv[1:]
    all_years = "--all" in args
    years = [int(a) for a in args if a.isdigit()]

    if all_years:
        years = sorted(
            int(f.replace("bracket_", "").replace(".json", ""))
            for f in os.listdir(DATA_DIR)
            if f.startswith("bracket_") and f.endswith(".json") and "projected" not in f
        )
    elif not years:
        years = [2026]

    overall_ok = True

    for year in years:
        result = validate_year(year)
        ok, issues = result[0], result[1]
        matched, total = result[2], result[3]

        if ok:
            print(f"  {year}: {matched}/{total} teams OK")
        else:
            print(f"  {year}: {matched}/{total} matched — {len(issues)} issue(s):")
            for issue in issues:
                print(issue)
            overall_ok = False

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
