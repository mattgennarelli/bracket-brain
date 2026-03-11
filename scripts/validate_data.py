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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from engine import _normalize_team_for_match, load_teams_merged

# Placeholder stats assigned when a team is not found
_PLACEHOLDER = {"adj_o": 85.0, "adj_d": 112.0, "barthag": 0.05}
_PLACEHOLDER_THRESHOLD = 90.0  # adj_o below this for any tournament team is suspicious


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

    for region, seed, tname in _iter_bracket_teams(bracket):
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

    total = matched + len(issues)
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
