"""
Fetch or create conference tournament results for a given season.

Output: data/conf_tourney_YYYY.json
  {"year": YYYY, "teams": {"TeamName": "champion"|"finalist"|"semifinal"|"quarterfinal"|"early"}}

Usage:
  python scripts/fetch_conf_tourney.py --scrape 2026
  python scripts/fetch_conf_tourney.py --from-csv data/conf_tourney_2026.csv 2026
  python scripts/fetch_conf_tourney.py --manual 2026
"""
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(SCRIPT_DIR, "sources"))
os.makedirs(DATA_DIR, exist_ok=True)

VALID_RESULTS = {"champion", "finalist", "semifinal", "quarterfinal", "early"}


def normalize_team(name):
    if not name or not isinstance(name, str):
        return ""
    return " ".join(name.strip().split())


def load_from_csv(path, year):
    """Load team,result from CSV. Returns dict team -> result."""
    teams = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = normalize_team(row.get("team", row.get("Team", "")))
            result = (row.get("result", row.get("Result", "")) or "").strip().lower()
            if not team:
                continue
            if result and result in VALID_RESULTS:
                teams[team] = result
            else:
                teams[team] = "early"  # default if missing
    return teams


def write_template(year):
    """Write empty template for manual fill."""
    path = os.path.join(DATA_DIR, f"conf_tourney_{year}.json")
    template = {
        "year": year,
        "teams": {},
        "_comment": "Add team: result. Results: champion, finalist, semifinal, quarterfinal, early",
    }
    with open(path, "w") as f:
        json.dump(template, f, indent=2)
    print(f"Wrote template -> {path}")
    print("  Add teams with results: champion | finalist | semifinal | quarterfinal | early")


def main():
    args = sys.argv[1:]
    year = None
    csv_path = None
    manual = "--manual" in args
    scrape = "--scrape" in args

    i = 0
    while i < len(args):
        if args[i] == "--from-csv" and i + 1 < len(args):
            csv_path = args[i + 1]
            i += 2
            continue
        if args[i] == "--manual":
            i += 1
            continue
        if args[i] == "--scrape":
            i += 1
            continue
        try:
            year = int(args[i])
            i += 1
            break
        except ValueError:
            i += 1

    if scrape and year:
        from sports_reference import scrape_conf_tourney_results

        teams = scrape_conf_tourney_results(year)
        if teams is None:
            return 1
        out = {"year": year, "teams": teams}
        path = os.path.join(DATA_DIR, f"conf_tourney_{year}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {len(teams)} teams -> {path}")
        return 0

    if manual and year:
        write_template(year)
        return 0

    if csv_path and year:
        if not os.path.isfile(csv_path):
            print(f"ERROR: CSV not found: {csv_path}")
            return 1
        teams = load_from_csv(csv_path, year)
        out = {"year": year, "teams": teams}
        path = os.path.join(DATA_DIR, f"conf_tourney_{year}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {len(teams)} teams -> {path}")
        return 0

    print("Usage:")
    print("  python scripts/fetch_conf_tourney.py --scrape YEAR")
    print("  python scripts/fetch_conf_tourney.py --from-csv PATH YEAR")
    print("  python scripts/fetch_conf_tourney.py --manual YEAR")
    return 1


if __name__ == "__main__":
    sys.exit(main())
