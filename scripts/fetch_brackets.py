"""
Fetch historical or projected brackets.

Usage:
  python scripts/fetch_brackets.py historical 2019 2021 2023   # danvk 1985-2017, SR 2018+
  python scripts/fetch_brackets.py projected 2026              # BracketMatrix

Output: data/bracket_YYYY.json (historical) or data/bracket_YYYY_projected.json (projected)
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_historical(year):
    """Fetch historical bracket: danvk (1985-2025) or Sports-Reference (2018+ fallback)."""
    if 1985 <= year <= 2025:
        from sources.danvk_brackets import fetch_danvk_bracket
        data, ok = fetch_danvk_bracket(year, DATA_DIR)
        if ok and data:
            return data
        # danvk may not have this year; fall through to Sports-Reference for 2018+

    if 2018 <= year <= 2025:
        from sources.sports_reference import scrape_sports_reference
        return scrape_sports_reference(year, DATA_DIR)

    return None


def fetch_projected(year):
    """Fetch projected bracket from BracketMatrix."""
    from sources.bracket_matrix import scrape_bracket_matrix
    return scrape_bracket_matrix(year, DATA_DIR)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("Usage:")
        print("  python scripts/fetch_brackets.py historical 2019 2021 2023")
        print("  python scripts/fetch_brackets.py projected 2026")
        sys.exit(1)

    mode = args[0].lower()
    years = []
    for a in args[1:]:
        try:
            years.append(int(a))
        except ValueError:
            pass
    if not years and mode == "projected":
        years = [2026]
    elif not years and mode == "historical":
        years = [2023, 2024]

    for year in years:
        if mode == "historical":
            print(f"Fetching historical bracket for {year}...")
            data = fetch_historical(year)
            out_path = os.path.join(DATA_DIR, f"bracket_{year}.json")
        elif mode == "projected":
            print(f"Fetching projected bracket for {year}...")
            data = fetch_projected(year)
            out_path = os.path.join(DATA_DIR, f"bracket_{year}_projected.json")
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)

        if not data:
            print(f"  Failed to fetch {year}")
            continue

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        total = sum(len(v) for v in data.get("regions", {}).values())
        print(f"  Wrote {total} teams to {out_path}")


if __name__ == "__main__":
    # Ensure we can import from sources
    sys.path.insert(0, SCRIPT_DIR)
    main()
