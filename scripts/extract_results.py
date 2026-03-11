"""
Extract full tournament results from danvk/march-madness-data or Sports-Reference.

Downloads JSON/HTML for each year and extracts every game:
  team_a, seed_a, score_a, team_b, seed_b, score_b, winner, margin, upset, round, region

Output:
  data/results_YYYY.json  — per-year
  data/results_all.json   — combined

Usage:
  python scripts/extract_results.py            # all years
  python scripts/extract_results.py 2025       # single year
"""
import json
import os
import re
import sys

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DANVK_BASE = "https://raw.githubusercontent.com/danvk/march-madness-data/master/data"
SR_BASE = "https://www.sports-reference.com/cbb/postseason/men"
REGION_NAMES = ["South", "East", "Midwest", "West"]
REGION_ORDER = ["East", "Midwest", "South", "West"]
ALL_YEARS = [y for y in range(2010, 2026) if y != 2020]

ROUND_LABELS = {64: "Round of 64", 32: "Round of 32", 16: "Sweet 16", 8: "Elite 8",
                4: "Final Four", 2: "Championship"}

REGIONAL_ROUND_MAP = {0: 64, 1: 32, 2: 16, 3: 8}


def _make_game(team_a, seed_a, score_a, team_b, seed_b, score_b, region, round_of):
    """Build a standardized game dict."""
    winner = team_a if score_a > score_b else team_b
    margin = abs(score_a - score_b)
    upset = (score_a > score_b and seed_a > seed_b) or (score_b > score_a and seed_b > seed_a)
    return {
        "round": round_of,
        "round_name": ROUND_LABELS.get(round_of, f"Round of {round_of}"),
        "region": region,
        "team_a": team_a, "seed_a": seed_a, "score_a": score_a,
        "team_b": team_b, "seed_b": seed_b, "score_b": score_b,
        "winner": winner,
        "margin": margin,
        "upset": upset,
    }


# ── danvk extraction ──────────────────────────────────────────────────────────

def extract_year_danvk(danvk_data, year):
    """Extract all games from a danvk year JSON."""
    games = []
    regions = danvk_data.get("regions", [])

    for ri, region_data in enumerate(regions):
        region_name = REGION_NAMES[ri] if ri < len(REGION_NAMES) else f"Region{ri+1}"
        for round_idx, round_games in enumerate(region_data):
            round_of = REGIONAL_ROUND_MAP.get(round_idx, 64 // (2 ** round_idx))
            for game in round_games:
                if len(game) == 2 and isinstance(game[0], dict) and isinstance(game[1], dict):
                    a, b = game[0], game[1]
                    games.append(_make_game(
                        a.get("team", ""), int(a.get("seed", 0)), int(a.get("score", 0)),
                        b.get("team", ""), int(b.get("seed", 0)), int(b.get("score", 0)),
                        region_name, round_of))

    ff_data = danvk_data.get("finalfour", [])
    if ff_data and len(ff_data) >= 1:
        semis = ff_data[0] if len(ff_data) > 0 else []
        for game in semis:
            if len(game) == 2 and isinstance(game[0], dict):
                a, b = game[0], game[1]
                games.append(_make_game(
                    a.get("team", ""), int(a.get("seed", 0)), int(a.get("score", 0)),
                    b.get("team", ""), int(b.get("seed", 0)), int(b.get("score", 0)),
                    "Final Four", 4))
        if len(ff_data) > 1:
            for game in ff_data[1]:
                if len(game) == 2 and isinstance(game[0], dict):
                    a, b = game[0], game[1]
                    games.append(_make_game(
                        a.get("team", ""), int(a.get("seed", 0)), int(a.get("score", 0)),
                        b.get("team", ""), int(b.get("seed", 0)), int(b.get("score", 0)),
                        "Championship", 2))

    return {"year": year, "games": games}


# ── Sports-Reference extraction ───────────────────────────────────────────────

def _sr_parse_entries(round_div):
    """Parse all (seed, team, score) entries from a single round div."""
    entries = []
    for div in round_div.find_all("div", recursive=True):
        if div.find_all("div"):
            continue
        text = div.get_text(separator="|").strip()
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if len(parts) >= 2 and re.match(r"^(1[0-6]|[1-9])$", parts[0]):
            seed = int(parts[0])
            team = parts[1]
            score = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
            if len(team) > 1 and not re.match(r"^\d+$", team):
                entries.append({"seed": seed, "team": team, "score": score})
    return entries


def _sr_detect_region(bracket_div):
    """Determine region name from the preceding sibling."""
    prev = bracket_div.find_previous_sibling()
    if prev:
        text = prev.get_text()
        for rn in REGION_ORDER:
            if rn in text:
                return rn
    return None


def scrape_sr_results(year):
    """Scrape Sports-Reference for full tournament results."""
    if requests is None or BeautifulSoup is None:
        print("  Install: pip install requests beautifulsoup4")
        return None

    url = f"{SR_BASE}/{year}-ncaa.html"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"  {year}: Sports-Reference fetch failed — {e}")
        return None

    brackets_div = soup.find("div", id="brackets")
    if not brackets_div:
        print(f"  {year}: No #brackets container found")
        return None

    all_bracket_divs = brackets_div.find_all("div", id="bracket")
    region_divs = [b for b in all_bracket_divs if "team16" in (b.get("class") or [])]
    ff_divs = [b for b in all_bracket_divs if "team4" in (b.get("class") or [])]

    if len(region_divs) != 4:
        print(f"  {year}: Expected 4 regional brackets, found {len(region_divs)}")
        return None

    games = []

    # Detect region names
    assigned = set()
    region_map = []
    for div in region_divs:
        name = _sr_detect_region(div)
        region_map.append(name)
        if name:
            assigned.add(name)
    remaining = [r for r in REGION_ORDER if r not in assigned]
    for i, name in enumerate(region_map):
        if name is None and remaining:
            region_map[i] = remaining.pop(0)
        elif name is None:
            region_map[i] = f"Region{i+1}"

    for div, region_name in zip(region_divs, region_map):
        rounds = div.find_all("div", class_="round")
        for ri, rd in enumerate(rounds):
            round_of = REGIONAL_ROUND_MAP.get(ri)
            if round_of is None:
                continue
            entries = _sr_parse_entries(rd)
            # Deduplicate entries (keep first occurrence of each team)
            seen = set()
            deduped = []
            for e in entries:
                if e["team"] not in seen:
                    seen.add(e["team"])
                    deduped.append(e)
            # Pair consecutive entries into games
            for j in range(0, len(deduped) - 1, 2):
                a, b = deduped[j], deduped[j + 1]
                if a["score"] is not None and b["score"] is not None:
                    games.append(_make_game(
                        a["team"], a["seed"], a["score"],
                        b["team"], b["seed"], b["score"],
                        region_name, round_of))

    # Final Four
    if ff_divs:
        ff_div = ff_divs[0]
        ff_rounds = ff_div.find_all("div", class_="round")
        ff_round_map = {0: 4, 1: 2}
        ff_region_map = {0: "Final Four", 1: "Championship"}
        for ri, rd in enumerate(ff_rounds):
            round_of = ff_round_map.get(ri)
            if round_of is None:
                continue
            entries = _sr_parse_entries(rd)
            seen = set()
            deduped = []
            for e in entries:
                if e["team"] not in seen:
                    seen.add(e["team"])
                    deduped.append(e)
            for j in range(0, len(deduped) - 1, 2):
                a, b = deduped[j], deduped[j + 1]
                if a["score"] is not None and b["score"] is not None:
                    games.append(_make_game(
                        a["team"], a["seed"], a["score"],
                        b["team"], b["seed"], b["score"],
                        ff_region_map[ri], round_of))

    if len(games) < 32:
        print(f"  {year}: Only extracted {len(games)} games from SR, expected ~63")
        return None

    print(f"  {year}: Extracted {len(games)} games from Sports-Reference")
    return {"year": year, "games": games}


# ── fetch orchestration ───────────────────────────────────────────────────────

def fetch_and_extract(year):
    """Download results for a year, trying danvk first then Sports-Reference."""
    url = f"{DANVK_BASE}/{year}.json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        result = extract_year_danvk(data, year)
        if result and result["games"]:
            return result
    except Exception:
        pass

    # Fallback to Sports-Reference
    return scrape_sr_results(year)


def main():
    if requests is None:
        print("Install requests: pip install requests beautifulsoup4")
        sys.exit(1)

    years = []
    for a in sys.argv[1:]:
        try:
            years.append(int(a))
        except ValueError:
            pass
    if not years:
        years = ALL_YEARS

    all_games = []
    for year in years:
        print(f"Extracting {year}...")
        result = fetch_and_extract(year)
        if not result or not result["games"]:
            print(f"  {year}: No games extracted")
            continue

        out_path = os.path.join(DATA_DIR, f"results_{year}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        n = len(result["games"])
        upsets = sum(1 for g in result["games"] if g["upset"])
        rounds = {}
        for g in result["games"]:
            rounds[g["round_name"]] = rounds.get(g["round_name"], 0) + 1
        round_summary = ", ".join(f"{v} {k}" for k, v in sorted(rounds.items(), key=lambda x: -x[1]))
        print(f"  {n} games ({upsets} upsets): {round_summary}")
        print(f"  Wrote {out_path}")

        all_games.extend(result["games"])

    # Rebuild combined results file when extracting multiple years
    if len(years) > 1:
        # Also include any existing years not in the current run
        existing_all_path = os.path.join(DATA_DIR, "results_all.json")
        if os.path.isfile(existing_all_path):
            with open(existing_all_path) as f:
                existing = json.load(f)
            existing_years = set()
            for g in existing.get("games", []):
                # Games don't have a year field; we need to track by file
                pass
        # Simpler: just combine all results_YYYY.json files in data dir
        all_games = []
        all_year_list = []
        for fname in sorted(os.listdir(DATA_DIR)):
            m = re.match(r"results_(\d{4})\.json$", fname)
            if m:
                yr = int(m.group(1))
                with open(os.path.join(DATA_DIR, fname)) as f:
                    data = json.load(f)
                yr_games = data.get("games", [])
                all_games.extend(yr_games)
                all_year_list.append(yr)

        all_path = os.path.join(DATA_DIR, "results_all.json")
        combined = {"years": all_year_list, "total_games": len(all_games), "games": all_games}
        with open(all_path, "w") as f:
            json.dump(combined, f, indent=2)
        total_upsets = sum(1 for g in all_games if g["upset"])
        print(f"\nCombined: {len(all_games)} games across {len(all_year_list)} years ({total_upsets} upsets)")
        print(f"Wrote {all_path}")


if __name__ == "__main__":
    main()
