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
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

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
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)

from engine import _normalize_team_for_match, _strip_mascot, resolve_ff_pairs
from espn_scores import fetch_espn_scoreboard

DANVK_BASE = "https://raw.githubusercontent.com/danvk/march-madness-data/master/data"
SR_BASE = "https://www.sports-reference.com/cbb/postseason/men"
REGION_NAMES = ["South", "East", "Midwest", "West"]
REGION_ORDER = ["East", "Midwest", "South", "West"]
ALL_YEARS = [y for y in range(2008, 2026) if y != 2020]

ROUND_LABELS = {64: "Round of 64", 32: "Round of 32", 16: "Sweet 16", 8: "Elite 8",
                4: "Final Four", 2: "Championship"}

REGIONAL_ROUND_MAP = {0: 64, 1: 32, 2: 16, 3: 8}
REGION_R64_SEED_PAIRS = (
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
)
ET_TZ = ZoneInfo("America/New_York")


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


def _matchup_key(team_a, team_b):
    a = _normalize_team_for_match(_strip_mascot(team_a))
    b = _normalize_team_for_match(_strip_mascot(team_b))
    if not a or not b or a == b:
        return None
    return tuple(sorted((a, b)))


def _add_matchups(matchup_info, round_of, region, teams_a, teams_b):
    for team_a in teams_a or []:
        for team_b in teams_b or []:
            key = _matchup_key(team_a, team_b)
            if key:
                matchup_info[key] = {"round": round_of, "region": region}


def _load_bracket_file(year):
    path = os.path.join(DATA_DIR, f"bracket_{year}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _build_tournament_context(year):
    """Return tournament metadata needed to map dated score records to result rows."""
    raw_bracket = _load_bracket_file(year)
    if not raw_bracket:
        return None

    team_map = {}
    team_meta = {}
    regions = {}
    for region, teams in (raw_bracket.get("regions") or {}).items():
        seed_map = {}
        if isinstance(teams, list):
            entries = teams
        else:
            entries = []
            for seed, entry in (teams or {}).items():
                if isinstance(entry, dict):
                    entries.append({"seed": seed, **entry})
        for entry in entries:
            team = entry.get("team")
            try:
                seed = int(entry.get("seed"))
            except (TypeError, ValueError):
                continue
            if not team:
                continue
            regions.setdefault(region, {})[seed] = team
            norm = _normalize_team_for_match(team)
            team_map[norm] = team
            team_meta[team] = {"seed": seed, "region": region}

    first_four_slots = {}
    matchup_info = {}
    for ff in raw_bracket.get("first_four", []):
        team_a = ff.get("team_a")
        team_b = ff.get("team_b")
        region = ff.get("region")
        try:
            seed = int(ff.get("seed"))
        except (TypeError, ValueError):
            continue
        for team in (team_a, team_b):
            if not team:
                continue
            team_map[_normalize_team_for_match(team)] = team
            team_meta[team] = {"seed": seed, "region": region}
        if region and seed and team_a and team_b:
            first_four_slots[(region, seed)] = [team_a, team_b]
            key = _matchup_key(team_a, team_b)
            if key:
                matchup_info[key] = {"round": 68, "region": "First Four"}

    region_winner_sets = {}
    for region, teams_by_seed in regions.items():
        def slot_teams(seed):
            out = set(first_four_slots.get((region, seed), []))
            entry = teams_by_seed.get(seed)
            if isinstance(entry, str):
                out.add(entry)
            return out

        round_slots = []
        for seed_a, seed_b in REGION_R64_SEED_PAIRS:
            teams_a = slot_teams(seed_a)
            teams_b = slot_teams(seed_b)
            _add_matchups(matchup_info, 64, region, teams_a, teams_b)
            round_slots.append(set(teams_a) | set(teams_b))

        next_slots = []
        for idx in range(0, len(round_slots), 2):
            left = round_slots[idx]
            right = round_slots[idx + 1]
            _add_matchups(matchup_info, 32, region, left, right)
            next_slots.append(set(left) | set(right))
        round_slots = next_slots

        next_slots = []
        for idx in range(0, len(round_slots), 2):
            left = round_slots[idx]
            right = round_slots[idx + 1]
            _add_matchups(matchup_info, 16, region, left, right)
            next_slots.append(set(left) | set(right))
        round_slots = next_slots

        if len(round_slots) == 2:
            left, right = round_slots
            _add_matchups(matchup_info, 8, region, left, right)
            region_winner_sets[region] = set(left) | set(right)

    quadrant_order = raw_bracket.get("quadrant_order") or list(regions.keys())
    ff_regions = resolve_ff_pairs(quadrant_order, raw_bracket.get("final_four_matchups"))
    semifinal_winner_sets = []
    for region_a, region_b in ff_regions:
        winners_a = region_winner_sets.get(region_a, set())
        winners_b = region_winner_sets.get(region_b, set())
        _add_matchups(matchup_info, 4, "Final Four", winners_a, winners_b)
        semifinal_winner_sets.append(set(winners_a) | set(winners_b))

    if len(semifinal_winner_sets) == 2:
        _add_matchups(matchup_info, 2, "Championship", semifinal_winner_sets[0], semifinal_winner_sets[1])

    return {"team_map": team_map, "team_meta": team_meta, "matchup_info": matchup_info}


def _resolve_tournament_team(team_map, *names):
    candidates = []
    for raw_name in names:
        if not raw_name:
            continue
        for candidate in (raw_name, _strip_mascot(raw_name)):
            key = _normalize_team_for_match(candidate)
            if key:
                candidates.append(key)
    for key in candidates:
        if key in team_map:
            return team_map[key]
    for candidate in candidates:
        for key, canonical in team_map.items():
            shorter, longer = (candidate, key) if len(candidate) <= len(key) else (key, candidate)
            if shorter and shorter in longer:
                return canonical
    return None


def _cutoff_date_value(cutoff_date):
    if cutoff_date is None:
        return None
    return date.fromisoformat(str(cutoff_date))


def _nth_weekday_of_month(year, month, weekday, occurrence):
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (occurrence - 1))


def _tournament_round_windows(year):
    r64_start = _nth_weekday_of_month(year, 3, 3, 3)  # third Thursday of March
    first_four_start = r64_start - timedelta(days=2)
    return {
        68: (first_four_start, first_four_start + timedelta(days=1)),
        64: (r64_start, r64_start + timedelta(days=1)),
        32: (r64_start + timedelta(days=2), r64_start + timedelta(days=3)),
        16: (r64_start + timedelta(days=7), r64_start + timedelta(days=8)),
        8: (r64_start + timedelta(days=9), r64_start + timedelta(days=10)),
        4: (_nth_weekday_of_month(year, 4, 5, 1), _nth_weekday_of_month(year, 4, 5, 1)),
        2: (_nth_weekday_of_month(year, 4, 0, 1), _nth_weekday_of_month(year, 4, 0, 1)),
    }


def _record_date(record):
    raw = record.get("date")
    if raw:
        return date.fromisoformat(str(raw))
    scheduled_at = record.get("scheduled_at")
    if scheduled_at:
        return datetime.fromisoformat(str(scheduled_at).replace("Z", "+00:00")).astimezone(ET_TZ).date()
    return None


def _infer_tournament_round(year, record):
    record_day = _record_date(record)
    if record_day is None:
        return None
    for round_of, (start, end) in _tournament_round_windows(year).items():
        if start <= record_day <= end:
            return round_of
    return None


def _game_from_score_record(year, record, context):
    """Convert a dated score record into tournament result schema."""
    team_map = context["team_map"]
    team_meta = context["team_meta"]
    matchup_info = context["matchup_info"]

    team_a = _resolve_tournament_team(
        team_map,
        record.get("home_team", ""),
        *(record.get("home_aliases") or []),
    )
    team_b = _resolve_tournament_team(
        team_map,
        record.get("away_team", ""),
        *(record.get("away_aliases") or []),
    )
    if not team_a or not team_b:
        return None

    key = _matchup_key(team_a, team_b)
    info = matchup_info.get(key)
    if not info:
        return None
    inferred_round = _infer_tournament_round(year, record)
    if inferred_round is None or info["round"] != inferred_round:
        return None
    if info["round"] == 68:
        return None

    home_score = record.get("home_score")
    away_score = record.get("away_score")
    if home_score is None or away_score is None:
        return None

    meta_a = team_meta.get(team_a)
    meta_b = team_meta.get(team_b)
    if not meta_a or not meta_b:
        return None

    try:
        home_score = int(float(home_score))
        away_score = int(float(away_score))
    except (TypeError, ValueError):
        return None
    if home_score == away_score:
        return None

    game = _make_game(
        team_a,
        meta_a["seed"],
        home_score,
        team_b,
        meta_b["seed"],
        away_score,
        info["region"],
        info["round"],
    )
    record_day = _record_date(record)
    if record_day:
        game["date"] = record_day.isoformat()
    return game


def extract_partial_results_from_records(year, records):
    """Build partial tournament results from local score records."""
    context = _build_tournament_context(year)
    if not context:
        return None

    games_by_matchup = {}
    for record in records:
        game = _game_from_score_record(year, record, context)
        if not game:
            continue
        key = _matchup_key(game["team_a"], game["team_b"])
        if key:
            games_by_matchup[key] = game

    games = list(games_by_matchup.values())
    games.sort(key=lambda g: (g.get("round", 99), g.get("region", ""), g.get("team_a", ""), g.get("team_b", "")))
    return {"year": year, "games": games}


def _load_local_score_records(cutoff_date):
    """Load already-settled score records from local ledgers up to cutoff_date."""
    cutoff = _cutoff_date_value(cutoff_date)
    seen = {}
    for fname in ("card_ledger.json", "bets_ledger.json"):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            ledger = json.load(f)
        for pick in ledger.get("picks", []):
            if pick.get("actual_score_home") is None or pick.get("actual_score_away") is None:
                continue
            record_day = _record_date({
                "date": pick.get("date"),
                "scheduled_at": pick.get("commence_time"),
            })
            if cutoff and record_day and record_day > cutoff:
                continue
            key = (
                _normalize_team_for_match(pick.get("home_team", "")),
                _normalize_team_for_match(pick.get("away_team", "")),
                pick.get("commence_time") or "",
            )
            seen[key] = {
                "home_team": pick.get("home_team", ""),
                "away_team": pick.get("away_team", ""),
                "home_score": pick.get("actual_score_home"),
                "away_score": pick.get("actual_score_away"),
                "scheduled_at": pick.get("commence_time"),
                "date": pick.get("date"),
            }
    return list(seen.values())


def _espn_dates_for_cutoff(year, cutoff_date):
    cutoff = _cutoff_date_value(cutoff_date)
    if cutoff is None:
        return []
    r64_start = cutoff
    while r64_start.weekday() != 3:
        r64_start -= timedelta(days=1)
    first_four_start = r64_start - timedelta(days=2)
    dates = []
    current = first_four_start
    while current <= cutoff:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def _fetch_partial_score_records(year, cutoff_date):
    """Fetch completed ESPN score records through cutoff_date."""
    cutoff = _cutoff_date_value(cutoff_date)
    dates = _espn_dates_for_cutoff(year, cutoff_date)
    records = []
    for game in fetch_espn_scoreboard(dates):
        if not game.get("completed"):
            continue
        record_day = _record_date({
            "scheduled_at": game.get("scheduled_at"),
        })
        if cutoff and record_day and record_day > cutoff:
            continue
        records.append(game)
    return records


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


def scrape_sr_results(year, allow_partial=False):
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

    if len(games) < 32 and not allow_partial:
        print(f"  {year}: Only extracted {len(games)} games from SR, expected ~63")
        return None

    print(f"  {year}: Extracted {len(games)} games from Sports-Reference")
    return {"year": year, "games": games}


def extract_partial_results(year, cutoff_date=None):
    """Extract partial results from local ledgers first, then ESPN if needed."""
    records = _load_local_score_records(cutoff_date)
    seen = {
        _matchup_key(rec.get("home_team", ""), rec.get("away_team", "")): rec
        for rec in records
        if _matchup_key(rec.get("home_team", ""), rec.get("away_team", ""))
    }
    for rec in _fetch_partial_score_records(year, cutoff_date):
        key = _matchup_key(rec.get("home_team", ""), rec.get("away_team", ""))
        if key and key not in seen:
            seen[key] = rec

    result = extract_partial_results_from_records(year, list(seen.values()))
    if result and result["games"]:
        cutoff_text = cutoff_date or "today"
        print(f"  {year}: Extracted {len(result['games'])} partial games through {cutoff_text}")
        return result
    return None


# ── fetch orchestration ───────────────────────────────────────────────────────

def fetch_and_extract(year, allow_partial=False, cutoff_date=None):
    """Download results for a year, trying danvk first then Sports-Reference."""
    if allow_partial:
        partial = extract_partial_results(year, cutoff_date=cutoff_date)
        if partial and partial["games"]:
            return partial

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
    return scrape_sr_results(year, allow_partial=allow_partial)


def main():
    if requests is None and "--allow-partial" not in sys.argv:
        print("Install requests: pip install requests beautifulsoup4")
        sys.exit(1)

    years = []
    args = sys.argv[1:]
    allow_partial = "--allow-partial" in args
    cutoff_date = None
    if "--cutoff-date" in args:
        idx = args.index("--cutoff-date")
        if idx + 1 < len(args):
            cutoff_date = args[idx + 1]

    skip_next = False
    for idx, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--cutoff-date":
            skip_next = True
            continue
        if a == "--allow-partial":
            continue
        try:
            years.append(int(a))
        except ValueError:
            pass
    if not years:
        years = ALL_YEARS

    all_games = []
    for year in years:
        print(f"Extracting {year}...")
        result = fetch_and_extract(year, allow_partial=allow_partial, cutoff_date=cutoff_date)
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
