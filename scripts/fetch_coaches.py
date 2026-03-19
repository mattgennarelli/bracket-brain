"""
Fetch NCAA men's basketball head coaches into data/coaches_YYYY.csv.

Supports:
  - current coaches from Wikipedia
  - historical season coach pages from Sports-Reference

Historical Sports-Reference rows include cumulative tournament resume fields.
For pre-tournament modeling we subtract the current season's tournament result
from those cumulative counts, so the stored values reflect resume entering the
tournament rather than including that year's outcome.

Usage:
  python scripts/fetch_coaches.py
  python scripts/fetch_coaches.py 2026
  python scripts/fetch_coaches.py --source wikipedia 2026
  python scripts/fetch_coaches.py --source sports-reference --years 2008-2026
"""
import csv
import os
import re
import sys
from collections import Counter

import requests
from bs4 import BeautifulSoup, Comment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from engine import _normalize_team_for_match
from scripts.fetch_data import _find_torvik_key, load_json

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_current_NCAA_Division_I_men%27s_basketball_coaches"
SPORTS_REFERENCE_URL = "https://www.sports-reference.com/cbb/seasons/men/{year}-coaches.html"
USER_AGENT = "BracketBrain/1.0 (+local coach fetch)"

COACH_CSV_FIELDS = [
    "team",
    "coach",
    "source",
    "career_ncaa_pre",
    "career_s16_pre",
    "career_ff_pre",
    "career_titles_pre",
    "school_ncaa_pre",
    "school_s16_pre",
    "school_ff_pre",
    "school_titles_pre",
    "coach_resume_points_pre",
    "coach_tourney_score",
]

TEAM_ALIASES = {
    "California Baptist Lancers": "Cal Baptist",
    "California Baptist": "Cal Baptist",
    "LIU Sharks": "Long Island",
    "LIU": "Long Island",
    "Loyola (IL)": "Loyola Chicago",
    "Loyola Chicago Ramblers": "Loyola Chicago",
    "Miami RedHawks": "Miami OH",
    "Miami Hurricanes": "Miami FL",
    "UIC Flames": "Illinois Chicago",
    "UIC": "Illinois Chicago",
    "Kansas City Roos": "Missouri-Kansas City",
    "Kansas City": "Missouri-Kansas City",
    "UT Martin Skyhawks": "Tennessee Martin",
    "UT Martin": "Tennessee Martin",
    "UTRGV Vaqueros": "UT Rio Grande Valley",
    "UTRGV": "UT Rio Grande Valley",
}


def _clean_text(value):
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_int(value):
    text = _clean_text(value).replace("*", "")
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _clean_float(value):
    text = _clean_text(value)
    if not text:
        return None


def _since_sort_value(value):
    text = _clean_text(value)
    match = re.match(r"(\d{4})", text)
    if match:
        return int(match.group(1))
    return -1
    try:
        return float(text)
    except ValueError:
        return None


def _clean_coach_name(value):
    text = _clean_text(value)
    if not text:
        return ""
    text = text.replace("*", "").strip()
    if text.lower() == "vacant":
        return ""
    text = re.sub(r"\s*\((?:interim|acting|temporary)\)\s*$", "", text, flags=re.I)
    return text.strip()


def _has_local_year_data(year):
    return os.path.isfile(os.path.join(DATA_DIR, f"teams_merged_{year}.json")) or os.path.isfile(
        os.path.join(DATA_DIR, f"bracket_{year}.json")
    )


def _map_team_name(team_name, merged_keys, allow_suffix_stripping=True):
    alias = TEAM_ALIASES.get(_clean_text(team_name))
    lookup_name = alias or _clean_text(team_name)
    if lookup_name in merged_keys:
        return lookup_name
    normalized_index = {}
    for key in merged_keys:
        normalized = _normalize_team_for_match(key)
        if normalized and normalized not in normalized_index:
            normalized_index[normalized] = key
    normalized_lookup = _normalize_team_for_match(lookup_name)
    if normalized_lookup in normalized_index:
        return normalized_index[normalized_lookup]
    if allow_suffix_stripping:
        tokens = _clean_text(team_name).split()
        for n in range(len(tokens) - 1, 0, -1):
            candidate = " ".join(tokens[:n])
            alias = TEAM_ALIASES.get(candidate)
            strict_candidate = alias or candidate
            if strict_candidate in merged_keys:
                return strict_candidate
            normalized_candidate = _normalize_team_for_match(strict_candidate)
            if normalized_candidate in normalized_index:
                return normalized_index[normalized_candidate]
        mapped = _find_torvik_key(lookup_name, merged_keys)
        if mapped:
            return mapped
    return None


def _table_soups(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    yield soup
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment_text = str(comment)
        if "<table" not in comment_text:
            continue
        try:
            yield BeautifulSoup(comment_text, "html.parser")
        except Exception:
            continue


def _header_texts(table):
    thead = table.find("thead")
    if thead:
        header_rows = thead.find_all("tr")
        if not header_rows:
            return []
        cells = header_rows[-1].find_all(["th", "td"])
    else:
        header_row = table.find("tr")
        if not header_row:
            return []
        cells = header_row.find_all(["th", "td"])
    return [_clean_text(cell.get_text(" ", strip=True)).lower() for cell in cells]


def _find_wikipedia_table(html_text):
    for soup in _table_soups(html_text):
        for table in soup.find_all("table"):
            headers = _header_texts(table)
            if "team" in headers and "current coach" in headers:
                return table
    raise ValueError("Could not find Wikipedia coaches table")


def _extract_wikipedia_rows(html_text):
    table = _find_wikipedia_table(html_text)
    headers = _header_texts(table)
    team_idx = headers.index("team")
    coach_idx = headers.index("current coach")
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) <= max(team_idx, coach_idx):
            continue
        rows.append(
            {
                "team": _clean_text(cells[team_idx].get_text(" ", strip=True)),
                "coach": _clean_text(cells[coach_idx].get_text(" ", strip=True)),
                "source": "wikipedia_current_coaches",
            }
        )
    return rows


def _find_sports_reference_table(html_text):
    for soup in _table_soups(html_text):
        for table in soup.find_all("table"):
            headers = _header_texts(table)
            if "coach" in headers and "school" in headers and "ncaa tournament" in headers:
                return table
    raise ValueError("Could not find Sports-Reference coaches table")


def _tourney_result_flags(result_text):
    text = _clean_text(result_text).lower()
    flags = {"ncaa": 0, "s16": 0, "ff": 0, "titles": 0}
    if not text:
        return flags
    flags["ncaa"] = 1
    if any(term in text for term in ("regional semifinal", "regional final", "national semifinal", "national final", "national champion")):
        flags["s16"] = 1
    if any(term in text for term in ("national semifinal", "national final", "national champion")):
        flags["ff"] = 1
    if "national champion" in text:
        flags["titles"] = 1
    return flags


def _percentile_rank(values, target):
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.5
    less = sum(1 for value in values if value < target)
    equal = sum(1 for value in values if value == target)
    return round((less + 0.5 * max(0, equal - 1)) / (len(values) - 1), 4)


def _sports_reference_resume_row(cells, year):
    values = [_clean_text(cell.get_text(" ", strip=True)) for cell in cells]
    if not values or values[0].lower() == "coach":
        return None
    by_stat = {cell.get("data-stat"): _clean_text(cell.get_text(" ", strip=True)) for cell in cells if cell.get("data-stat")}

    if "coach" in by_stat and "school" in by_stat:
        coach = _clean_coach_name(by_stat.get("coach"))
        school = _clean_text(by_stat.get("school"))
        result_text = by_stat.get("ncaa_seas", "")
        since_raw = by_stat.get("since", "")
        wins_seas_raw = by_stat.get("wins_seas", "")
        losses_seas_raw = by_stat.get("losses_seas", "")
        school_ncaa_raw = by_stat.get("ncaa_cur", "")
        school_s16_raw = by_stat.get("sw16_cur", "")
        school_ff_raw = by_stat.get("ff_cur", "")
        school_titles_raw = by_stat.get("champ_cur", "")
        career_ncaa_raw = by_stat.get("ncaa_car", "")
        career_s16_raw = by_stat.get("sw16_car", "")
        career_ff_raw = by_stat.get("ff_car", "")
        career_titles_raw = by_stat.get("champ_car", "")
    else:
        if len(values) < 27:
            return None
        coach = _clean_coach_name(values[0])
        school = _clean_text(values[1])
        result_text = values[9]
        since_raw = values[11]
        wins_seas_raw = values[4]
        losses_seas_raw = values[5]
        school_ncaa_raw = values[15]
        school_s16_raw = values[16]
        school_ff_raw = values[17]
        school_titles_raw = values[18]
        career_ncaa_raw = values[23]
        career_s16_raw = values[24]
        career_ff_raw = values[25]
        career_titles_raw = values[26]

    if not coach or not school:
        return None

    result_flags = _tourney_result_flags(result_text)
    school_ncaa = max(0, _clean_int(school_ncaa_raw) - result_flags["ncaa"])
    school_s16 = max(0, _clean_int(school_s16_raw) - result_flags["s16"])
    school_ff = max(0, _clean_int(school_ff_raw) - result_flags["ff"])
    school_titles = max(0, _clean_int(school_titles_raw) - result_flags["titles"])
    career_ncaa = max(0, _clean_int(career_ncaa_raw) - result_flags["ncaa"])
    career_s16 = max(0, _clean_int(career_s16_raw) - result_flags["s16"])
    career_ff = max(0, _clean_int(career_ff_raw) - result_flags["ff"])
    career_titles = max(0, _clean_int(career_titles_raw) - result_flags["titles"])
    resume_points = career_ncaa + 2 * career_s16 + 4 * career_ff + 8 * career_titles

    return {
        "team": school,
        "coach": coach,
        "source": f"sports_reference_{year}_coaches",
        "career_ncaa_pre": career_ncaa,
        "career_s16_pre": career_s16,
        "career_ff_pre": career_ff,
        "career_titles_pre": career_titles,
        "school_ncaa_pre": school_ncaa,
        "school_s16_pre": school_s16,
        "school_ff_pre": school_ff,
        "school_titles_pre": school_titles,
        "coach_resume_points_pre": resume_points,
        "_since_sort": _since_sort_value(since_raw),
        "_season_games": _clean_int(wins_seas_raw) + _clean_int(losses_seas_raw),
    }


def _extract_sports_reference_rows(html_text, year):
    table = _find_sports_reference_table(html_text)
    tbody = table.find("tbody") or table
    rows = []
    for tr in tbody.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        record = _sports_reference_resume_row(tr.find_all(["th", "td"]), year)
        if record:
            rows.append(record)
    scores = [row["coach_resume_points_pre"] for row in rows]
    for row in rows:
        row["coach_tourney_score"] = _percentile_rank(scores, row["coach_resume_points_pre"])
    return rows


def _rows_to_records(rows, teams_merged=None):
    merged_keys = set(teams_merged or [])
    out = {}
    matched = set()
    for row in rows:
        team = _clean_text(row.get("team"))
        coach = _clean_coach_name(row.get("coach"))
        if not team or not coach:
            continue
        allow_suffix_stripping = str(row.get("source", "")).startswith("wikipedia_")
        mapped_team = _map_team_name(team, merged_keys, allow_suffix_stripping=allow_suffix_stripping) if merged_keys else None
        record = dict(row)
        record["team"] = mapped_team or team
        record["coach"] = coach
        for key in COACH_CSV_FIELDS:
            if key not in record:
                record[key] = ""
        key = record["team"]
        existing = out.get(key)
        if existing is None:
            out[key] = record
        else:
            sort_key = (record.get("_since_sort", -1), record.get("_season_games", -1), record.get("coach_resume_points_pre", -1))
            existing_key = (
                existing.get("_since_sort", -1),
                existing.get("_season_games", -1),
                existing.get("coach_resume_points_pre", -1),
            )
            if sort_key >= existing_key:
                out[key] = record
        if mapped_team:
            matched.add(mapped_team)
    cleaned = []
    for record in out.values():
        cleaned.append({field: record.get(field, "") for field in COACH_CSV_FIELDS})
    return cleaned, len(matched)


def fetch_wikipedia_rows(url=WIKIPEDIA_URL):
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return _extract_wikipedia_rows(response.text)


def fetch_sports_reference_rows(year):
    url = SPORTS_REFERENCE_URL.format(year=year)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return _extract_sports_reference_rows(response.text, year)


def load_merged_team_names(year):
    path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
    data = load_json(path)
    if not data:
        return []
    rows = data.values() if isinstance(data, dict) else data
    return [row.get("team") for row in rows if row.get("team")]


def load_bracket_team_names(year):
    path = os.path.join(DATA_DIR, f"bracket_{year}.json")
    data = load_json(path)
    if not isinstance(data, dict):
        return []
    teams = []
    for region_entries in data.get("regions", {}).values():
        for entry in region_entries if isinstance(region_entries, list) else []:
            if isinstance(entry, dict) and entry.get("team"):
                teams.append(entry["team"])
    return teams


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COACH_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def find_duplicate_teams(rows):
    counts = Counter(row["team"] for row in rows)
    return {team: count for team, count in counts.items() if count > 1}


def _parse_years(year_args):
    years = []
    for arg in year_args:
        if "," in arg:
            years.extend(_parse_years([part.strip() for part in arg.split(",") if part.strip()]))
            continue
        if "-" in arg:
            start_text, end_text = arg.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                continue
            step = 1 if end >= start else -1
            years.extend(list(range(start, end + step, step)))
            continue
        try:
            years.append(int(arg))
        except ValueError:
            continue
    deduped = []
    seen = set()
    for year in years:
        if year in seen:
            continue
        seen.add(year)
        deduped.append(year)
    return deduped


def _save_year(source, year):
    teams_merged = load_merged_team_names(year)
    if source == "wikipedia":
        source_rows = fetch_wikipedia_rows()
    elif source == "sports-reference":
        source_rows = fetch_sports_reference_rows(year)
    else:
        raise ValueError(f"Unsupported source: {source}")

    rows, matched = _rows_to_records(source_rows, teams_merged=teams_merged)
    out_path = os.path.join(DATA_DIR, f"coaches_{year}.csv")
    write_csv(out_path, rows)
    print(f"Wrote {len(rows)} coaches to {out_path}")
    if teams_merged:
        print(f"Matched {matched}/{len(rows)} rows directly to teams_merged_{year}.json")
    duplicates = find_duplicate_teams(rows)
    if duplicates:
        print(f"WARNING: duplicate mapped teams detected: {duplicates}")
    bracket_teams = load_bracket_team_names(year)
    if bracket_teams:
        coach_map = {row["team"]: row for row in rows}
        covered = sum(1 for team in bracket_teams if team in coach_map)
        print(f"Bracket coverage: {covered}/{len(bracket_teams)}")


def main():
    source = "wikipedia"
    year_args = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--source" and i + 1 < len(args):
            source = args[i + 1].strip().lower()
            i += 2
            continue
        if arg == "--years" and i + 1 < len(args):
            year_args.extend([args[i + 1]])
            i += 2
            continue
        year_args.append(arg)
        i += 1

    years = _parse_years(year_args) or [2026]
    if len(years) > 1:
        years = [year for year in years if _has_local_year_data(year)]

    for year in years:
        print(f"\n=== {year} ({source}) ===")
        _save_year(source, year)
    return 0


if __name__ == "__main__":
    sys.exit(main())
