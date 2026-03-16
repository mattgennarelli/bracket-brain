"""
Scrape historical brackets from Sports-Reference (2018-2025).
Parses the #brackets container which holds 4 regional bracket divs
(class=team16) plus a Final Four div (class=team4).
"""
import json
import os
import re

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

REGION_ORDER = ["East", "Midwest", "South", "West"]
SR_BASE = "https://www.sports-reference.com/cbb/postseason/men"


def _norm(s):
    return " ".join((s or "").strip().split())


def load_teams_merged(data_dir, year):
    path = os.path.join(data_dir, f"teams_merged_{year}.json")
    if not os.path.isfile(path):
        path = os.path.join(data_dir, f"torvik_{year}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {_norm(r.get("team", "")).lower(): r for r in data if r.get("team")}
    return {}


def _lookup(team_name, merged):
    key = _norm(team_name).lower()
    if key in merged:
        return dict(merged[key])
    for k, v in merged.items():
        if key in k or k in key:
            return dict(v)
    return {"adj_o": 85, "adj_d": 112, "adj_tempo": 64, "barthag": 0.05}


def _detect_region(bracket_div):
    """Determine region name from the preceding sibling of a bracket div."""
    prev = bracket_div.find_previous_sibling()
    if prev:
        text = prev.get_text()
        for rn in REGION_ORDER:
            if rn in text:
                return rn
    return None


def _parse_r64_teams(bracket_div):
    """Extract R64 seed+team pairs from a regional bracket div."""
    rounds = bracket_div.find_all("div", class_="round")
    if not rounds:
        return []

    r0 = rounds[0]
    teams = []
    for div in r0.find_all("div", recursive=True):
        if div.find_all("div"):
            continue
        text = div.get_text(separator="|").strip()
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if len(parts) >= 2 and re.match(r"^(1[0-6]|[1-9])$", parts[0]):
            seed = int(parts[0])
            team = parts[1]
            if len(team) > 1 and not re.match(r"^\d+$", team):
                teams.append((seed, team))

    seen = set()
    unique = []
    for s, t in teams:
        if t not in seen:
            seen.add(t)
            unique.append((s, t))
    return unique


def scrape_sports_reference(year, data_dir):
    """
    Scrape Sports-Reference tournament page for year.
    Returns our bracket format or None.
    """
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
        print(f"  Sports-Reference fetch failed: {e}")
        return None

    brackets_div = soup.find("div", id="brackets")
    if not brackets_div:
        print("  Could not find #brackets container on page")
        return None

    all_bracket_divs = brackets_div.find_all("div", id="bracket")
    region_divs = [b for b in all_bracket_divs if "team16" in (b.get("class") or [])]

    if len(region_divs) != 4:
        print(f"  Expected 4 regional brackets, found {len(region_divs)}")
        return None

    merged = load_teams_merged(data_dir, year)
    output = {"regions": {}, "final_four_matchups": [[0, 1], [2, 3]]}

    assigned_regions = set()
    region_data = []

    for div in region_divs:
        region_name = _detect_region(div)
        teams = _parse_r64_teams(div)
        region_data.append((region_name, teams))
        if region_name:
            assigned_regions.add(region_name)

    # Assign unknown regions by elimination
    remaining = [r for r in REGION_ORDER if r not in assigned_regions]
    for i, (region_name, teams) in enumerate(region_data):
        if region_name is None:
            if remaining:
                region_name = remaining.pop(0)
            else:
                region_name = f"Region{i + 1}"
            region_data[i] = (region_name, teams)

    for region_name, teams in region_data:
        team_list = []
        for seed, team in teams:
            stats = _lookup(team, merged)
            team_list.append({
                "team": team,
                "seed": seed,
                "adj_o": stats.get("adj_o", 100),
                "adj_d": stats.get("adj_d", 100),
                "adj_tempo": stats.get("adj_tempo", 67.5),
                "barthag": stats.get("barthag", 0.5),
            })
        output["regions"][region_name] = team_list

    total = sum(len(v) for v in output["regions"].values())
    if total < 32:
        print(f"  Only found {total} teams, need at least 32")
        return None

    print(f"  Scraped {total} teams from Sports-Reference ({year})")
    return output


SR_CONF_BASE = "https://www.sports-reference.com/cbb/seasons/men"


def scrape_conf_tourney_results(year):
    """
    Scrape conference tournament champions and runners-up from Sports-Reference.
    Returns dict team_name -> "champion" | "finalist".
    Main page has Champ and Runner-up columns; semifinalists would require per-conference pages.
    """
    if requests is None or BeautifulSoup is None:
        print("  Install: pip install requests beautifulsoup4")
        return None

    url = f"{SR_CONF_BASE}/{year}-conference-tournaments.html"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"  Sports-Reference conf tourney fetch failed: {e}")
        return None

    # Find ALL Conference Tournament Champions tables (SR may split into multiple tables)
    tables = []
    for t in soup.find_all("table"):
        tid = (t.get("id") or "").lower()
        if "conference" in tid and "tournament" in tid:
            tables.append(t)
    if not tables:
        for t in soup.find_all("table"):
            if t.find("th", string=lambda s: s and "Champ" in str(s)) or t.find("td", string=lambda s: s and "Champ" in str(s)):
                tables.append(t)
                break
    if not tables:
        # Fallback: look in HTML comments (SR often hides full table there)
        from bs4 import Comment
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "conference" in str(comment) and "tournament" in str(comment):
                sub = BeautifulSoup(str(comment), "html.parser")
                for t in sub.find_all("table"):
                    if t.find("th", string=lambda s: s and "Champ" in str(s)):
                        tables.append(t)
                        break
                if tables:
                    break
    if not tables:
        print("  Could not find conference tournament table")
        return None

    teams = {}
    for table in tables:
        tbody = table.find("tbody") or table
        rows = tbody.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            # Columns: Conference (th), Dates, Champ, Runner-up -> 3 td: [Dates, Champ, Runner-up]
            if len(cells) < 3:
                continue
            champ_cell = cells[1] if len(cells) > 1 else None
            runner_cell = cells[2] if len(cells) > 2 else None
            if champ_cell:
                a = champ_cell.find("a")
                if a:
                    name = _norm(a.get_text())
                    if name:
                        teams[name] = "champion"
            if runner_cell:
                a = runner_cell.find("a")
                if a:
                    name = _norm(a.get_text())
                    if name and name not in teams:
                        teams[name] = "finalist"

    print(f"  Scraped {len(teams)} conf tourney teams ({year}): {sum(1 for v in teams.values() if v == 'champion')} champs, {sum(1 for v in teams.values() if v == 'finalist')} finalists")
    return teams
