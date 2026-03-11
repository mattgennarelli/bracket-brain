"""
Scrape projected bracket from BracketMatrix (aggregates 70+ bracketology sources).
Extracts team + average seed, assigns regions via S-curve to produce our format.
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

REGION_NAMES = ["South", "East", "Midwest", "West"]
BM_URL = "https://www.bracketmatrix.com/"

# S-curve region assignment: seed 1-4 go to regions in order, 5-8 in reverse, etc.
# Order: South, East, Midwest, West
def _region_for_position(idx):
    return REGION_NAMES[idx % 4]


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
    return {"adj_o": 100, "adj_d": 100, "adj_tempo": 67.5, "barthag": 0.5}


def scrape_bracket_matrix(year, data_dir):
    """
    Scrape BracketMatrix projected bracket.
    Returns our bracket format with teams assigned to regions by S-curve.
    """
    if requests is None or BeautifulSoup is None:
        print("  Install: pip install requests beautifulsoup4")
        return None

    try:
        r = requests.get(BM_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; BracketBrain/1.0)"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except requests.exceptions.SSLError:
        print("  BracketMatrix SSL error — retrying without certificate verification...")
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            r = requests.get(BM_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; BracketBrain/1.0)"}, timeout=15, verify=False)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e2:
            print(f"  BracketMatrix fetch failed (even without SSL verify): {e2}")
            return None
    except Exception as e:
        print(f"  BracketMatrix fetch failed: {e}")
        return None

    merged = load_teams_merged(data_dir, year)

    # BracketMatrix table: col0=seed, col1=team, col2=conf, col3=avg_seed, ...
    rows = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        first = cells[0].get_text().strip()
        second = cells[1].get_text().strip()
        third = cells[2].get_text().strip() if len(cells) > 2 else ""

        seed_val = None
        team_name = None
        if re.match(r"^([1-9]|1[0-6])$", first):
            seed_val = int(first)
            team_name = second or third
        elif re.match(r"^([1-9]|1[0-6])$", second):
            seed_val = int(second)
            team_name = first or third
        else:
            team_name = first or second
            for c in cells[2:5]:
                m = re.search(r"^(\d+\.?\d*)$", c.get_text().strip())
                if m:
                    seed_val = max(1, min(16, round(float(m.group(1)))))
                    break

        if not team_name or len(team_name) < 2:
            continue
        if team_name.lower() in ("team", "school", "average seed", "# of brackets"):
            continue
        if re.match(r"^[\d\.]+$", team_name):
            continue
        if seed_val is None:
            continue
        rows.append((int(seed_val), _norm(team_name)))

    if len(rows) < 32:
        return None

    # Group by seed, take first 4 per seed (top 4 teams at each seed line)
    by_seed = {}
    for seed, team in rows:
        if seed not in by_seed:
            by_seed[seed] = []
        if len(by_seed[seed]) < 4:
            by_seed[seed].append(team)

    # S-curve: 1 seeds -> South, East, Midwest, West; 2 seeds -> East, South, West, Midwest; etc.
    region_order = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    region_slots = {r: [] for r in REGION_NAMES}
    for seed in range(1, 17):
        teams = by_seed.get(seed, [])[:4]
        order = region_order[(seed - 1) % 4]
        for i, team in enumerate(teams):
            if i < len(REGION_NAMES):
                region_slots[REGION_NAMES[order[i]]].append((seed, team))

    output = {"regions": {}, "final_four_matchups": [[0, 1], [2, 3]]}
    for rn in REGION_NAMES:
        teams = region_slots[rn]
        output["regions"][rn] = []
        for seed, team in teams:
            stats = _lookup(team, merged)
            output["regions"][rn].append({
                "team": team,
                "seed": seed,
                "adj_o": stats.get("adj_o", 100),
                "adj_d": stats.get("adj_d", 100),
                "adj_tempo": stats.get("adj_tempo", 67.5),
                "barthag": stats.get("barthag", 0.5),
            })

    total = sum(len(v) for v in output["regions"].values())
    if total < 32:
        return None
    return output
