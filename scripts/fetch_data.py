"""
Fetch all data sources and merge into a single teams file for the bracket.
  - Torvik: required (run fetch_torvik.py)
  - KenPom: optional (run fetch_kenpom.py if you have a subscription)
  - Evan Miya: optional (place data/evanmiya_YYYY.csv or run fetch_evanmiya.py)
Output: data/teams_merged_YYYY.json — one record per team with adj_o, adj_d, adj_tempo, barthag, luck (if any), star_score (if any).
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from engine import _normalize_team_for_match


def normalize_team(name):
    if not name or not isinstance(name, str):
        return ""
    return " ".join(name.strip().split())


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_torvik(year):
    path = os.path.join(DATA_DIR, f"torvik_{year}.json")
    data = load_json(path)
    if not data:
        return {}
    return {normalize_team(r["team"]): r for r in data if r.get("team")}


def load_kenpom(year):
    path = os.path.join(DATA_DIR, f"kenpom_{year}.csv")
    if not os.path.isfile(path):
        return {}
    import csv
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            team = normalize_team(row.get("team", row.get("Team", "")))
            if not team:
                continue
            try:
                luck = float(row.get("luck", 0) or 0)
            except (TypeError, ValueError):
                luck = 0
            # KenPom efficiency/tempo (if present) kept separate from Torvik
            def _num(key, default):
                try:
                    return float(row.get(key, default) or default)
                except (TypeError, ValueError):
                    return default
            kp_adj_o = _num("kp_adj_o", 0.0)
            kp_adj_d = _num("kp_adj_d", 0.0)
            kp_adj_tempo = _num("kp_adj_tempo", 0.0)
            out[team] = {
                "luck": luck,
                "kp_adj_o": kp_adj_o,
                "kp_adj_d": kp_adj_d,
                "kp_adj_tempo": kp_adj_tempo,
            }
    return out


def load_evanmiya(year):
    path = os.path.join(DATA_DIR, f"evanmiya_{year}.json")
    data = load_json(path)
    if not data:
        return {}
    _EM_FIELDS = (
        # Team BPR ratings
        "em_obpr", "em_dbpr", "em_bpr",
        # Adjustment signals
        "em_opponent_adjust", "em_pace_adjust",
        # Rankings
        "em_off_rank", "em_def_rank", "em_tempo_rank", "em_home_rank",
        # Tempo
        "em_tempo",
        # Scoring-burst / runs
        "em_runs_per_game", "em_runs_conceded", "em_runs_margin",
        # Player depth
        "top_player", "top_player_bpr",
        "em_top5_bpr", "em_star_concentration", "em_poss_weighted_bpr", "em_depth_score",
        # EvanMiya efficiency on Torvik scale (from player CSV context)
        "em_adj_o", "em_adj_d",
        # Legacy fields (kept for backwards compat)
        "em_o_rate", "em_d_rate", "em_rel_rating", "em_roster_rank",
        # Conf rating (if present)
        "conf_rating",
    )
    out = {}
    for r in data:
        team = normalize_team(r.get("team", ""))
        if not team:
            continue
        entry = {"star_score": r.get("star_score", 0.5)}
        for k in _EM_FIELDS:
            if r.get(k) is not None:
                entry[k] = r[k]
        out[team] = entry
    return out


def _find_torvik_key(team_name, torvik_keys):
    """Find the Torvik key that matches this team (handles 'Michigan State' vs 'Michigan St.' etc)."""
    norm = _normalize_team_for_match(team_name)
    if not norm:
        return None
    if team_name in torvik_keys:
        return team_name
    for k in torvik_keys:
        if _normalize_team_for_match(k) == norm:
            return k
    return None


def merge_sources(torvik, kenpom, evanmiya):
    """Merge by normalized team name. Torvik is base; KenPom and Evan Miya overlay."""
    merged = {}
    for team, row in torvik.items():
        merged[team] = dict(row)
    torvik_keys = set(merged.keys())
    for team, row in kenpom.items():
        key = _find_torvik_key(team, torvik_keys) or team
        if key not in merged:
            merged[key] = {}
        merged[key]["luck"] = row.get("luck", 0)
        if row.get("kp_adj_o"):
            merged[key]["kp_adj_o"] = row["kp_adj_o"]
        if row.get("kp_adj_d"):
            merged[key]["kp_adj_d"] = row["kp_adj_d"]
        if row.get("kp_adj_tempo"):
            merged[key]["kp_adj_tempo"] = row["kp_adj_tempo"]
    for team, row in evanmiya.items():
        key = _find_torvik_key(team, torvik_keys) or team
        if key not in merged:
            merged[key] = {}
        for k, v in row.items():
            merged[key][k] = v
    return merged


def run_fetch_torvik(year):
    """Run fetch_torvik.py and return True if we have data."""
    import subprocess
    p = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "fetch_torvik.py"), str(year)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        print(p.stderr or p.stdout or "fetch_torvik failed")
        return False
    return os.path.isfile(os.path.join(DATA_DIR, f"torvik_{year}.json"))


def main():
    year = 2026
    args = sys.argv[1:]
    for a in args:
        if a == "--no-fetch":
            continue
        try:
            year = int(a)
            break
        except ValueError:
            pass
    skip_torvik_fetch = "--no-fetch" in args

    print(f"Data merge for {year}")
    print("=" * 50)

    torvik_path = os.path.join(DATA_DIR, f"torvik_{year}.json")
    if not os.path.isfile(torvik_path) and not skip_torvik_fetch:
        print("Torvik data not found. Running fetch_torvik.py...")
        if not run_fetch_torvik(year):
            print("Torvik fetch failed. Create data/torvik_YYYY.json manually or run scripts/fetch_torvik.py")
            sys.exit(1)
    elif not os.path.isfile(torvik_path):
        print("Torvik data missing. Run: python scripts/fetch_torvik.py")
        sys.exit(1)

    torvik = load_torvik(year)
    if not torvik:
        print("No Torvik records loaded. Check data/torvik_YYYY.json")
        sys.exit(1)
    print(f"Torvik: {len(torvik)} teams")

    kenpom = load_kenpom(year)
    if kenpom:
        print(f"KenPom: {len(kenpom)} teams (luck overlay)")
    else:
        print("KenPom: not used (optional)")

    evanmiya = load_evanmiya(year)
    if not evanmiya and os.path.isfile(os.path.join(DATA_DIR, f"evanmiya_{year}.csv")):
        print("Evan Miya CSV found. Run: python scripts/fetch_evanmiya.py")
    if evanmiya:
        print(f"Evan Miya: {len(evanmiya)} teams (star_score overlay)")

    merged = merge_sources(torvik, kenpom, evanmiya)
    # Output list of dicts with team name in each row
    out_list = []
    for team, row in merged.items():
        r = dict(row)
        r["team"] = team
        out_list.append(r)

    out_path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
    with open(out_path, "w") as f:
        json.dump(out_list, f, indent=2)
    print(f"\nMerged {len(out_list)} teams -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
