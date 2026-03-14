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


def load_injuries(year):
    """Load injury data from fetch_injuries.py output. Returns dict team -> list of injury dicts."""
    path = os.path.join(DATA_DIR, f"injuries_{year}.json")
    if not os.path.isfile(path):
        return {}
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    return data


def merge_injuries(merged, injuries_data):
    """Overlay injuries onto merged teams. Matches by normalized team name."""
    if not injuries_data:
        return
    merged_keys = set(merged.keys())
    for team_name, inj_list in injuries_data.items():
        key = _find_torvik_key(team_name, merged_keys) or normalize_team(team_name)
        if key in merged and isinstance(inj_list, list):
            merged[key]["injuries"] = inj_list


def load_momentum(year):
    """Load momentum data from compute_momentum.py output. Returns dict team -> {momentum, ...}."""
    path = os.path.join(DATA_DIR, f"momentum_{year}.json")
    if not os.path.isfile(path):
        return {}
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    return data


def merge_momentum(merged, momentum_data):
    """Overlay momentum onto merged teams. Matches by normalized team name."""
    if not momentum_data:
        return
    merged_keys = set(merged.keys())
    for team_name, mom in momentum_data.items():
        key = _find_torvik_key(team_name, merged_keys) or normalize_team(team_name)
        if key in merged and isinstance(mom, dict):
            merged[key]["momentum"] = mom.get("momentum", 0)
            if mom.get("adj_o_recent") is not None:
                merged[key]["adj_o_recent"] = mom["adj_o_recent"]
            if mom.get("adj_d_recent") is not None:
                merged[key]["adj_d_recent"] = mom["adj_d_recent"]


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


def compute_conf_strength_scores(merged):
    """Derive conf_strength_score (0-1) from Torvik conf_adj_o/conf_adj_d.

    Teams in stronger conferences have conf_adj_o > adj_o (conference adjustment boosts them).
    Formula: raw = (conf_adj_o - adj_o + adj_d - conf_adj_d) / 20.
    Normalize across all teams so min=0, max=1.
    """
    raw_scores = []
    for team, row in merged.items():
        conf_o = row.get("conf_adj_o")
        conf_d = row.get("conf_adj_d")
        adj_o = row.get("adj_o")
        adj_d = row.get("adj_d")
        if conf_o is not None and conf_d is not None and adj_o is not None and adj_d is not None:
            raw = (conf_o - adj_o + adj_d - conf_d) / 20.0
            raw_scores.append((team, raw))
    if not raw_scores:
        return
    vals = [r for _, r in raw_scores]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    for team, raw in raw_scores:
        merged[team]["conf_strength_score"] = round(max(0.0, min(1.0, (raw - lo) / span)), 4)


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
    compute_conf_strength_scores(merged)
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

    momentum_data = load_momentum(year)
    if momentum_data:
        merge_momentum(merged, momentum_data)
        print(f"Momentum: {len(momentum_data)} teams (recent form overlay)")

    injuries_data = load_injuries(year)
    if injuries_data:
        merge_injuries(merged, injuries_data)
        print(f"Injuries: {len(injuries_data)} teams with injury data")
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
