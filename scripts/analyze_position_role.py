#!/usr/bin/env python3
"""
Analyze position and role columns from EvanMiya player CSVs as potential
prediction features for NCAA tournament games.

Position: continuous 1 (PG) to 5 (C)
Role: continuous 1 (primary/starter) to 5 (bench/specialist)
"""
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")

# Years with BOTH player CSVs and results
YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
         2021, 2022, 2023, 2024, 2025]

# ── Name normalization (mirrors engine.py) ──────────────────────────────────

_NAME_ALIASES = {
    "uconn": "connecticut", "unc": "north carolina",
    "miami fl": "miami", "miami fla": "miami", "miami (fl)": "miami", "miami (fla)": "miami",
    "miami ohio": "miami oh", "miami (ohio)": "miami oh",
    "ucf": "central florida", "usc": "southern california",
    "lsu": "louisiana state", "ole miss": "mississippi",
    "pitt": "pittsburgh", "umass": "massachusetts",
    "vcu": "virginia commonwealth", "smu": "southern methodist",
    "byu": "brigham young", "tcu": "texas christian",
    "utep": "texas el paso", "unlv": "nevada las vegas",
    "uab": "alabama birmingham", "niu": "northern illinois",
    "siu": "southern illinois", "etsu": "east tennessee state",
    "sfa": "stephen f austin",
    "saint marys": "st marys", "saint marys ca": "st marys",
    "st marys ca": "st marys", "st marys (ca)": "st marys",
    "saint josephs": "st josephs", "saint louis": "st louis",
    "saint johns": "st johns", "saint peters": "st peters",
    "saint bonaventure": "st bonaventure",
    "uc irvine": "irvine", "uc santa barbara": "uc santa barb",
    "long island university": "long island", "liu": "long island",
    "fdu": "fairleigh dickinson",
    "a&m corpus christi": "texas a&m corpus christi",
    "texas a&m corpus chris": "texas a&m corpus christi",
    "texas a&m-corpus christi": "texas a&m corpus christi",
    "texas a&m\u2013corpus christi": "texas a&m corpus christi",
    "nc state": "north carolina state",
    "grambling state": "grambling st",
    "boston": "boston university",
    "nc central": "north carolina central",
    "loyolachicago": "loyola chicago",
    "iowa state": "iowa st", "michigan state": "michigan st",
    "ohio state": "ohio st", "florida state": "florida st",
    "kansas state": "kansas st", "colorado state": "colorado st",
    "utah state": "utah st", "penn state": "penn st",
    "oregon state": "oregon st", "mississippi state": "mississippi st",
    "boise state": "boise st", "arizona state": "arizona st",
    "oklahoma state": "oklahoma st", "washington state": "washington st",
    "san diego state": "san diego st", "fresno state": "fresno st",
    "norfolk state": "norfolk st", "alabama state": "alabama st",
    "mcneese state": "mcneese st", "morehead state": "morehead st",
    "montana state": "montana st", "cleveland state": "cleveland st",
    "wright state": "wright st", "kent state": "kent st",
    "wichita state": "wichita st", "murray state": "murray st",
    "south dakota state": "south dakota st", "north dakota state": "north dakota st",
    "jacksonville state": "jacksonville st", "appalachian state": "appalachian st",
    "kennesaw state": "kennesaw st",
    "st johns (ny)": "st johns", "saint johns (ny)": "st johns", "st johns ny": "st johns",
    "siuedwardsville": "siu edwardsville",
    "omaha": "nebraska omaha", "nebraskoomaha": "nebraska omaha",
    "wku": "western kentucky", "texas arlington": "ut arlington",
    "texasarlington": "ut arlington",
    "cal state fullerton": "cal st fullerton", "cal state northridge": "cal st northridge",
    "miss valley st": "mississippi valley st",
    "ualr": "arkansas little rock", "college of charleston": "charleston",
    "pennsylvania": "penn", "pennsylvania quakers": "penn",
    "mass lowell": "massachusetts lowell", "masslowell": "massachusetts lowell",
    "s dakota st": "south dakota st", "uta": "ut arlington",
}


def normalize_name(name):
    if not name or not isinstance(name, str):
        return ""
    s = " ".join(name.strip().split()).lower()
    s = re.sub(r"['\-\.\u2013\u2014]", "", s)
    s = re.sub(r"\(([^)]+)\)", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _NAME_ALIASES.get(s, s)


# ── Data loading ─────────────────────────────────────────────────────────────

def _num(val, default=0.0):
    if val is None or val == "":
        return default
    try:
        return float(str(val).replace("%", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def load_players(year):
    """Load player CSV, return dict: normalized_team -> list of player dicts."""
    path = os.path.join(DATA_DIR, f"evanmiya_players_{year}.csv")
    if not os.path.isfile(path):
        return {}
    teams = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team_raw = row.get("team", "").strip()
            if not team_raw:
                continue
            key = normalize_name(team_raw)
            bpr = _num(row.get("bpr"))
            obpr = _num(row.get("obpr"))
            dbpr = _num(row.get("dbpr"))
            pos = _num(row.get("position"), 3.0)
            pos = max(1.0, min(5.0, pos))
            role = _num(row.get("role"), 3.0)
            role = max(1.0, min(5.0, role))
            poss = max(_num(row.get("poss"), 1), 1)
            adj_off = _num(row.get("adj_team_off_eff"), 0)
            adj_def = _num(row.get("adj_team_def_eff"), 0)
            if key not in teams:
                teams[key] = {"players": [], "adj_off": adj_off, "adj_def": adj_def}
            teams[key]["players"].append({
                "name": row.get("name", ""),
                "bpr": bpr, "obpr": obpr, "dbpr": dbpr,
                "pos": pos, "role": role, "poss": poss,
            })
    # Sort by BPR descending
    for t in teams.values():
        t["players"].sort(key=lambda p: -p["bpr"])
    return teams


def load_results(year):
    """Load tournament results for a year."""
    path = os.path.join(DATA_DIR, f"results_{year}.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("games", [])


# ── Feature computation ──────────────────────────────────────────────────────

def compute_team_features(team_data, top_n=8):
    """Compute position/role features from a team's player list.
    Returns dict of features or None if insufficient data."""
    players = team_data["players"][:top_n]
    if len(players) < 3:
        return None

    top5 = players[:5]
    top1 = players[0]

    # Existing model metrics (for residual analysis)
    top5_bpr = sum(p["bpr"] for p in top5)
    star_conc = top1["bpr"] / top5_bpr if top5_bpr > 0 else 1.0
    supporting_bpr = sum(p["bpr"] for p in players[1:5])
    depth_quality = max(0.0, min(1.0, (supporting_bpr - 5) / 20))
    depth_balance = 1.0 - star_conc
    depth_score = 0.6 * depth_quality + 0.4 * depth_balance

    adj_off = team_data.get("adj_off", 100)
    adj_def = team_data.get("adj_def", 100)
    eff_margin = adj_off - adj_def if adj_off > 50 and adj_def > 50 else 0

    # ── NEW POSITION/ROLE FEATURES ──

    # 1. Positional diversity (std dev of positions among top players)
    positions = [p["pos"] for p in top5]
    pos_mean = sum(positions) / len(positions)
    pos_std = (sum((p - pos_mean)**2 for p in positions) / len(positions)) ** 0.5

    # 2. Guard BPR vs Big BPR (threshold at position 3.0)
    guard_players = [p for p in players if p["pos"] < 3.0]
    big_players = [p for p in players if p["pos"] >= 3.0]
    guard_bpr = sum(p["bpr"] for p in guard_players[:4])
    big_bpr = sum(p["bpr"] for p in big_players[:4])
    guard_big_ratio = guard_bpr / big_bpr if big_bpr > 0 else 2.0

    # 3. Role distribution: BPR in role-1 players vs others
    role1_players = [p for p in players if p["role"] < 1.5]
    role_other = [p for p in players if p["role"] >= 1.5]
    role1_bpr = sum(p["bpr"] for p in role1_players[:5])
    role_other_bpr = sum(p["bpr"] for p in role_other[:5])
    role_concentration = role1_bpr / (role1_bpr + role_other_bpr) if (role1_bpr + role_other_bpr) > 0 else 0.5

    # 4. Backcourt depth vs frontcourt depth
    bc_depth = sum(p["bpr"] for p in guard_players[:3])  # top 3 guards
    fc_depth = sum(p["bpr"] for p in big_players[:3])     # top 3 bigs
    bc_fc_diff = bc_depth - fc_depth

    # 5. Star position
    star_pos = top1["pos"]
    star_is_guard = 1 if star_pos < 3.0 else 0
    star_is_big = 1 if star_pos >= 4.0 else 0
    star_bpr = top1["bpr"]

    # 6. Guard floor general: highest-BPR pure PG (pos < 1.5)
    pgs = [p for p in players if p["pos"] < 1.5]
    best_pg_bpr = pgs[0]["bpr"] if pgs else 0.0

    # 7. Positional balance score: how evenly distributed is talent across positions?
    # Bin into 3 buckets: guards (<2.5), wings (2.5-3.5), bigs (>3.5)
    bins = {"guard": [], "wing": [], "big": []}
    for p in top5:
        if p["pos"] < 2.5:
            bins["guard"].append(p["bpr"])
        elif p["pos"] <= 3.5:
            bins["wing"].append(p["bpr"])
        else:
            bins["big"].append(p["bpr"])
    bin_sums = [sum(v) for v in bins.values()]
    total_bin = sum(bin_sums)
    if total_bin > 0:
        fracs = [max(0, b / total_bin) for b in bin_sums]
        # Entropy-based balance (max at equal thirds)
        pos_entropy = -sum(f * math.log(f + 1e-10) for f in fracs if f > 0) / math.log(3)
    else:
        pos_entropy = 0.0

    # 8. Interior dominance: top big BPR
    best_big_bpr = big_players[0]["bpr"] if big_players else 0.0

    # 9. Wing talent (pos 2.5-3.5)
    wings = [p for p in players if 2.5 <= p["pos"] <= 3.5]
    wing_bpr = sum(p["bpr"] for p in wings[:3])

    # 10. Role depth: how many role-1 or role-2 players (primary contributors)?
    primary_count = sum(1 for p in players if p["role"] < 2.0)

    # 11. Average role of top-5 (lower = more starters)
    avg_role_top5 = sum(p["role"] for p in top5) / len(top5)

    # 12. Possession-weighted position (team's "effective size")
    total_poss = sum(p["poss"] for p in players)
    if total_poss > 0:
        eff_size = sum(p["pos"] * p["poss"] for p in players) / total_poss
    else:
        eff_size = 3.0

    return {
        # Existing features for residual analysis
        "eff_margin": eff_margin,
        "top5_bpr": top5_bpr,
        "star_bpr": star_bpr,
        "star_conc": star_conc,
        "depth_score": depth_score,
        # New features
        "pos_diversity": pos_std,
        "guard_bpr": guard_bpr,
        "big_bpr": big_bpr,
        "guard_big_ratio": guard_big_ratio,
        "role_concentration": role_concentration,
        "bc_fc_diff": bc_fc_diff,
        "star_pos": star_pos,
        "star_is_guard": star_is_guard,
        "star_is_big": star_is_big,
        "best_pg_bpr": best_pg_bpr,
        "pos_entropy": pos_entropy,
        "best_big_bpr": best_big_bpr,
        "wing_bpr": wing_bpr,
        "primary_count": primary_count,
        "avg_role_top5": avg_role_top5,
        "eff_size": eff_size,
    }


# ── Matchup assembly ────────────────────────────────────────────────────────

def build_matchups():
    """Build all tournament matchups with features for both teams."""
    matchups = []
    unmatched_teams = defaultdict(int)

    for year in YEARS:
        players = load_players(year)
        results = load_results(year)
        if not players or not results:
            continue

        for game in results:
            team_a_raw = game.get("team_a", "")
            team_b_raw = game.get("team_b", "")
            winner_raw = game.get("winner", "")
            seed_a = game.get("seed_a", 8)
            seed_b = game.get("seed_b", 8)
            rnd = game.get("round", 64)

            ka = normalize_name(team_a_raw)
            kb = normalize_name(team_b_raw)
            kw = normalize_name(winner_raw)

            # Fuzzy fallback: if exact match fails, try substring
            def find_team(key, players_dict):
                if key in players_dict:
                    return players_dict[key]
                for k in players_dict:
                    if len(key) >= 4 and key in k and len(key) >= len(k) * 0.6:
                        return players_dict[k]
                    if len(k) >= 4 and k in key and len(k) >= len(key) * 0.6:
                        return players_dict[k]
                return None

            data_a = find_team(ka, players)
            data_b = find_team(kb, players)

            if data_a is None:
                unmatched_teams[f"{year}:{team_a_raw}"] += 1
                continue
            if data_b is None:
                unmatched_teams[f"{year}:{team_b_raw}"] += 1
                continue

            fa = compute_team_features(data_a)
            fb = compute_team_features(data_b)
            if fa is None or fb is None:
                continue

            # Who won? a_won = 1 if team_a won
            a_won = 1 if kw == ka or (kw and kw in ka) or (kw and ka in kw) else 0
            # Double-check with raw names
            if a_won == 0 and winner_raw.strip().lower() == team_a_raw.strip().lower():
                a_won = 1

            matchups.append({
                "year": year, "round": rnd,
                "team_a": team_a_raw, "team_b": team_b_raw,
                "seed_a": seed_a, "seed_b": seed_b,
                "a_won": a_won,
                "fa": fa, "fb": fb,
            })

    if unmatched_teams:
        top_unmatched = sorted(unmatched_teams.items(), key=lambda x: -x[1])[:10]
        print(f"\nTop unmatched teams (could not find in player CSV):")
        for name, count in top_unmatched:
            print(f"  {name}: {count} games")
    return matchups


# ── Statistical helpers ──────────────────────────────────────────────────────

def win_rate_analysis(matchups, feature_diff_fn, label, bins=None):
    """Given a function that computes a differential feature (team_a - team_b),
    analyze win rate correlation."""
    diffs = []
    for m in matchups:
        try:
            diff = feature_diff_fn(m["fa"], m["fb"])
        except (ZeroDivisionError, KeyError):
            continue
        if diff is None or not math.isfinite(diff):
            continue
        diffs.append((diff, m["a_won"]))

    if len(diffs) < 30:
        print(f"\n{'='*70}")
        print(f"  {label}: INSUFFICIENT DATA (n={len(diffs)})")
        return None

    # Sort by diff
    diffs.sort(key=lambda x: x[0])
    n = len(diffs)

    # Overall correlation (point-biserial)
    x_vals = [d[0] for d in diffs]
    y_vals = [d[1] for d in diffs]
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / n
    std_x = (sum((x - x_mean)**2 for x in x_vals) / n) ** 0.5
    std_y = (sum((y - y_mean)**2 for y in y_vals) / n) ** 0.5
    corr = cov_xy / (std_x * std_y) if std_x > 0 and std_y > 0 else 0

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  N={n}, Correlation with A winning: {corr:+.4f}")

    # Quintile analysis
    q_size = n // 5
    if q_size >= 5:
        print(f"  {'Quintile':<12} {'Range':>20} {'Win%':>8} {'N':>6}")
        for q in range(5):
            start = q * q_size
            end = (q + 1) * q_size if q < 4 else n
            subset = diffs[start:end]
            wins = sum(d[1] for d in subset)
            total = len(subset)
            lo = subset[0][0]
            hi = subset[-1][0]
            print(f"  Q{q+1} (low→high) [{lo:+6.2f}, {hi:+6.2f}]  {wins/total*100:5.1f}%  {total:5d}")

    # Binary split: positive diff vs negative diff
    pos = [(d, w) for d, w in diffs if d > 0]
    neg = [(d, w) for d, w in diffs if d < 0]
    zero = [(d, w) for d, w in diffs if d == 0]
    if pos and neg:
        pos_wr = sum(w for _, w in pos) / len(pos) * 100
        neg_wr = sum(w for _, w in neg) / len(neg) * 100
        print(f"  When A has HIGHER feature: {pos_wr:.1f}% win rate (n={len(pos)})")
        print(f"  When A has LOWER  feature: {neg_wr:.1f}% win rate (n={len(neg)})")

    return {"corr": corr, "n": n, "diffs": diffs}


def residual_analysis(matchups, feature_diff_fn, label, base_features=None):
    """Check if feature adds signal BEYOND base efficiency metrics.
    Uses logistic-like residual: compute expected win rate from base features,
    then check if the new feature correlates with prediction errors."""
    if base_features is None:
        base_features = ["eff_margin"]

    # Simple approach: bin games by base feature agreement, check new feature
    triples = []  # (base_diff, new_diff, a_won)
    for m in matchups:
        fa, fb = m["fa"], m["fb"]
        try:
            new_diff = feature_diff_fn(fa, fb)
        except (ZeroDivisionError, KeyError):
            continue
        if new_diff is None or not math.isfinite(new_diff):
            continue

        # Base signal: efficiency margin differential
        base_diff = fa["eff_margin"] - fb["eff_margin"]
        triples.append((base_diff, new_diff, m["a_won"]))

    if len(triples) < 50:
        print(f"  Residual analysis: insufficient data (n={len(triples)})")
        return

    # Convert base_diff to simple expected win probability
    # Using logistic: P(A wins) = 1 / (1 + exp(-base_diff / scale))
    scale = 11.0  # roughly calibrated to NCAA scoring
    residuals = []  # (new_diff, residual)
    for base_d, new_d, won in triples:
        expected = 1.0 / (1.0 + math.exp(-base_d / scale))
        residual = won - expected  # positive = A won more than expected
        residuals.append((new_d, residual))

    n = len(residuals)
    x_vals = [r[0] for r in residuals]
    y_vals = [r[1] for r in residuals]
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / n
    sx = (sum((x - x_mean)**2 for x in x_vals) / n) ** 0.5
    sy = (sum((y - y_mean)**2 for y in y_vals) / n) ** 0.5
    res_corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0

    print(f"  Residual correlation (beyond efficiency): {res_corr:+.4f}")

    # Also check: in CLOSE games (small base_diff), does feature matter more?
    close_games = [(nd, won) for bd, nd, won in triples if abs(bd) < 8]
    if len(close_games) >= 30:
        pos = [(nd, w) for nd, w in close_games if nd > 0]
        neg = [(nd, w) for nd, w in close_games if nd < 0]
        if pos and neg and len(pos) >= 10 and len(neg) >= 10:
            pos_wr = sum(w for _, w in pos) / len(pos) * 100
            neg_wr = sum(w for _, w in neg) / len(neg) * 100
            print(f"  In CLOSE games (eff_margin diff < 8, n={len(close_games)}):")
            print(f"    A higher → {pos_wr:.1f}% WR (n={len(pos)}), A lower → {neg_wr:.1f}% WR (n={len(neg)})")

    # Even more restricted: equal-talent games (top5_bpr similar too)
    equal_games = [(nd, won) for bd, nd, won in triples if abs(bd) < 8]
    # Additional filter by top5_bpr
    equal2 = []
    for m in matchups:
        fa, fb = m["fa"], m["fb"]
        try:
            new_diff = feature_diff_fn(fa, fb)
        except (ZeroDivisionError, KeyError):
            continue
        if new_diff is None or not math.isfinite(new_diff):
            continue
        eff_diff = abs(fa["eff_margin"] - fb["eff_margin"])
        bpr_diff = abs(fa["top5_bpr"] - fb["top5_bpr"])
        if eff_diff < 8 and bpr_diff < 8:
            equal2.append((new_diff, m["a_won"]))

    if len(equal2) >= 30:
        pos = [(nd, w) for nd, w in equal2 if nd > 0]
        neg = [(nd, w) for nd, w in equal2 if nd < 0]
        if pos and neg and len(pos) >= 10 and len(neg) >= 10:
            pos_wr = sum(w for _, w in pos) / len(pos) * 100
            neg_wr = sum(w for _, w in neg) / len(neg) * 100
            print(f"  In EQUAL-TALENT games (eff<8 AND top5_bpr<8, n={len(equal2)}):")
            print(f"    A higher → {pos_wr:.1f}% WR (n={len(pos)}), A lower → {neg_wr:.1f}% WR (n={len(neg)})")


def categorical_analysis(matchups, cat_fn, label):
    """For star position type: guard star vs big star vs wing star."""
    buckets = defaultdict(lambda: {"wins": 0, "total": 0})
    for m in matchups:
        try:
            cat_a = cat_fn(m["fa"])
            cat_b = cat_fn(m["fb"])
        except (ZeroDivisionError, KeyError):
            continue
        key = f"{cat_a}_vs_{cat_b}"
        buckets[key]["total"] += 1
        buckets[key]["wins"] += m["a_won"]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {'Matchup':<25} {'A_WinRate':>10} {'N':>6}")
    for key in sorted(buckets.keys()):
        b = buckets[key]
        if b["total"] >= 10:
            wr = b["wins"] / b["total"] * 100
            print(f"  {key:<25} {wr:8.1f}%  {b['total']:5d}")


# ── Main analysis ────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  NCAA TOURNAMENT POSITION/ROLE FEATURE ANALYSIS")
    print("  EvanMiya Player Data, 2010-2025 (excl 2020)")
    print("="*70)

    matchups = build_matchups()
    print(f"\nTotal matchups with both teams' player data: {len(matchups)}")
    print(f"Years covered: {sorted(set(m['year'] for m in matchups))}")

    # ── Descriptive stats on position and role ──
    print(f"\n{'='*70}")
    print("  DESCRIPTIVE: Position and Role distributions among tournament teams")
    all_star_pos = []
    all_roles = []
    for m in matchups:
        all_star_pos.append(m["fa"]["star_pos"])
        all_star_pos.append(m["fb"]["star_pos"])
        all_roles.append(m["fa"]["avg_role_top5"])
        all_roles.append(m["fb"]["avg_role_top5"])
    print(f"  Star player position: mean={sum(all_star_pos)/len(all_star_pos):.2f}, "
          f"min={min(all_star_pos):.2f}, max={max(all_star_pos):.2f}")
    print(f"  Avg role (top 5): mean={sum(all_roles)/len(all_roles):.2f}, "
          f"min={min(all_roles):.2f}, max={max(all_roles):.2f}")

    # Distribution of star positions
    guard_stars = sum(1 for p in all_star_pos if p < 2.5)
    wing_stars = sum(1 for p in all_star_pos if 2.5 <= p <= 3.5)
    big_stars = sum(1 for p in all_star_pos if p > 3.5)
    print(f"  Star type: Guards(<2.5)={guard_stars} ({guard_stars/len(all_star_pos)*100:.0f}%), "
          f"Wings(2.5-3.5)={wing_stars} ({wing_stars/len(all_star_pos)*100:.0f}%), "
          f"Bigs(>3.5)={big_stars} ({big_stars/len(all_star_pos)*100:.0f}%)")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 1: Positional diversity
    # ─────────────────────────────────────────────────────────────────────
    result = win_rate_analysis(
        matchups,
        lambda fa, fb: fa["pos_diversity"] - fb["pos_diversity"],
        "ANALYSIS 1: Positional Diversity (std dev of top-5 positions)"
    )
    if result:
        residual_analysis(matchups,
                          lambda fa, fb: fa["pos_diversity"] - fb["pos_diversity"],
                          "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 2: Guard BPR vs Big BPR (total)
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["guard_bpr"] - fb["guard_bpr"],
        "ANALYSIS 2a: Guard BPR advantage (sum of top-4 guards, pos<3)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["guard_bpr"] - fb["guard_bpr"],
                      "")

    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["big_bpr"] - fb["big_bpr"],
        "ANALYSIS 2b: Big BPR advantage (sum of top-4 bigs, pos>=3)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["big_bpr"] - fb["big_bpr"],
                      "")

    # Which matters MORE: guard advantage or big advantage?
    print(f"\n{'='*70}")
    print("  ANALYSIS 2c: Guard advantage vs Big advantage — which predicts better?")
    guard_better_wins = 0
    guard_better_n = 0
    big_better_wins = 0
    big_better_n = 0
    for m in matchups:
        fa, fb = m["fa"], m["fb"]
        guard_diff = fa["guard_bpr"] - fb["guard_bpr"]
        big_diff = fa["big_bpr"] - fb["big_bpr"]
        if guard_diff > 0 and big_diff < 0:
            # A has better guards, B has better bigs
            guard_better_n += 1
            guard_better_wins += m["a_won"]
        elif guard_diff < 0 and big_diff > 0:
            # B has better guards, A has better bigs
            big_better_n += 1
            big_better_wins += (1 - m["a_won"])  # B winning = big advantage winning
    if guard_better_n >= 10 and big_better_n >= 10:
        print(f"  When teams split (one has guard edge, other has big edge):")
        print(f"    Guard-advantage team wins: {guard_better_wins/guard_better_n*100:.1f}% (n={guard_better_n})")
        print(f"    Big-advantage team wins:   {big_better_wins/big_better_n*100:.1f}% (n={big_better_n})")
    else:
        print(f"  Insufficient split-advantage games: guard={guard_better_n}, big={big_better_n}")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 3: Role concentration
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["role_concentration"] - fb["role_concentration"],
        "ANALYSIS 3: Role Concentration (fraction of BPR in role<1.5 players)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["role_concentration"] - fb["role_concentration"],
                      "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 4: Backcourt vs Frontcourt depth
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["bc_fc_diff"] - fb["bc_fc_diff"],
        "ANALYSIS 4: Backcourt-Frontcourt Balance (bc_depth - fc_depth differential)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["bc_fc_diff"] - fb["bc_fc_diff"],
                      "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 5: Star position interaction
    # ─────────────────────────────────────────────────────────────────────
    # Does a guard star outperform a big star in tournament?
    def star_type(fa):
        if fa["star_pos"] < 2.5:
            return "guard"
        elif fa["star_pos"] <= 3.5:
            return "wing"
        else:
            return "big"

    categorical_analysis(matchups, star_type,
                         "ANALYSIS 5a: Star Position Type matchups (guard/wing/big)")

    # Controlling for BPR: among games where star BPR is similar
    print(f"\n{'='*70}")
    print("  ANALYSIS 5b: Star position effect CONTROLLING for star BPR")
    # Guard star vs big star, similar BPR
    guard_star_wins = 0
    guard_star_n = 0
    for m in matchups:
        fa, fb = m["fa"], m["fb"]
        bpr_diff = abs(fa["star_bpr"] - fb["star_bpr"])
        if bpr_diff > 2:
            continue  # only look at similar-BPR matchups
        a_guard = fa["star_pos"] < 2.5
        b_guard = fb["star_pos"] < 2.5
        if a_guard and not b_guard:
            guard_star_n += 1
            guard_star_wins += m["a_won"]
        elif b_guard and not a_guard:
            guard_star_n += 1
            guard_star_wins += (1 - m["a_won"])
    if guard_star_n >= 20:
        print(f"  Guard star vs non-guard star (similar BPR, diff<2):")
        print(f"    Guard star wins: {guard_star_wins/guard_star_n*100:.1f}% (n={guard_star_n})")
    else:
        print(f"  Insufficient similar-BPR guard vs non-guard matchups: n={guard_star_n}")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 6: Positional matchup advantages
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ANALYSIS 6: Positional matchup advantages")
    print("  When A has better guards but B has better bigs (both by BPR sum):")
    a_guards_b_bigs = []
    for m in matchups:
        fa, fb = m["fa"], m["fb"]
        a_guard_edge = fa["guard_bpr"] - fb["guard_bpr"]
        a_big_edge = fa["big_bpr"] - fb["big_bpr"]
        if a_guard_edge > 2 and a_big_edge < -2:
            a_guards_b_bigs.append(m["a_won"])
        elif a_guard_edge < -2 and a_big_edge > 2:
            a_guards_b_bigs.append(1 - m["a_won"])  # flip so "guard team" = 1
    if len(a_guards_b_bigs) >= 10:
        wr = sum(a_guards_b_bigs) / len(a_guards_b_bigs) * 100
        print(f"    Guard-heavy team wins: {wr:.1f}% (n={len(a_guards_b_bigs)})")
    else:
        print(f"    Insufficient clear-split matchups: n={len(a_guards_b_bigs)}")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 7: Guard floor general effect
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["best_pg_bpr"] - fb["best_pg_bpr"],
        "ANALYSIS 7: PG Floor General (best PG BPR, pos<1.5)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["best_pg_bpr"] - fb["best_pg_bpr"],
                      "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 8: Positional entropy (balance)
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["pos_entropy"] - fb["pos_entropy"],
        "ANALYSIS 8: Positional Entropy (balanced talent across guard/wing/big)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["pos_entropy"] - fb["pos_entropy"],
                      "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 9: Effective team size
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["eff_size"] - fb["eff_size"],
        "ANALYSIS 9: Effective Team Size (poss-weighted avg position)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["eff_size"] - fb["eff_size"],
                      "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 10: Wing BPR
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["wing_bpr"] - fb["wing_bpr"],
        "ANALYSIS 10: Wing BPR advantage (pos 2.5-3.5, top 3)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["wing_bpr"] - fb["wing_bpr"],
                      "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 11: Primary contributor count
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["primary_count"] - fb["primary_count"],
        "ANALYSIS 11: Primary Contributor Count (role<2.0 players)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["primary_count"] - fb["primary_count"],
                      "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 12: Average role of top 5
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["avg_role_top5"] - fb["avg_role_top5"],
        "ANALYSIS 12: Avg Role of Top 5 (lower = more primary contributors)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["avg_role_top5"] - fb["avg_role_top5"],
                      "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 13: Best big BPR
    # ─────────────────────────────────────────────────────────────────────
    win_rate_analysis(
        matchups,
        lambda fa, fb: fa["best_big_bpr"] - fb["best_big_bpr"],
        "ANALYSIS 13: Best Big BPR (top player with pos>=3)"
    )
    residual_analysis(matchups,
                      lambda fa, fb: fa["best_big_bpr"] - fb["best_big_bpr"],
                      "")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS 14: Round-by-round — do features matter more in later rounds?
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ANALYSIS 14: Feature importance by tournament round")
    features_to_check = [
        ("guard_bpr", lambda fa, fb: fa["guard_bpr"] - fb["guard_bpr"]),
        ("big_bpr", lambda fa, fb: fa["big_bpr"] - fb["big_bpr"]),
        ("best_pg_bpr", lambda fa, fb: fa["best_pg_bpr"] - fb["best_pg_bpr"]),
        ("pos_entropy", lambda fa, fb: fa["pos_entropy"] - fb["pos_entropy"]),
        ("eff_size", lambda fa, fb: fa["eff_size"] - fb["eff_size"]),
        ("best_big_bpr", lambda fa, fb: fa["best_big_bpr"] - fb["best_big_bpr"]),
    ]
    rounds = [(64, "R64"), (32, "R32"), (16, "S16"), (8, "E8"), (4, "F4"), (2, "Final")]

    print(f"  {'Feature':<16}", end="")
    for _, rname in rounds:
        print(f" {rname:>8}", end="")
    print(f" {'All':>8}")

    for fname, ffn in features_to_check:
        print(f"  {fname:<16}", end="")
        for rnd, _ in rounds:
            rm = [m for m in matchups if m["round"] == rnd]
            if len(rm) < 10:
                print(f" {'n/a':>8}", end="")
                continue
            pos_wins = 0
            pos_n = 0
            for m in rm:
                try:
                    d = ffn(m["fa"], m["fb"])
                except:
                    continue
                if d > 0:
                    pos_n += 1
                    pos_wins += m["a_won"]
                elif d < 0:
                    pos_n += 1
                    pos_wins += (1 - m["a_won"])
            if pos_n >= 5:
                print(f" {pos_wins/pos_n*100:7.1f}%", end="")
            else:
                print(f" {'n/a':>8}", end="")
        # All rounds
        pos_wins_all = 0
        pos_n_all = 0
        for m in matchups:
            try:
                d = ffn(m["fa"], m["fb"])
            except:
                continue
            if d > 0:
                pos_n_all += 1
                pos_wins_all += m["a_won"]
            elif d < 0:
                pos_n_all += 1
                pos_wins_all += (1 - m["a_won"])
        if pos_n_all > 0:
            print(f" {pos_wins_all/pos_n_all*100:7.1f}%", end="")
        print()

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY: All features — raw correlation and residual correlation")
    print(f"  {'Feature':<35} {'RawCorr':>8} {'ResCorr':>8} {'Adv%':>8} {'N':>6}")

    summary_features = [
        ("pos_diversity", lambda fa, fb: fa["pos_diversity"] - fb["pos_diversity"]),
        ("guard_bpr", lambda fa, fb: fa["guard_bpr"] - fb["guard_bpr"]),
        ("big_bpr", lambda fa, fb: fa["big_bpr"] - fb["big_bpr"]),
        ("guard_big_ratio", lambda fa, fb: fa["guard_big_ratio"] - fb["guard_big_ratio"]),
        ("role_concentration", lambda fa, fb: fa["role_concentration"] - fb["role_concentration"]),
        ("bc_fc_diff", lambda fa, fb: fa["bc_fc_diff"] - fb["bc_fc_diff"]),
        ("best_pg_bpr", lambda fa, fb: fa["best_pg_bpr"] - fb["best_pg_bpr"]),
        ("pos_entropy", lambda fa, fb: fa["pos_entropy"] - fb["pos_entropy"]),
        ("best_big_bpr", lambda fa, fb: fa["best_big_bpr"] - fb["best_big_bpr"]),
        ("wing_bpr", lambda fa, fb: fa["wing_bpr"] - fb["wing_bpr"]),
        ("primary_count", lambda fa, fb: fa["primary_count"] - fb["primary_count"]),
        ("avg_role_top5", lambda fa, fb: fa["avg_role_top5"] - fb["avg_role_top5"]),
        ("eff_size", lambda fa, fb: fa["eff_size"] - fb["eff_size"]),
        # Existing features for comparison
        ("(baseline) eff_margin", lambda fa, fb: fa["eff_margin"] - fb["eff_margin"]),
        ("(baseline) top5_bpr", lambda fa, fb: fa["top5_bpr"] - fb["top5_bpr"]),
        ("(baseline) star_bpr", lambda fa, fb: fa["star_bpr"] - fb["star_bpr"]),
        ("(baseline) depth_score", lambda fa, fb: fa["depth_score"] - fb["depth_score"]),
    ]

    for fname, ffn in summary_features:
        diffs = []
        for m in matchups:
            try:
                d = ffn(m["fa"], m["fb"])
            except:
                continue
            if d is None or not math.isfinite(d):
                continue
            diffs.append((d, m["a_won"]))

        if len(diffs) < 30:
            print(f"  {fname:<35} {'N/A':>8} {'N/A':>8} {'N/A':>8} {len(diffs):>5}")
            continue

        n = len(diffs)
        x = [d[0] for d in diffs]
        y = [d[1] for d in diffs]
        xm = sum(x) / n
        ym = sum(y) / n
        cov = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y)) / n
        sx = (sum((xi - xm)**2 for xi in x) / n) ** 0.5
        sy = (sum((yi - ym)**2 for yi in y) / n) ** 0.5
        raw_corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0

        # Residual correlation
        scale = 11.0
        res_pairs = []
        for m in matchups:
            try:
                nd = ffn(m["fa"], m["fb"])
            except:
                continue
            if nd is None or not math.isfinite(nd):
                continue
            bd = m["fa"]["eff_margin"] - m["fb"]["eff_margin"]
            exp_wr = 1.0 / (1.0 + math.exp(-bd / scale))
            resid = m["a_won"] - exp_wr
            res_pairs.append((nd, resid))

        if len(res_pairs) >= 30:
            rx = [p[0] for p in res_pairs]
            ry = [p[1] for p in res_pairs]
            rxm = sum(rx) / len(rx)
            rym = sum(ry) / len(ry)
            rcov = sum((a - rxm) * (b - rym) for a, b in zip(rx, ry)) / len(rx)
            rsx = (sum((a - rxm)**2 for a in rx) / len(rx)) ** 0.5
            rsy = (sum((b - rym)**2 for b in ry) / len(ry)) ** 0.5
            res_corr = rcov / (rsx * rsy) if rsx > 0 and rsy > 0 else 0
        else:
            res_corr = float('nan')

        # Advantage win rate
        pos = [w for d, w in diffs if d > 0]
        neg = [w for d, w in diffs if d < 0]
        if pos and neg:
            # Combined: when you have the advantage, how often do you win?
            adv_wr = (sum(pos) + len(neg) - sum(neg)) / (len(pos) + len(neg)) * 100
        else:
            adv_wr = 50.0

        rc_str = f"{res_corr:+.4f}" if math.isfinite(res_corr) else "N/A"
        print(f"  {fname:<35} {raw_corr:+.4f}  {rc_str:>8}  {adv_wr:6.1f}%  {n:5d}")

    print(f"\n{'='*70}")
    print("  INTERPRETATION GUIDE:")
    print("  RawCorr: Point-biserial correlation with win outcome (higher = stronger)")
    print("  ResCorr: Correlation with residual after accounting for efficiency margin")
    print("           (>0 means feature adds signal BEYOND efficiency)")
    print("  Adv%:    Win rate when team has advantage in this feature")
    print("  Benchmark: eff_margin raw corr ~0.25-0.35 is a 'strong' predictor")
    print("  A ResCorr of +0.03-0.05 is meaningful; <0.02 is likely noise")
    print("="*70)


if __name__ == "__main__":
    main()
