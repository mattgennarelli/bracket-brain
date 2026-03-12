"""
engine.py — Bracket Brain Prediction Engine v2

A comprehensive March Madness prediction model incorporating 12+ factors:

CORE METRICS (from Torvik T-Rank):
  1. Tempo-adjusted offensive efficiency
  2. Tempo-adjusted defensive efficiency
  3. Four Factors (eFG%, TO rate, ORB rate, FT rate)

VOLATILITY & STYLE:
  4. Three-point dependency → variance adjustment (high 3PT = more volatile)
  5. Size/physicality (2PT%, block rate, ORB rate) → reliability bonus

INTANGIBLES & HISTORY:
  6. Team experience / roster age → tournament performance bonus
  7. Coach tournament history → proven March performers
  8. Program pedigree → blue blood institutional edge
  9. Preseason ranking → true talent signal

SITUATIONAL:
  10. Geographic proximity to game site → travel/crowd advantage
  11. Recent form / momentum → hot team adjustment
  12. Key injuries → efficiency reduction

CALIBRATION:
  13. Historical seed performance → committee knowledge prior
  14. KenPom "luck" / overperformance → regression signal
"""

import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass

# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class ModelConfig:
    num_sims: int = 10000
    base_scoring_stdev: float = 11.0
    national_avg_efficiency: float = 100.0
    national_avg_tempo: float = 67.5
    seed_weight: float = 0.18
    three_pt_volatility_factor: float = 0.15
    tempo_volatility_weight: float = 0.3
    experience_max_bonus: float = 2.5
    coach_tourney_max_bonus: float = 1.5
    pedigree_max_bonus: float = 1.0
    preseason_max_bonus: float = 1.5
    proximity_max_bonus: float = 2.0
    proximity_distance_threshold: float = 500
    momentum_max_bonus: float = 1.5
    star_player_max_bonus: float = 1.5
    size_max_bonus: float = 1.0
    injury_penalty_per_level: float = 3.0
    luck_regression_factor: float = 0.5
    win_pct_max_bonus: float = 1.0
    conf_rating_max_bonus: float = 0.3
    # New: possessions / FT / schedule strength
    possession_edge_max_bonus: float = 4.0
    ft_clutch_max_bonus: float = 3.0
    sos_max_bonus: float = 2.5
    # EvanMiya-sourced signals
    depth_max_bonus: float = 2.0          # em_depth_score: deep/balanced roster bonus
    em_opp_adjust_max_bonus: float = 2.0  # em_opponent_adjust: opponent quality adjustment
    em_adj_o_weight: float = 0.3          # blend weight for EvanMiya adj_o/adj_d in base score (0=Torvik only)
    ft_foul_rate_max_bonus: float = 2.0   # margin bonus from differential FT rate (foul drawing vs committing)
    score_scale: float = 0.942            # tournament scoring discount vs regular-season efficiency baseline
                                          # (calibrated: model over-predicts by 8.8 pts on 187 tournament games)

def _load_calibrated_config():
    """Load calibrated parameters from data/calibrated_config.json if it exists."""
    config = ModelConfig()
    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibrated_config.json")
    if os.path.isfile(cal_path):
        try:
            with open(cal_path) as f:
                cal = json.load(f)
            for k, v in cal.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        except Exception:
            pass
    return config

DEFAULT_CONFIG = _load_calibrated_config()

# Exports for run.py / CLI
DEFAULT_NUM_SIMS = DEFAULT_CONFIG.num_sims
SEED_WEIGHT = DEFAULT_CONFIG.seed_weight


def win_probability(team_a, team_b, game_site=None, config=DEFAULT_CONFIG):
    """Return win probability for team_a (0..1)."""
    r = predict_game(team_a, team_b, game_site, config=config)
    return r["win_prob_a"]


def predict_game_score(team_a, team_b, game_site=None, config=DEFAULT_CONFIG):
    """Return predicted scores and win probs (same structure as predict_game)."""
    return predict_game(team_a, team_b, game_site, config=config)


# ===========================================================================
# FACTOR CALCULATIONS
# ===========================================================================

def calc_game_volatility(team, config=DEFAULT_CONFIG):
    """Compute per-team volatility multiplier from 3PT dependency, tempo, and star concentration.

    High 3PT reliance with poor shooting -> more variance.
    Very fast or very slow tempo deviations from average -> more unpredictability.
    High em_star_concentration -> more game-to-game variance (one player can swing outcome).
    """
    three_rate = team.get("three_rate", 0.35)
    three_pct = team.get("three_pct", 0.34)
    dependency = three_rate * 1.5
    consistency_penalty = max(0, (0.34 - three_pct) * 2)
    three_vol = (dependency - 0.4) * 0.5 + consistency_penalty * 0.3

    tempo = team.get("adj_tempo", config.national_avg_tempo)
    tempo_dev = abs(tempo - config.national_avg_tempo) / config.national_avg_tempo
    tempo_vol = tempo_dev * config.tempo_volatility_weight

    # Star concentration: deviation from balanced roster (avg ~0.35) adds variance.
    # Range 0-1; national avg ~0.35; one-man-band ~0.6+ adds meaningful volatility.
    star_conc = team.get("em_star_concentration", 0.35)
    star_vol = max(0.0, (star_conc - 0.35) * 0.4)

    volatility = 1.0 + three_vol + tempo_vol + star_vol
    return max(0.85, min(1.35, volatility))

def calc_experience_bonus(team, config=DEFAULT_CONFIG):
    exp = team.get("experience", 0.5)
    if exp > 1:
        exp = (exp - 1.5) / 2.5
    exp = max(0, min(1, exp))
    return exp * config.experience_max_bonus

def calc_coach_bonus(team, config=DEFAULT_CONFIG):
    return team.get("coach_tourney_score", 0.0) * config.coach_tourney_max_bonus

def calc_pedigree_bonus(team, config=DEFAULT_CONFIG):
    return team.get("pedigree_score", 0.0) * config.pedigree_max_bonus

def calc_preseason_bonus(team, config=DEFAULT_CONFIG):
    rank = team.get("preseason_rank", 0)
    if not rank or rank == 0:
        return 0.0
    score = max(0, (26 - rank) / 25)
    return score * config.preseason_max_bonus

def calc_proximity_bonus(team, game_site=None, config=DEFAULT_CONFIG):
    if "proximity_bonus" in team:
        return team["proximity_bonus"] * config.proximity_max_bonus
    if game_site is None or "location" not in team:
        return 0.0
    distance = _haversine(team["location"][0], team["location"][1], game_site[0], game_site[1])
    if distance >= config.proximity_distance_threshold:
        return 0.0
    return (1.0 - distance / config.proximity_distance_threshold) * config.proximity_max_bonus

def calc_momentum_bonus(team, config=DEFAULT_CONFIG):
    if "momentum" in team:
        return max(-1, min(1, team["momentum"])) * config.momentum_max_bonus
    recent_o = team.get("adj_o_recent", team.get("adj_o", 85))
    season_o = team.get("adj_o", 85)
    recent_d = team.get("adj_d_recent", team.get("adj_d", 112))
    season_d = team.get("adj_d", 112)
    m = ((recent_o - season_o) / 10 + (season_d - recent_d) / 10) / 2
    return max(-1, min(1, m)) * config.momentum_max_bonus

def calc_star_player_bonus(team, config=DEFAULT_CONFIG):
    return team.get("star_score", 0.0) * config.star_player_max_bonus


def calc_depth_bonus(team, config=DEFAULT_CONFIG):
    """Bonus for deep, balanced rosters (em_depth_score 0-1).

    Tournament survival across 6 games favours teams where multiple players
    contribute; a high depth_score captures both roster quality and balance.
    Falls back to 0 if EvanMiya data is unavailable.
    """
    depth = team.get("em_depth_score")
    if depth is None:
        return 0.0
    return depth * config.depth_max_bonus

def calc_size_bonus(team, config=DEFAULT_CONFIG):
    if "size_score" in team:
        return team["size_score"] * config.size_max_bonus
    two_pt = max(0, min(1, (team.get("two_pt_pct", 0.50) - 0.45) / 0.12))
    block = max(0, min(1, (team.get("block_rate", 8.0) - 5) / 10))
    orb = max(0, min(1, (team.get("orb_rate", 28.0) - 22) / 16))
    return (two_pt * 0.4 + block * 0.3 + orb * 0.3) * config.size_max_bonus

def calc_injury_penalty(team, config=DEFAULT_CONFIG):
    if "injury_level" in team:
        return team["injury_level"] * config.injury_penalty_per_level
    injuries = team.get("injuries", [])
    total = sum(i.get("severity",1) * i.get("importance",0.5) * config.injury_penalty_per_level for i in injuries)
    return min(total, 10.0)

def calc_luck_regression(team, config=DEFAULT_CONFIG):
    return -team.get("luck", 0.0) * config.luck_regression_factor * 10


def calc_win_pct_bonus(team, config=DEFAULT_CONFIG):
    """Small margin bonus for teams with high win % vs expectation (e.g. 0.85+ gets bonus)."""
    win_pct = team.get("win_pct")
    if win_pct is None or win_pct < 0.85:
        return 0.0
    # Scale from 0 at 0.85 to full bonus at 1.0
    excess = min(0.15, win_pct - 0.85)
    return (excess / 0.15) * config.win_pct_max_bonus


def calc_conf_bonus(team, config=DEFAULT_CONFIG):
    """Teams from stronger conferences get small bonus (conf_rating 1 = top tier)."""
    conf_rating = team.get("conf_rating")
    if conf_rating is None:
        return 0.0
    # Lower conf_rating = stronger conference (1 = top, 2 = mid, 3 = lower)
    if conf_rating <= 1.0:
        return config.conf_rating_max_bonus
    if conf_rating <= 2.0:
        return config.conf_rating_max_bonus * 0.5
    return 0.0

# ===========================================================================
# CORE PREDICTION
# ===========================================================================

def predict_game(team_a, team_b, game_site=None, config=DEFAULT_CONFIG):
    score_a, score_b, poss = _predict_base_score(team_a, team_b, config)
    base_margin = score_a - score_b

    # Possession, FT, and schedule-strength driven margin tweaks
    possession_margin = _calc_possession_edge_margin(team_a, team_b, poss, config)
    ft_margin = _calc_ft_edge_margin(team_a, team_b, base_margin, config)
    foul_rate_margin = _calc_foul_rate_margin(team_a, team_b, poss, config)
    sos_margin = _calc_sos_edge_margin(team_a, team_b, config)

    # All factor adjustments
    factor_names = ["experience", "coach", "pedigree", "preseason", "proximity",
                    "momentum", "star_player", "depth", "size", "injuries",
                    "luck_regression", "win_pct"]
    calc_fns = [
        lambda t: calc_experience_bonus(t, config),
        lambda t: calc_coach_bonus(t, config),
        lambda t: calc_pedigree_bonus(t, config),
        lambda t: calc_preseason_bonus(t, config),
        lambda t: calc_proximity_bonus(t, game_site, config),
        lambda t: calc_momentum_bonus(t, config),
        lambda t: calc_star_player_bonus(t, config),
        lambda t: calc_depth_bonus(t, config),
        lambda t: calc_size_bonus(t, config),
        lambda t: -calc_injury_penalty(t, config),
        lambda t: calc_luck_regression(t, config),
        lambda t: calc_win_pct_bonus(t, config),
    ]

    factors_a, factors_b = {}, {}
    for name, fn in zip(factor_names, calc_fns):
        factors_a[name] = fn(team_a)
        factors_b[name] = fn(team_b)

    factor_margin = sum(factors_a.values()) - sum(factors_b.values())
    extra_margin = possession_margin + ft_margin + foul_rate_margin + sos_margin
    adjusted_margin = base_margin + factor_margin + extra_margin

    # Variance (3PT volatility)
    vol = (calc_game_volatility(team_a, config) + calc_game_volatility(team_b, config)) / 2
    game_stdev = config.base_scoring_stdev * math.sqrt(poss / config.national_avg_tempo) * vol

    # Win probability
    if game_stdev == 0:
        eff_prob = 0.5
    else:
        eff_prob = 0.5 * (1.0 + math.erf((adjusted_margin / game_stdev) / math.sqrt(2)))

    # Blend with seed prior
    seed_a = team_a.get("seed", 8)
    seed_b = team_b.get("seed", 8)
    seed_prob = _seed_win_prob(seed_a, seed_b)
    final_prob = _blend_probs(eff_prob, seed_prob, config.seed_weight)

    return {
        "team_a": team_a["team"], "team_b": team_b["team"],
        "seed_a": seed_a, "seed_b": seed_b,
        "win_prob_a": round(final_prob, 4),
        "win_prob_b": round(1 - final_prob, 4),
        "predicted_score_a": round((score_a + factor_margin/2) * config.score_scale, 1),
        "predicted_score_b": round((score_b - factor_margin/2) * config.score_scale, 1),
        "predicted_margin": round(adjusted_margin, 1),
        "base_margin": round(base_margin, 1),
        "factor_margin": round(factor_margin, 1),
        "possession_margin": round(possession_margin, 2),
        "ft_margin": round(ft_margin, 2),
        "sos_margin": round(sos_margin, 2),
        "efficiency_prob": round(eff_prob, 4),
        "seed_prob": round(seed_prob, 4),
        "game_stdev": round(game_stdev, 1),
        "volatility": round(vol, 3),
        "factors_a": {k: round(v, 2) for k, v in factors_a.items()},
        "factors_b": {k: round(v, 2) for k, v in factors_b.items()},
    }

def _record_path_win(winner, loser):
    """Update winner's tournament path stats after beating loser."""
    opp_barthag = loser.get("barthag", 0.5)
    prev = winner.get("path_opponents_barthag", [])
    prev = list(prev) + [opp_barthag]
    winner["path_opponents_barthag"] = prev
    winner["path_avg_barthag"] = sum(prev) / len(prev)
    winner["path_rounds"] = len(prev)


def simulate_game(team_a, team_b, game_site=None, config=DEFAULT_CONFIG):
    result = predict_game(team_a, team_b, game_site, config=config)
    winner = team_a if random.random() < result["win_prob_a"] else team_b
    loser = team_b if winner is team_a else team_a
    _record_path_win(winner, loser)
    return winner

# ===========================================================================
# HELPERS
# ===========================================================================

def _predict_base_score(team_a, team_b, config):
    """Compute base scores from blended efficiency estimates.

    Priority: Torvik adj_o/adj_d is always the primary source.
    Secondary blend order: EvanMiya adj_o/adj_d (from player CSV context) >
    KenPom > Torvik alone. The em_adj_o_weight config param controls how much
    EvanMiya efficiency shifts the base score (0 = Torvik only, 1 = EvanMiya only).
    """
    avg = config.national_avg_efficiency
    w_em = max(0.0, min(1.0, config.em_adj_o_weight))
    w_torvik = 1.0 - w_em

    def _blend(team, torvik_key, em_key, kp_key):
        base = team.get(torvik_key, avg)
        # EvanMiya efficiency estimate (same scale as Torvik)
        em = team.get(em_key)
        if em and em > 50:
            return w_torvik * base + w_em * em
        # Fall back to KenPom if no EvanMiya
        kp = team.get(kp_key)
        if kp and kp > 50:
            return 0.7 * base + 0.3 * kp
        return base

    ta = team_a.get("adj_tempo", config.national_avg_tempo)
    tb = team_b.get("adj_tempo", config.national_avg_tempo)
    poss = (ta * tb) / config.national_avg_tempo

    off_a = _blend(team_a, "adj_o", "em_adj_o", "kp_adj_o")
    def_b = _blend(team_b, "adj_d", "em_adj_d", "kp_adj_d")
    off_b = _blend(team_b, "adj_o", "em_adj_o", "kp_adj_o")
    def_a = _blend(team_a, "adj_d", "em_adj_d", "kp_adj_d")

    eff_a = (off_a * def_b) / avg
    eff_b = (off_b * def_a) / avg
    return eff_a * poss / 100, eff_b * poss / 100, poss


def _calc_possession_edge_margin(team_a, team_b, poss, config):
    """Approximate margin swing from ORB% and TO% differences."""
    to_a = team_a.get("to_rate")
    to_b = team_b.get("to_rate")
    orb_a = team_a.get("orb_rate")
    orb_b = team_b.get("orb_rate")
    if to_a is None or to_b is None or orb_a is None or orb_b is None:
        return 0.0
    # Normalize if values look like 0-100 instead of 0-1
    if to_a > 2 or to_b > 2:
        to_a, to_b = to_a / 100.0, to_b / 100.0
    if orb_a > 2 or orb_b > 2:
        orb_a, orb_b = orb_a / 100.0, orb_b / 100.0
    # Net extra possessions per 100 trips for A vs B
    net_orb_edge = (orb_a - orb_b)
    net_to_edge = (to_b - to_a)  # fewer TOs = more possessions
    net_edge = net_orb_edge + net_to_edge
    # Translate to points using average offensive efficiency
    avg_eff = config.national_avg_efficiency / 100.0
    margin = net_edge * avg_eff * (poss / config.national_avg_tempo)
    # Clamp to reasonable bounds
    return max(-config.possession_edge_max_bonus, min(config.possession_edge_max_bonus, margin))


def _calc_ft_edge_margin(team_a, team_b, base_margin, config):
    """FT% only matters meaningfully in close games."""
    ft_a = team_a.get("ft_pct")
    ft_b = team_b.get("ft_pct")
    if ft_a is None or ft_b is None:
        return 0.0
    if ft_a > 2 or ft_b > 2:
        ft_a, ft_b = ft_a / 100.0, ft_b / 100.0
    diff = ft_a - ft_b
    # Only apply when game is reasonably close
    close_factor = max(0.0, 1.0 - min(abs(base_margin) / 8.0, 1.0))
    raw = diff * config.ft_clutch_max_bonus * close_factor
    return raw


def _calc_path_quality_margin(team_a, team_b, config):
    """Path quality is tracked (via _record_path_win) but not applied as an in-game
    penalty here. The signal is available for bracket optimization / reporting use,
    but the formula for translating it into a point spread would be speculative.
    """
    return 0.0


def _calc_foul_rate_margin(team_a, team_b, poss, config):
    """Margin from differential free throw attempt rate (foul drawing vs committing).

    ft_rate (FTR = FTA/FGA) measures how often a team gets to the line per shot.
    A team with higher FTR draws more fouls → more FTA per game → scoring advantage.
    This is game-length persistent, unlike ft_pct which is only clutch-relevant.
    """
    ftr_a = team_a.get("ft_rate")
    ftr_b = team_b.get("ft_rate")
    if ftr_a is None or ftr_b is None:
        return 0.0
    # Normalize if stored as 0-100
    if ftr_a > 2 or ftr_b > 2:
        ftr_a, ftr_b = ftr_a / 100.0, ftr_b / 100.0
    # Net extra FTAs per game: ~0.45 FGAs per possession, FT avg ~0.72 pct, 2 shots per trip
    avg_fga_per_poss = 0.45
    avg_ft_pct = 0.72
    net_ft_pts = (ftr_a - ftr_b) * avg_fga_per_poss * poss * 2 * avg_ft_pct
    return max(-config.ft_foul_rate_max_bonus, min(config.ft_foul_rate_max_bonus, net_ft_pts))


def _calc_sos_edge_margin(team_a, team_b, config):
    """Schedule-strength / opponent-quality edge.

    Prefers em_opponent_adjust (EvanMiya Bayesian SOS) when available;
    falls back to Torvik raw sos. em_opponent_adjust measures how much a
    team's BPR shifts when fully adjusting for opponent quality — a more
    stable signal than raw SOS for tournament prediction.
    """
    opp_a = team_a.get("em_opponent_adjust")
    opp_b = team_b.get("em_opponent_adjust")
    if opp_a is not None and opp_b is not None:
        diff = opp_a - opp_b
        # Typical practical range: ±30; scale to ±max_bonus
        margin = diff * (config.em_opp_adjust_max_bonus / 30.0)
        return max(-config.em_opp_adjust_max_bonus, min(config.em_opp_adjust_max_bonus, margin))
    sos_a = team_a.get("sos")
    sos_b = team_b.get("sos")
    if sos_a is None or sos_b is None:
        return 0.0
    diff = sos_a - sos_b
    margin = diff * 0.5
    return max(-config.sos_max_bonus, min(config.sos_max_bonus, margin))

def _seed_win_prob(seed_a, seed_b):
    return 1.0 / (1.0 + math.exp(-0.145 * (seed_b - seed_a)))

def _blend_probs(p1, p2, w2):
    def logit(p): return math.log(max(0.001, min(0.999, p)) / (1 - max(0.001, min(0.999, p))))
    def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
    return sigmoid((1 - w2) * logit(p1) + w2 * logit(p2))

def _haversine(lat1, lon1, lat2, lon2):
    R = 3959
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# ===========================================================================
# ENRICHMENT — Auto-fill intangibles from built-in databases
# ===========================================================================

COACH_SCORES = {
    "Bill Self": 0.95, "Mark Few": 0.85, "Scott Drew": 0.80,
    "Dan Hurley": 0.90, "John Calipari": 0.85, "Tom Izzo": 0.90,
    "Kelvin Sampson": 0.75, "Rick Barnes": 0.70, "Bruce Pearl": 0.75,
    "Jon Scheyer": 0.60, "Matt Painter": 0.70, "Nate Oats": 0.65,
    "Rick Pitino": 0.85, "Greg Gard": 0.55, "Jerome Tang": 0.40,
    "Dusty May": 0.50, "TJ Otzelberger": 0.45, "Brad Underwood": 0.45,
    "Eric Musselman": 0.60, "Mick Cronin": 0.55, "Chris Beard": 0.50,
    "Shaka Smart": 0.50, "Dennis Gates": 0.35, "Lamont Paris": 0.30,
}

PEDIGREE = {
    "Kansas": 0.95, "Duke": 0.95, "North Carolina": 0.95, "Kentucky": 0.95,
    "UCLA": 0.90, "UConn": 0.90, "Villanova": 0.85, "Indiana": 0.80,
    "Michigan St": 0.85, "Louisville": 0.80, "Syracuse": 0.75, "Ohio St": 0.70,
    "Michigan": 0.70, "Florida": 0.75, "Arizona": 0.70, "Georgetown": 0.65,
    "Wisconsin": 0.65, "Gonzaga": 0.65, "Purdue": 0.65, "Virginia": 0.65,
    "Baylor": 0.60, "Houston": 0.60, "Alabama": 0.55, "Tennessee": 0.55,
    "Arkansas": 0.60, "Iowa St": 0.50, "Marquette": 0.55, "Xavier": 0.50,
    "Creighton": 0.45, "Memphis": 0.55, "St. John's": 0.55, "Cincinnati": 0.55,
    "Maryland": 0.55, "Oregon": 0.50, "Texas Tech": 0.45, "Clemson": 0.35,
    "Auburn": 0.50, "San Diego St": 0.40, "BYU": 0.35,
}

def enrich_team(team):
    t = dict(team)
    if "coach_tourney_score" not in t:
        t["coach_tourney_score"] = COACH_SCORES.get(t.get("coach",""), 0.3)
    if "pedigree_score" not in t:
        t["pedigree_score"] = PEDIGREE.get(t["team"], 0.15)
    if "three_rate" not in t:
        t["three_rate"] = 0.35
    if "three_pct" not in t:
        t["three_pct"] = t.get("three_pt_pct", 34.0)
        if t["three_pct"] > 1.0:
            t["three_pct"] = t["three_pct"] / 100.0
    if "experience" not in t:
        t["experience"] = 0.5
    # Normalize two_pt_pct: engine expects 0-1, Torvik adv data gives percentage
    if "two_pt_pct" in t and t["two_pt_pct"] > 1.0:
        t["two_pt_pct"] = t["two_pt_pct"] / 100.0
    # Map blk_rate → block_rate (Torvik adv field name)
    if "block_rate" not in t and "blk_rate" in t:
        t["block_rate"] = t["blk_rate"]
    return t

# ===========================================================================
# TOURNAMENT SIMULATION
# ===========================================================================

FIRST_ROUND_MATCHUPS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
REGIONS = ["South", "East", "Midwest", "West"]

def simulate_region(teams_by_seed, config=DEFAULT_CONFIG):
    enriched = {s: enrich_team(t) for s, t in teams_by_seed.items()}
    matchups = [(enriched[a], enriched[b]) for a, b in FIRST_ROUND_MATCHUPS if a in enriched and b in enriched]
    results = {"Round of 64": [], "Round of 32": [], "Sweet 16": [], "Elite 8": []}

    def play_round(teams, round_name):
        winners = []
        for i in range(0, len(teams), 2):
            w = simulate_game(teams[i], teams[i+1], config=config)
            results[round_name].append({
                "team_a": teams[i]["team"], "seed_a": teams[i]["seed"],
                "team_b": teams[i+1]["team"], "seed_b": teams[i+1]["seed"],
                "winner": w["team"]
            })
            winners.append(w)
        return winners

    r64_winners = []
    for a, b in matchups:
        w = simulate_game(a, b, config=config)
        results["Round of 64"].append({"team_a": a["team"], "seed_a": a["seed"],
            "team_b": b["team"], "seed_b": b["seed"], "winner": w["team"]})
        r64_winners.append(w)

    r32_winners = play_round(r64_winners, "Round of 32")
    s16_winners = play_round(r32_winners, "Sweet 16")
    e8_winners = play_round(s16_winners, "Elite 8")
    return e8_winners[0], results

def simulate_tournament(bracket, config=DEFAULT_CONFIG):
    results = {}
    final_four = []
    for region in REGIONS:
        if region not in bracket: continue
        winner, rr = simulate_region(bracket[region], config)
        results[region] = rr
        final_four.append((region, winner))

    results["Final Four"] = []
    champ_teams = []
    for i, j in [(0,1),(2,3)]:
        if i < len(final_four) and j < len(final_four):
            _, a = final_four[i]; _, b = final_four[j]
            w = simulate_game(a, b, config=config)
            results["Final Four"].append({"team_a": a["team"], "seed_a": a["seed"],
                "team_b": b["team"], "seed_b": b["seed"], "winner": w["team"]})
            champ_teams.append(w)

    results["Championship"] = []
    if len(champ_teams) == 2:
        w = simulate_game(champ_teams[0], champ_teams[1], config=config)
        results["Championship"].append({"team_a": champ_teams[0]["team"], "seed_a": champ_teams[0]["seed"],
            "team_b": champ_teams[1]["team"], "seed_b": champ_teams[1]["seed"], "winner": w["team"]})
        return w, results
    return None, results

def run_monte_carlo(bracket, config=DEFAULT_CONFIG):
    counts = {k: defaultdict(int) for k in ["champ","ff","e8","s16","r32"]}
    game_results = defaultdict(lambda: defaultdict(int))

    print(f"Running {config.num_sims:,} simulations...")
    print(f"  Factors: efficiency, seed, experience, coach, pedigree, preseason,")
    print(f"           momentum, star player, size, injuries, luck, 3PT volatility, proximity")

    for sim in range(config.num_sims):
        if (sim+1) % 2000 == 0:
            print(f"  {sim+1:,}/{config.num_sims:,}")
        champ, results = simulate_tournament(bracket, config)
        if champ: counts["champ"][champ["team"]] += 1

        for rname, rdata in results.items():
            if rname in REGIONS:
                for rd, games in rdata.items():
                    for g in games:
                        key = f"{g['seed_a']}{g['team_a']}_vs_{g['seed_b']}{g['team_b']}"
                        game_results[key][g["winner"]] += 1
                        rmap = {"Round of 64":"r32","Round of 32":"s16","Sweet 16":"e8","Elite 8":"ff"}
                        if rd in rmap: counts[rmap[rd]][g["winner"]] += 1
            elif rname in ("Final Four","Championship"):
                for g in rdata:
                    key = f"{rname}_{g['team_a']}_vs_{g['team_b']}"
                    game_results[key][g["winner"]] += 1

    def probs(c): return {k: round(v/config.num_sims, 4) for k, v in sorted(c.items(), key=lambda x:-x[1])}

    gp = {}
    for m, oc in game_results.items():
        tot = sum(oc.values())
        gp[m] = {t: round(c/tot, 4) for t, c in sorted(oc.items(), key=lambda x:-x[1])}

    return {
        "num_simulations": config.num_sims,
        "champion_probs": probs(counts["champ"]),
        "final_four_probs": probs(counts["ff"]),
        "elite_eight_probs": probs(counts["e8"]),
        "sweet_sixteen_probs": probs(counts["s16"]),
        "round_of_32_probs": probs(counts["r32"]),
        "game_probs": gp,
    }

def analyze_matchup_perspectives(team_a, team_b, game_site=None, config=DEFAULT_CONFIG):
    """Full matchup analysis with multiple analytical perspectives (debate-style)."""
    a, b = enrich_team(team_a), enrich_team(team_b)
    result = predict_game(a, b, game_site, config=config)
    perspectives = []

    # Efficiency Nerd
    margin = result["base_margin"]
    eff_prob = result["efficiency_prob"]
    if abs(margin) >= 8:
        pick = "A" if margin > 0 else "B"
        team = result["team_a"] if margin > 0 else result["team_b"]
        perspectives.append({
            "name": "Efficiency Nerd",
            "argument": f"{team}'s offense vs the other's defense gives a {abs(margin):.1f} pt expected margin. Tempo-adjusted efficiency says clear edge.",
            "pick": pick,
            "confidence": "high",
        })
    elif abs(margin) >= 3:
        pick = "A" if margin > 0 else "B"
        team = result["team_a"] if margin > 0 else result["team_b"]
        perspectives.append({
            "name": "Efficiency Nerd",
            "argument": f"{team} has the efficiency edge ({abs(margin):.1f} pt margin). I lean that way.",
            "pick": pick,
            "confidence": "medium",
        })
    else:
        perspectives.append({
            "name": "Efficiency Nerd",
            "argument": f"Virtually even on efficiency (margin {margin:.1f}). Could go either way.",
            "pick": "toss-up",
            "confidence": "low",
        })

    # Seed Historian
    seed_a, seed_b = result["seed_a"], result["seed_b"]
    seed_prob = result["seed_prob"]
    hist_pick = "A" if seed_prob >= 0.55 else ("B" if seed_prob <= 0.45 else "toss-up")
    hist_team = result["team_a"] if seed_prob >= 0.5 else result["team_b"]
    pct = seed_prob * 100 if seed_prob >= 0.5 else (1 - seed_prob) * 100
    perspectives.append({
        "name": "Seed Historian",
        "argument": f"({seed_a}) vs ({seed_b}): history says the higher seed wins ~{pct:.0f}% of the time. I pick {hist_team}.",
        "pick": hist_pick,
        "confidence": "high" if abs(seed_prob - 0.5) > 0.2 else "medium",
    })

    # Intangibles Believer
    factor_margin = result["factor_margin"]
    if abs(factor_margin) >= 2:
        pick = "A" if factor_margin > 0 else "B"
        team = result["team_a"] if factor_margin > 0 else result["team_b"]
        perspectives.append({
            "name": "Intangibles Believer",
            "argument": f"Coach, pedigree, experience, and momentum favor {team}. March is about more than raw efficiency.",
            "pick": pick,
            "confidence": "medium",
        })
    else:
        perspectives.append({
            "name": "Intangibles Believer",
            "argument": "Intangibles are a wash. No clear edge in coaching or pedigree.",
            "pick": "toss-up",
            "confidence": "low",
        })

    # Variance Advocate
    vol = result["volatility"]
    underdog_prob = result["win_prob_b"]
    seed_high = min(seed_a, seed_b)
    seed_low = max(seed_a, seed_b)
    underdog = result["team_b"] if a.get("seed", 8) < b.get("seed", 8) else result["team_a"]
    if vol > 1.1 and underdog_prob >= 0.28:
        perspectives.append({
            "name": "Variance Advocate",
            "argument": f"High 3PT volatility and a live underdog ({underdog}). Upset alert — the underdog has a real shot.",
            "pick": "B" if underdog == result["team_b"] else "A",
            "confidence": "medium",
        })
    elif vol < 0.92:
        perspectives.append({
            "name": "Variance Advocate",
            "argument": "Both teams score inside; lower variance. Favor the steadier favorite.",
            "pick": "A" if result["win_prob_a"] >= 0.5 else "B",
            "confidence": "low",
        })
    else:
        perspectives.append({
            "name": "Variance Advocate",
            "argument": f"Standard variance. Underdog at {underdog_prob*100:.0f}% — not crazy to pick them.",
            "pick": "toss-up",
            "confidence": "low",
        })

    # Consensus / Model
    final_prob = result["win_prob_a"]
    consensus_pick = "A" if final_prob >= 0.52 else ("B" if final_prob <= 0.48 else "toss-up")
    perspectives.append({
        "name": "Consensus / Model",
        "argument": f"Blended model: {final_prob*100:.0f}% {result['team_a']}, {result['predicted_score_a']:.0f}-{result['predicted_score_b']:.0f}.",
        "pick": consensus_pick,
        "confidence": "high" if abs(final_prob - 0.5) > 0.15 else "medium",
    })

    result["perspectives"] = perspectives
    return result


def analyze_matchup(team_a, team_b, game_site=None, config=DEFAULT_CONFIG):
    """Detailed matchup analysis with narratives, perspectives, spread, and historical context."""
    a, b = enrich_team(team_a), enrich_team(team_b)
    result = analyze_matchup_perspectives(a, b, game_site, config=config)
    narratives = []

    margin = result["base_margin"]
    if abs(margin) > 3:
        better = result["team_a"] if margin > 0 else result["team_b"]
        narratives.append(f"{better} has a clear efficiency edge ({abs(margin):.1f} pt expected margin)")

    for label, key, threshold, template_pos, template_neg in [
        ("experience", "experience", 0.5, "{t} has significantly more tournament experience", None),
        ("coaching", "coach", 0.3, "{t} has the coaching edge in March", None),
        ("pedigree", "pedigree", 0.3, "{t}'s tournament pedigree gives them a historical edge", None),
        ("star player", "star_player", 0.5, "{t} has the better go-to player for crunch time", None),
        ("momentum", "momentum", 0.5, "{t} enters the tournament on a hot streak", None),
    ]:
        diff = result["factors_a"][key] - result["factors_b"][key]
        if abs(diff) > threshold:
            t = result["team_a"] if diff > 0 else result["team_b"]
            narratives.append(template_pos.format(t=t))

    if result["volatility"] > 1.1:
        narratives.append("High 3PT dependency makes this a volatile matchup — upset more likely")
    elif result["volatility"] < 0.92:
        narratives.append("Both teams score inside — expect a grind-it-out game with fewer surprises")

    for side, label in [("a", result["team_a"]), ("b", result["team_b"])]:
        inj = result[f"factors_{side}"]["injuries"]
        if inj < -1:
            narratives.append(f"{label} is dealing with key injuries ({inj:.1f} pts impact)")

    luck_diff = result["factors_a"]["luck_regression"] - result["factors_b"]["luck_regression"]
    if abs(luck_diff) > 0.5:
        unlucky = result["team_a"] if luck_diff > 0 else result["team_b"]
        lucky = result["team_b"] if luck_diff > 0 else result["team_a"]
        narratives.append(f"{lucky} may have overperformed their true level; regression risk")

    # Historical seed matchup data
    hist = get_seed_matchup_history(result["seed_a"], result["seed_b"])
    if hist and hist["total"] >= 3:
        sa, sb = result["seed_a"], result["seed_b"]
        better_seed = min(sa, sb)
        worse_seed = max(sa, sb)
        w, l = hist["higher_seed_wins"], hist["lower_seed_wins"]
        pct = hist["higher_seed_win_pct"] * 100
        result["historical_record"] = f"{w}-{l}"
        result["historical_win_pct"] = round(hist["higher_seed_win_pct"], 3)
        result["historical_upset_rate"] = round(hist["upset_rate"], 3)
        result["historical_avg_margin"] = round(hist["avg_margin"], 1)
        narratives.append(f"({better_seed}) vs ({worse_seed}) seeds: {w}-{l} ({pct:.0f}%) historically, avg margin {hist['avg_margin']:.1f} pts")

    # Projected spread and confidence
    result["projected_spread"] = round(abs(result["predicted_margin"]), 1)
    result["confidence"] = _confidence_tier(result["win_prob_a"])
    result["upset_rating"] = _upset_rating(
        result["seed_a"], result["seed_b"],
        result["win_prob_a"], result.get("volatility", 1.0))
    result["variability"] = _variability_label(
        result.get("volatility", 1.0),
        abs(result["seed_a"] - result["seed_b"]),
        result["win_prob_a"])

    # Pick
    result["pick"] = result["team_a"] if result["win_prob_a"] >= 0.5 else result["team_b"]
    result["pick_seed"] = result["seed_a"] if result["win_prob_a"] >= 0.5 else result["seed_b"]

    # SOS / possession / FT edges as insights
    sos_m = result.get("sos_margin", 0)
    if abs(sos_m) > 1.0:
        stronger = result["team_a"] if sos_m > 0 else result["team_b"]
        narratives.append(f"{stronger} played a significantly tougher schedule")

    poss_m = result.get("possession_margin", 0)
    if abs(poss_m) > 1.0:
        better = result["team_a"] if poss_m > 0 else result["team_b"]
        narratives.append(f"{better} has a rebounding and turnover margin advantage")

    result["narratives"] = narratives
    result["upset_alert"] = result["win_prob_b"] > 0.30 and result["seed_a"] < result["seed_b"]
    result["key_factors"] = narratives[:4] if narratives else [p["name"] for p in result["perspectives"][:2]]
    return result

_NAME_ALIASES = {
    "uconn": "connecticut",
    "unc": "north carolina",
    "miami fl": "miami",
    "miami fla": "miami",
    "miami (fl)": "miami",
    "miami (fla)": "miami",
    "miami ohio": "miami oh",
    "miami (ohio)": "miami oh",
    "ucf": "central florida",
    "usc": "southern california",
    "lsu": "louisiana state",
    "ole miss": "mississippi",
    "pitt": "pittsburgh",
    "umass": "massachusetts",
    "vcu": "virginia commonwealth",
    "smu": "southern methodist",
    "byu": "brigham young",
    "tcu": "texas christian",
    "utep": "texas el paso",
    "unlv": "nevada las vegas",
    "uab": "alabama birmingham",
    "niu": "northern illinois",
    "siu": "southern illinois",
    "etsu": "east tennessee state",
    "sfa": "stephen f austin",
    "saint marys": "st marys",
    "saint marys ca": "st marys",
    "st marys ca": "st marys",
    "st marys (ca)": "st marys",
    "saint josephs": "st josephs",
    "saint louis": "st louis",
    "saint johns": "st johns",
    "saint peters": "st peters",
    "saint bonaventure": "st bonaventure",
    "uc irvine": "irvine",
    "uc santa barbara": "uc santa barb",
    "uc davis": "uc davis",
    "ucsan diego": "uc san diego",   # UC-San Diego -> UC San Diego (Big West)
    "long island university": "long island",
    "liu": "long island",
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
    # State / St. variants — bracket says "State", Torvik says "St."
    "iowa state": "iowa st",
    "michigan state": "michigan st",
    "ohio state": "ohio st",
    "florida state": "florida st",
    "kansas state": "kansas st",
    "colorado state": "colorado st",
    "utah state": "utah st",
    "penn state": "penn st",
    "oregon state": "oregon st",
    "mississippi state": "mississippi st",
    "boise state": "boise st",
    "arizona state": "arizona st",
    "oklahoma state": "oklahoma st",
    "washington state": "washington st",
    "san diego state": "san diego st",
    "fresno state": "fresno st",
    "norfolk state": "norfolk st",
    "alabama state": "alabama st",
    "mcneese state": "mcneese st",
    "morehead state": "morehead st",
    "montana state": "montana st",
    "cleveland state": "cleveland st",
    "wright state": "wright st",
    "kent state": "kent st",
    "wichita state": "wichita st",
    "murray state": "murray st",
    "south dakota state": "south dakota st",
    "north dakota state": "north dakota st",
    "jacksonville state": "jacksonville st",
    "appalachian state": "appalachian st",
    "kennesaw state": "kennesaw st",
    # St. John's variants
    "st johns (ny)": "st johns",
    "saint johns (ny)": "st johns",
    "st johns ny": "st johns",
    # SIU variants
    "siuedwardsville": "siu edwardsville",
    "siu edwardsville": "siu edwardsville",
    # Omaha
    "omaha": "nebraska omaha",
    "nebraskoomaha": "nebraska omaha",
}


def _normalize_team_for_match(name):
    """Normalize team name for lookup (lowercase, collapse spaces, remove punctuation)."""
    if not name or not isinstance(name, str):
        return ""
    import re
    s = " ".join(name.strip().split()).lower()
    s = re.sub(r"['\-\.\u2013\u2014]", "", s)  # ASCII and unicode dashes
    s = re.sub(r"\(([^)]+)\)", r"\1", s)       # remove parentheses, keep contents: (MD) -> MD
    s = re.sub(r"\s+", " ", s).strip()
    return _NAME_ALIASES.get(s, s)


def load_teams_merged(data_dir, year):
    """Load teams_merged_YYYY.json or torvik_YYYY.json for bracket enrichment.
    Returns dict keyed by normalized team name. Skips Unknown_N entries.
    """
    for fname in (f"teams_merged_{year}.json", f"torvik_{year}.json"):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, list):
            out = {}
            for r in data:
                t = r.get("team", "")
                if not t or str(t).startswith("Unknown"):
                    continue
                key = _normalize_team_for_match(t)
                if key:
                    out[key] = dict(r)
            return out
        return {}
    return {}


_BAD_DEFAULTS = {"adj_o": 85, "adj_d": 112, "adj_tempo": 64, "barthag": 0.05}


def enrich_bracket_with_teams(bracket, teams_merged):
    """Overwrite bracket team stats from teams_merged when we have real data.
    Teams not found in teams_merged are reset to bad defaults so corrupted
    data baked into bracket JSONs doesn't persist.
    Returns number of teams enriched.
    """
    if not teams_merged:
        return 0
    enriched = 0
    enriched_names = set()
    for region in bracket.values():
        for team_obj in region.values():
            tname = team_obj.get("team", "")
            if not tname or str(tname).startswith("TEAM_") or str(tname).startswith("Unknown"):
                continue
            key = _normalize_team_for_match(tname)
            if key not in teams_merged:
                # Fallback: only match when bracket key is substring of teams_merged key
                # (prevents "michigan st" from wrongly matching "michigan")
                for k, v in teams_merged.items():
                    if len(key) <= len(k) and key in k and len(key) >= len(k) * 0.65:
                        key = k
                        break
                else:
                    continue
            merged = teams_merged[key]
            adj_o = merged.get("adj_o")
            adj_d = merged.get("adj_d")
            team_obj["adj_o"] = adj_o if adj_o is not None else _BAD_DEFAULTS["adj_o"]
            team_obj["adj_d"] = adj_d if adj_d is not None else _BAD_DEFAULTS["adj_d"]
            team_obj["adj_tempo"] = merged.get("adj_tempo") or _BAD_DEFAULTS["adj_tempo"]
            team_obj["barthag"] = merged.get("barthag") if merged.get("barthag") is not None else _BAD_DEFAULTS["barthag"]
            for k in ("to_rate", "orb_rate", "ft_pct", "ft_rate", "sos", "luck",
                      "kp_adj_o", "kp_adj_d", "star_score",
                      "three_rate", "three_pct", "two_pt_pct", "block_rate",
                      "wab", "elite_sos", "qual_o", "qual_d", "qual_barthag",
                      "conf_adj_o", "conf_adj_d", "win_pct", "conf_win_pct", "conf_rating",
                      "em_o_rate", "em_d_rate", "em_rel_rating", "em_roster_rank",
                      "em_tempo", "top_player", "top_player_bpr"):
                if k in merged and merged[k] is not None:
                    team_obj[k] = merged[k]
            enriched += 1
            enriched_names.add(tname)

    for region in bracket.values():
        for team_obj in region.values():
            tname = team_obj.get("team", "")
            if not tname or tname in enriched_names:
                continue
            if str(tname).startswith("TEAM_") or str(tname).startswith("Unknown"):
                continue
            cur_o = team_obj.get("adj_o")
            cur_d = team_obj.get("adj_d")
            has_real_stats = (cur_o is not None and cur_d is not None
                             and not (cur_o == _BAD_DEFAULTS["adj_o"]
                                      and cur_d == _BAD_DEFAULTS["adj_d"]))
            if not has_real_stats:
                for stat, val in _BAD_DEFAULTS.items():
                    team_obj[stat] = val

    return enriched


# ===========================================================================
# HISTORICAL SEED STATS (loaded from results_all.json)
# ===========================================================================

_SEED_MATCHUP_CACHE = None

def load_historical_seed_stats(data_dir=None):
    """Load results_all.json and compute per-seed-matchup win rates."""
    global _SEED_MATCHUP_CACHE
    if _SEED_MATCHUP_CACHE is not None:
        return _SEED_MATCHUP_CACHE

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    path = os.path.join(data_dir, "results_all.json")
    if not os.path.isfile(path):
        _SEED_MATCHUP_CACHE = {}
        return _SEED_MATCHUP_CACHE

    with open(path) as f:
        data = json.load(f)

    stats = {}
    for g in data.get("games", []):
        sa, sb = g["seed_a"], g["seed_b"]
        key = (min(sa, sb), max(sa, sb))
        if key not in stats:
            stats[key] = {"higher_seed_wins": 0, "lower_seed_wins": 0, "total": 0,
                          "margins": [], "upsets": 0}
        s = stats[key]
        s["total"] += 1
        s["margins"].append(g["margin"])
        winner_seed = g["seed_a"] if g["winner"] == g["team_a"] else g["seed_b"]
        if winner_seed == key[0]:
            s["higher_seed_wins"] += 1
        else:
            s["lower_seed_wins"] += 1
        if g.get("upset"):
            s["upsets"] += 1

    for key, s in stats.items():
        s["higher_seed_win_pct"] = s["higher_seed_wins"] / s["total"] if s["total"] else 0.5
        s["avg_margin"] = sum(s["margins"]) / len(s["margins"]) if s["margins"] else 0
        s["upset_rate"] = s["upsets"] / s["total"] if s["total"] else 0

    _SEED_MATCHUP_CACHE = stats
    return stats


def get_seed_matchup_history(seed_a, seed_b):
    """Get historical stats for a seed matchup. Returns dict or None."""
    stats = load_historical_seed_stats()
    key = (min(seed_a, seed_b), max(seed_a, seed_b))
    return stats.get(key)


_H2H_CACHE = {}

def get_head_to_head(team_a, team_b, data_dir=None, current_year=None):
    """Get head-to-head matchups: past tournaments and this season.

    Returns { past_tournament: [...], this_season: [...] } with rich game records
    for display and future ML use.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if current_year is None:
        current_year = 2026

    na = _normalize_team_for_match(team_a)
    nb = _normalize_team_for_match(team_b)
    if not na or not nb:
        return {"past_tournament": [], "this_season": []}
    cache_key = (min(na, nb), max(na, nb), current_year)
    if cache_key in _H2H_CACHE:
        return _H2H_CACHE[cache_key]

    past = []
    import glob
    for fpath in sorted(glob.glob(os.path.join(data_dir, "results_[0-9][0-9][0-9][0-9].json"))):
        m = re.search(r"results_(\d{4})\.json$", fpath)
        if not m:
            continue
        yr = int(m.group(1))
        if yr >= current_year:
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue
        for g in data.get("games", []):
            ga = _normalize_team_for_match(g.get("team_a", ""))
            gb = _normalize_team_for_match(g.get("team_b", ""))
            if not ga or not gb:
                continue
            pair = (min(ga, gb), max(ga, gb))
            if pair != (min(na, nb), max(na, nb)):
                continue
            past.append({
                "year": yr,
                "round": g.get("round"),
                "round_name": g.get("round_name", ""),
                "region": g.get("region", ""),
                "team_a": g["team_a"], "team_b": g["team_b"],
                "seed_a": g.get("seed_a"), "seed_b": g.get("seed_b"),
                "score_a": g.get("score_a"), "score_b": g.get("score_b"),
                "winner": g.get("winner"), "margin": g.get("margin", 0),
                "upset": g.get("upset", False),
                "game_type": "tournament",
            })
    past.sort(key=lambda x: -x["year"])

    this_season = []
    sg_path = os.path.join(data_dir, f"season_games_{current_year}.json")
    if os.path.isfile(sg_path):
        try:
            with open(sg_path) as f:
                sg_data = json.load(f)
        except Exception:
            pass
        else:
            games = sg_data if isinstance(sg_data, list) else sg_data.get("games", [])
            for g in games:
                ga = _normalize_team_for_match(g.get("team_a", ""))
                gb = _normalize_team_for_match(g.get("team_b", ""))
                if not ga or not gb:
                    continue
                pair = (min(ga, gb), max(ga, gb))
                if pair != (min(na, nb), max(na, nb)):
                    continue
                this_season.append({
                    "date": g.get("date", ""),
                    "team_a": g.get("team_a"), "team_b": g.get("team_b"),
                    "score_a": g.get("score_a"), "score_b": g.get("score_b"),
                    "winner": g.get("winner"),
                    "margin": abs((g.get("score_a") or 0) - (g.get("score_b") or 0)),
                    "location": g.get("location", ""),
                    "game_type": "regular_season",
                })
            this_season.sort(key=lambda x: x.get("date", ""))

    result = {"past_tournament": past, "this_season": this_season}
    _H2H_CACHE[cache_key] = result
    return result


# ===========================================================================
# BRACKET PICKS GENERATION
# ===========================================================================

def _confidence_tier(win_prob):
    """Classify win probability into a confidence tier."""
    p = max(win_prob, 1 - win_prob)
    if p >= 0.90:
        return "lock"
    elif p >= 0.75:
        return "strong"
    elif p >= 0.60:
        return "lean"
    else:
        return "tossup"


def _upset_rating(seed_a, seed_b, win_prob_a, volatility):
    """Compute upset likelihood score (0-1). Higher = more likely upset."""
    seed_diff = abs(seed_a - seed_b)
    if seed_diff == 0:
        return 0.0
    underdog_prob = min(win_prob_a, 1 - win_prob_a)
    hist = get_seed_matchup_history(seed_a, seed_b)
    hist_upset_rate = hist["upset_rate"] if hist else 0.3
    vol_factor = max(0, (volatility - 0.9) / 0.4)
    rating = 0.4 * underdog_prob + 0.3 * hist_upset_rate + 0.2 * vol_factor + 0.1 * min(seed_diff / 15, 1.0)
    return round(min(1.0, rating), 3)


def _variability_label(volatility, seed_diff, win_prob):
    """Classify game variability."""
    closeness = 1 - abs(win_prob - 0.5) * 2
    score = 0.4 * max(0, (volatility - 0.9) / 0.4) + 0.4 * closeness + 0.2 * min(seed_diff / 10, 1.0)
    if score >= 0.65:
        return "extreme"
    elif score >= 0.45:
        return "high"
    elif score >= 0.25:
        return "medium"
    return "low"


def _generate_insight(result, hist, team_a, team_b):
    """Generate a concise insight string for a matchup."""
    margin = result["predicted_margin"]
    prob_a = result["win_prob_a"]
    pick = team_a["team"] if prob_a >= 0.5 else team_b["team"]
    loser = team_b["team"] if prob_a >= 0.5 else team_a["team"]

    parts = []

    if abs(margin) >= 12:
        parts.append(f"{pick} dominates on efficiency ({abs(margin):.0f}-pt edge)")
    elif abs(margin) >= 5:
        parts.append(f"{pick} has a solid {abs(margin):.0f}-pt efficiency advantage")
    elif abs(margin) >= 2:
        parts.append(f"Slight edge to {pick} ({abs(margin):.1f} pts)")
    else:
        parts.append(f"Razor-thin margin — essentially a coin flip")

    if hist and hist["total"] >= 5:
        sa, sb = team_a["seed"], team_b["seed"]
        better = min(sa, sb)
        worse = max(sa, sb)
        pct = hist["higher_seed_win_pct"] * 100
        w = hist["higher_seed_wins"]
        l = hist["lower_seed_wins"]
        parts.append(f"{better}-seeds are {w}-{l} ({pct:.0f}%) vs {worse}-seeds historically")

    vol = result.get("volatility", 1.0)
    if vol > 1.1:
        parts.append(f"3PT-dependent offenses add variance")
    elif vol < 0.92:
        parts.append(f"Physical, inside-scoring teams reduce randomness")

    sos_m = result.get("sos_margin", 0)
    if abs(sos_m) > 1.0:
        stronger = team_a["team"] if sos_m > 0 else team_b["team"]
        parts.append(f"{stronger} played a tougher schedule")

    return ". ".join(parts) + "."


def _generate_key_factors(result, team_a, team_b):
    """Generate list of key factors driving the pick."""
    factors = []
    margin = result["predicted_margin"]
    if abs(margin) >= 3:
        better = team_a["team"] if margin > 0 else team_b["team"]
        factors.append(f"{better}: {abs(margin):.0f}-pt efficiency edge")

    sos_m = result.get("sos_margin", 0)
    if abs(sos_m) > 0.5:
        t = team_a["team"] if sos_m > 0 else team_b["team"]
        factors.append(f"{t}: stronger schedule")

    poss_m = result.get("possession_margin", 0)
    if abs(poss_m) > 0.5:
        t = team_a["team"] if poss_m > 0 else team_b["team"]
        factors.append(f"{t}: rebounding/turnover edge")

    fa, fb = result.get("factors_a", {}), result.get("factors_b", {})
    for f_name, label in [("coach", "coaching"), ("pedigree", "program pedigree")]:
        diff = fa.get(f_name, 0) - fb.get(f_name, 0)
        if abs(diff) > 0.5:
            t = team_a["team"] if diff > 0 else team_b["team"]
            factors.append(f"{t}: {label} advantage")

    vol = result.get("volatility", 1.0)
    if vol > 1.1:
        factors.append("High 3PT variance — upset risk")

    return factors[:4]


def _should_pick_upset(prob_a, seed_a, seed_b, aggression):
    """Decide whether to pick an upset based on aggression level."""
    if aggression <= 0:
        return prob_a >= 0.5
    seed_diff = abs(seed_a - seed_b)
    upset_boost = min(seed_diff / 15, 1.0) * 0.3
    if prob_a >= 0.5:
        adjusted = prob_a - aggression * upset_boost
    else:
        adjusted = prob_a + aggression * upset_boost
    return random.random() < adjusted


def _compute_upset_alert(seed_a, seed_b, projected_margin, win_prob_a, hist):
    """Flag games where projected margin deviates from what's expected for this seed matchup.

    Requires BOTH % difference AND absolute spread to flag — avoids skewing towards
    large or small spreads alone.
    """
    if seed_a == seed_b:
        return None

    higher_seed = min(seed_a, seed_b)
    lower_seed = max(seed_a, seed_b)
    seed_diff = lower_seed - higher_seed
    margin_favoring_higher = projected_margin if seed_a < seed_b else -projected_margin

    if hist and hist["total"] >= 3:
        win_pct = hist["higher_seed_win_pct"]
        abs_margin = hist["avg_margin"]
        expected_directional = abs_margin * (2 * win_pct - 1)
    else:
        expected_directional = seed_diff * 1.2

    base = max(abs(expected_directional), 1.0)
    gap = expected_directional - margin_favoring_higher
    gap_pct = gap / base if base else 0
    underdog_prob = 1 - max(win_prob_a, 1 - win_prob_a)

    # Require both % AND absolute pts: e.g. 40% tighter AND at least 5 pts
    if (gap_pct > 0.45 and gap > 5.0 and underdog_prob > 0.38 and seed_diff >= 3):
        return {
            "level": "strong",
            "reason": f"({higher_seed}) seed favored by only {margin_favoring_higher:.1f} pts vs expected {expected_directional:.1f} ({gap_pct*100:.0f}% tighter). Upset danger.",
            "icon": "\U0001F525",
            "badge_on_underdog": True,
        }
    elif (gap_pct > 0.35 and gap > 3.5 and underdog_prob > 0.33 and seed_diff >= 3):
        return {
            "level": "mild",
            "reason": f"Margin ({margin_favoring_higher:.1f}) {gap_pct*100:.0f}% tighter than typical ({expected_directional:.1f}) for {higher_seed}-{lower_seed} seeds.",
            "icon": "\u26A0\uFE0F",
            "badge_on_underdog": True,
        }
    excess = margin_favoring_higher - expected_directional
    excess_pct = excess / base if base else 0
    if excess_pct > 0.6 and excess > 8.0 and seed_diff >= 4:
        return {
            "level": "blowout",
            "reason": f"Projected margin ({margin_favoring_higher:.1f}) {excess_pct*100:.0f}% above the {higher_seed}-{lower_seed} seed norm ({expected_directional:.1f}). Dominant.",
            "icon": "\U0001F4AA",
            "badge_on_underdog": False,
        }

    return None


def _make_pick_dict(game_num, round_of, round_name, region, a, b, result, pick_team,
                    data_dir=None, year=None):
    """Build a pick dict from prediction result."""
    hist = get_seed_matchup_history(a["seed"], b["seed"])
    upset_alert = _compute_upset_alert(
        a["seed"], b["seed"], result["predicted_margin"],
        result["win_prob_a"], hist)
    margin = result["predicted_margin"]
    spread_amt = round(abs(margin), 1)
    fav_team = a["team"] if result["win_prob_a"] >= 0.5 else b["team"]
    dog_team = b["team"] if result["win_prob_a"] >= 0.5 else a["team"]
    h2h = get_head_to_head(a["team"], b["team"], data_dir=data_dir, current_year=year or 2026)
    d = {
        "game_num": game_num,
        "round": round_of,
        "round_name": round_name,
        "region": region,
        "team_a": a["team"], "seed_a": a["seed"],
        "team_b": b["team"], "seed_b": b["seed"],
        "pick": pick_team["team"],
        "pick_seed": pick_team["seed"],
        "win_prob": round(max(result["win_prob_a"], result["win_prob_b"]), 4),
        "projected_spread": spread_amt,
        "spread_fav": f"{fav_team} -{spread_amt}",
        "spread_dog": f"{dog_team} +{spread_amt}",
        "projected_score": f"{result['predicted_score_a']:.0f}-{result['predicted_score_b']:.0f}",
        "confidence": _confidence_tier(result["win_prob_a"]),
        "upset_rating": _upset_rating(a["seed"], b["seed"], result["win_prob_a"], result.get("volatility", 1.0)),
        "variability": _variability_label(result.get("volatility", 1.0), abs(a["seed"] - b["seed"]), result["win_prob_a"]),
        "key_factors": _generate_key_factors(result, a, b),
        "insight": _generate_insight(result, hist, a, b),
        "historical": f"{hist['higher_seed_wins']}-{hist['lower_seed_wins']}" if hist else None,
        "head_to_head": h2h,
    }
    if upset_alert:
        d["upset_alert"] = upset_alert
    return d


def generate_bracket_picks(bracket, config=DEFAULT_CONFIG, upset_aggression=0.0, quadrant_order=None,
                          data_dir=None, year=None):
    """Generate a complete 63-game bracket with analysis for every pick.

    Args:
        bracket: dict of region -> {seed: team_dict}
        config: ModelConfig
        upset_aggression: 0.0 (always pick favorite) to 1.0 (chaos mode)
        quadrant_order: [TL, TR, BR, BL] region names for FF pairing
        data_dir: path to data dir for head-to-head lookup
        year: current bracket year for head-to-head

    Returns dict with 'picks', 'champion', 'final_four', 'biggest_upsets', 'most_uncertain_games'.
    """
    if quadrant_order is None:
        quadrant_order = REGIONS[:4]
    _h2h_kw = {"data_dir": data_dir, "year": year or 2026}
    picks = []
    game_num = 0

    enriched_bracket = {}
    for region in REGIONS:
        if region not in bracket:
            continue
        enriched_bracket[region] = {s: enrich_team(t) for s, t in bracket[region].items()}

    region_winners = {}

    for region in REGIONS:
        if region not in enriched_bracket:
            continue
        teams = enriched_bracket[region]

        r64_winners = []
        for seed_a, seed_b in FIRST_ROUND_MATCHUPS:
            a = teams.get(seed_a)
            b = teams.get(seed_b)
            if not a or not b:
                continue
            game_num += 1
            result = predict_game(a, b, config=config)
            pick_a = _should_pick_upset(result["win_prob_a"], a["seed"], b["seed"], upset_aggression)
            pick_team = a if pick_a else b
            picks.append(_make_pick_dict(game_num, 64, "Round of 64", region, a, b, result, pick_team, **_h2h_kw))
            r64_winners.append(pick_team)

        def _play_round(teams_in, round_of, round_name):
            winners = []
            for i in range(0, len(teams_in), 2):
                if i + 1 >= len(teams_in):
                    winners.append(teams_in[i])
                    continue
                a, b = teams_in[i], teams_in[i + 1]
                nonlocal game_num
                game_num += 1
                result = predict_game(a, b, config=config)
                pick_a = _should_pick_upset(result["win_prob_a"], a["seed"], b["seed"], upset_aggression)
                pick_team = a if pick_a else b
                picks.append(_make_pick_dict(game_num, round_of, round_name, region, a, b, result, pick_team, **_h2h_kw))
                winners.append(pick_team)
            return winners

        r32_winners = _play_round(r64_winners, 32, "Round of 32")
        s16_winners = _play_round(r32_winners, 16, "Sweet 16")
        e8_winners = _play_round(s16_winners, 8, "Elite 8")
        if e8_winners:
            region_winners[region] = e8_winners[0]

    # Final Four — TL vs BL, TR vs BR (quadrant_order = [TL, TR, BR, BL])
    qo = quadrant_order
    ff_pairs = [(qo[0], qo[3]), (qo[1], qo[2])]
    ff_winners = []
    for r_a, r_b in ff_pairs:
        a = region_winners.get(r_a)
        b = region_winners.get(r_b)
        if not a and not b:
            continue
        if not a or not b:
            ff_winners.append(a or b)
            continue
        game_num += 1
        result = predict_game(a, b, config=config)
        pick_a = _should_pick_upset(result["win_prob_a"], a["seed"], b["seed"], upset_aggression)
        pick_team = a if pick_a else b
        picks.append(_make_pick_dict(game_num, 4, "Final Four", None, a, b, result, pick_team))
        ff_winners.append(pick_team)

    # Championship
    champion = None
    if len(ff_winners) >= 2:
        a, b = ff_winners[0], ff_winners[1]
        game_num += 1
        result = predict_game(a, b, config=config)
        pick_a = _should_pick_upset(result["win_prob_a"], a["seed"], b["seed"], upset_aggression)
        pick_team = a if pick_a else b
        picks.append(_make_pick_dict(game_num, 2, "Championship", None, a, b, result, pick_team, **_h2h_kw))
        champion = pick_team["team"]

    final_four = [region_winners[r]["team"] for r in quadrant_order if r in region_winners]
    biggest_upsets = sorted([p for p in picks if p["upset_rating"] > 0.3], key=lambda x: -x["upset_rating"])[:8]
    most_uncertain = sorted([p for p in picks if p["confidence"] == "tossup"], key=lambda x: x["win_prob"])[:8]

    return {
        "picks": picks,
        "champion": champion,
        "final_four": final_four,
        "biggest_upsets": biggest_upsets,
        "most_uncertain_games": most_uncertain,
    }


def _infer_quadrant_order(bracket):
    """Rank regions by their #1 seed's barthag to determine overall seed order.
    Returns [TL, TR, BR, BL] = [#1 overall, #2, #3, #4]."""
    region_strength = []
    for rname, teams in bracket.items():
        one_seed = teams.get(1)
        strength = one_seed.get("barthag", 0) if one_seed else 0
        region_strength.append((rname, strength))
    region_strength.sort(key=lambda x: -x[1])
    return [r[0] for r in region_strength]


def load_bracket(filepath, data_dir=None, year=None):
    """Load bracket from JSON. If data_dir and year provided, enrich with teams_merged stats.
    Returns (bracket, ff_matchups, quadrant_order).
    quadrant_order is [TL, TR, BR, BL] region names."""
    with open(filepath) as f:
        data = json.load(f)
    bracket = {}
    for rname, teams in data.get("regions", {}).items():
        bracket[rname] = {t["seed"]: t for t in teams}
    if data_dir and year:
        merged = load_teams_merged(data_dir, year)
        if merged:
            n = enrich_bracket_with_teams(bracket, merged)
            if n > 0:
                print(f"  Enriched {n} teams from teams_merged_{year}.json")
    quadrant_order = data.get("quadrant_order")
    if not quadrant_order or len(quadrant_order) != 4:
        quadrant_order = _infer_quadrant_order(bracket)
    return bracket, data.get("final_four_matchups", [[0, 1], [2, 3]]), quadrant_order


# ===========================================================================
# MAIN — Testing
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BRACKET BRAIN v2 — Enhanced Prediction Engine")
    print("=" * 60)

    houston = {
        "team": "Houston", "seed": 1,
        "adj_o": 118.2, "adj_d": 85.8, "adj_tempo": 64.8,
        "three_rate": 0.32, "three_pct": 0.345,
        "two_pt_pct": 0.56, "block_rate": 12.5, "orb_rate": 35.0,
        "experience": 0.75, "coach": "Kelvin Sampson",
        "preseason_rank": 3, "momentum": 0.3,
        "star_score": 0.7, "luck": 0.02, "injury_level": 0,
    }
    mcneese = {
        "team": "McNeese", "seed": 12,
        "adj_o": 109.8, "adj_d": 97.2, "adj_tempo": 71.2,
        "three_rate": 0.42, "three_pct": 0.360,
        "two_pt_pct": 0.48, "block_rate": 6.0, "orb_rate": 26.0,
        "experience": 0.65, "coach": "",
        "preseason_rank": 0, "momentum": 0.5,
        "star_score": 0.5, "luck": 0.08, "injury_level": 0,
    }
    duke = {
        "team": "Duke", "seed": 2,
        "adj_o": 124.2, "adj_d": 92.5, "adj_tempo": 71.8,
        "three_rate": 0.36, "three_pct": 0.370,
        "two_pt_pct": 0.55, "block_rate": 10.0, "orb_rate": 31.0,
        "experience": 0.35, "coach": "Jon Scheyer",
        "preseason_rank": 7, "momentum": 0.1,
        "star_score": 0.85, "luck": -0.03, "injury_level": 0,
    }
    wisconsin = {
        "team": "Wisconsin", "seed": 3,
        "adj_o": 117.5, "adj_d": 91.2, "adj_tempo": 63.2,
        "three_rate": 0.30, "three_pct": 0.350,
        "two_pt_pct": 0.54, "block_rate": 8.5, "orb_rate": 29.0,
        "experience": 0.80, "coach": "Greg Gard",
        "preseason_rank": 15, "momentum": 0.4,
        "star_score": 0.60, "luck": 0.01, "injury_level": 0,
    }

    print("\n--- Houston (1) vs McNeese (12) ---")
    r = analyze_matchup(houston, mcneese)
    print(f"  Win prob: {r['win_prob_a']:.1%} - {r['win_prob_b']:.1%}")
    print(f"  Score: {r['predicted_score_a']:.0f} - {r['predicted_score_b']:.0f}")
    print(f"  Base margin: {r['base_margin']:.1f}, Factor adj: {r['factor_margin']:.1f}")
    print(f"  Volatility: {r['volatility']:.3f}")
    print(f"  Factors A: {r['factors_a']}")
    print(f"  Factors B: {r['factors_b']}")
    for n in r["narratives"]: print(f"  -> {n}")

    print("\n--- Duke (2) vs Wisconsin (3) ---")
    r = analyze_matchup(duke, wisconsin)
    print(f"  Win prob: {r['win_prob_a']:.1%} - {r['win_prob_b']:.1%}")
    print(f"  Score: {r['predicted_score_a']:.0f} - {r['predicted_score_b']:.0f}")
    print(f"  Base margin: {r['base_margin']:.1f}, Factor adj: {r['factor_margin']:.1f}")
    print(f"  Factors A: {r['factors_a']}")
    print(f"  Factors B: {r['factors_b']}")
    for n in r["narratives"]: print(f"  -> {n}")

    print("\n--- Duke (2) vs Wisconsin (3) — Duke star injured ---")
    duke_hurt = dict(duke); duke_hurt["injury_level"] = 2
    r = analyze_matchup(duke_hurt, wisconsin)
    print(f"  Win prob: {r['win_prob_a']:.1%} - {r['win_prob_b']:.1%} (was ~55% healthy)")
    print(f"  Factor adj: {r['factor_margin']:.1f} (injury impact visible)")
    for n in r["narratives"]: print(f"  -> {n}")
