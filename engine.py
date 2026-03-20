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
import logging
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass

_logger = logging.getLogger("bracketbrain")

# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class ModelConfig:
    num_sims: int = 10000
    base_scoring_stdev: float = 11.0
    national_avg_efficiency: float = 100.0
    national_avg_tempo: float = 67.5
    seed_weight: float = 0.0   # Removed from calibration: efficiency already captures seed info
    three_pt_volatility_factor: float = 0.15
    tempo_volatility_weight: float = 0.3
    experience_max_bonus: float = 2.5
    coach_tourney_max_bonus: float = 1.5
    pedigree_max_bonus: float = 1.0
    preseason_max_bonus: float = 1.5
    proximity_max_bonus: float = 2.0
    proximity_neutral_distance: float = 1200  # miles at which score=0 (home max, cross-country min)
    proximity_home_threshold_mi: float = 75   # extra "home court" bonus when team within this many miles
    proximity_home_bonus: float = 1.2        # additive pts when distance <= threshold (quasi-home games)
    momentum_max_bonus: float = 1.5
    star_player_max_bonus: float = 1.5
    size_max_bonus: float = 1.0
    injury_penalty_per_level: float = 3.0
    luck_regression_factor: float = 0.5
    win_pct_max_bonus: float = 1.0
    conf_rating_max_bonus: float = 0.3
    conf_tourney_max_bonus: float = 0.5  # conference tournament momentum (champion/finalist/etc)
    # New: possessions / FT / schedule strength
    possession_edge_max_bonus: float = 4.0
    ft_clutch_max_bonus: float = 3.0
    sos_max_bonus: float = 2.5
    # EvanMiya-sourced signals
    depth_max_bonus: float = 2.0          # em_depth_score: deep/balanced roster bonus
    em_opp_adjust_max_bonus: float = 2.0  # em_opponent_adjust: opponent quality adjustment
    em_adj_o_weight: float = 0.3          # blend weight for EvanMiya adj_o/adj_d in base score (0=Torvik only)
    em_runs_margin_max_bonus: float = 2.0  # scoring-burst margin bonus (ability to go on runs)
    big_bpr_max_bonus: float = 0.0         # frontcourt quality differential (em_big_bpr) — default 0, let calibrator find signal
    guard_bpr_max_bonus: float = 0.0       # backcourt quality differential (em_guard_bpr) — default 0, let calibrator find signal
    creator_count_max_bonus: float = 0.0   # penalty for too many ball-dominant creators — default 0, let calibrator find signal
    ft_foul_rate_max_bonus: float = 0.0   # margin bonus from differential FT rate — default 0, good config calibrated to 0.0
    score_scale: float = 0.942            # tournament scoring discount vs regular-season efficiency baseline
                                          # (calibrated: model over-predicts by 8.8 pts on 187 tournament games)
    score_scale_r64: float = 0.960
    score_scale_r32: float = 0.950
    score_scale_s16: float = 0.935
    score_scale_e8: float = 0.920
    score_scale_ff: float = 0.910
    factor_margin_cap: float = 6.0
    round_stdev_inflation_r64: float = 1.00
    round_stdev_inflation_r32: float = 1.03
    round_stdev_inflation_s16: float = 1.06
    round_stdev_inflation_e8: float = 1.10
    round_stdev_inflation_ff: float = 1.12
    late_round_dampening: float = 0.0  # Shift win-probs toward 0.5 in Sweet 16+ (0=none, 0.3=30% pull)
    # Close-game upset tolerance: shift margin toward underdog when game is close and underdog has favorable indicators
    upset_spread_threshold: float = 6.0   # games with |margin| < this get bonus consideration
    upset_tolerance_max_bonus: float = 3.0  # max pts to shift toward underdog
    upset_seed_gate: bool = False  # if True, only apply bonus when underdog is lower seed (true upset)
    upset_proximity_power: float = 0.5  # 0.5=sqrt, 1.0=linear, 0.25=more aggressive on very close games
    upset_spread_threshold_r64: float = 0.0  # 0=use main threshold; else round-specific for R64
    close_game_stdev_boost: float = 0.0  # inflate stdev when |margin|<2 (0=off, 0.2=20% boost)

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
    # Neutral-point curve: home=max, cross-country=min, linear to 0 at neutral_distance.
    # far = 2 * neutral_distance (fixed ratio, no extra param)
    nd = config.proximity_neutral_distance
    far = 2 * nd
    if distance <= nd:
        score = config.proximity_max_bonus * (1 - distance / nd)
    else:
        score = -config.proximity_max_bonus * (distance - nd) / (far - nd)
        score = max(-config.proximity_max_bonus, score)
    # Extra home bonus when team is very close (quasi-home game)
    home_thresh = getattr(config, "proximity_home_threshold_mi", 75)
    home_bonus = getattr(config, "proximity_home_bonus", 1.2)
    if distance <= home_thresh and home_bonus > 0:
        score += home_bonus
    return score

def calc_momentum_bonus(team, config=DEFAULT_CONFIG):
    momentum = team.get("momentum")
    if momentum is not None:
        return max(-1, min(1, momentum)) * config.momentum_max_bonus
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

def calc_runs_margin_bonus(team_a, team_b, config=DEFAULT_CONFIG):
    """Margin bonus from scoring-burst differential (em_runs_margin).

    EvanMiya's runs metric measures a team's ability to go on scoring runs
    minus their tendency to give up runs. Higher = more "clutch" / momentum-capable.
    Typical range: 0.0 to 1.0. Normalized diff capped at ±max_bonus.
    """
    rm_a = team_a.get("em_runs_margin")
    rm_b = team_b.get("em_runs_margin")
    if rm_a is None or rm_b is None:
        return 0.0
    # Diff typically in [-1.0, +1.0], scale so a 0.5 diff = max_bonus
    diff = rm_a - rm_b
    margin = diff * (config.em_runs_margin_max_bonus / 0.5)
    return max(-config.em_runs_margin_max_bonus, min(config.em_runs_margin_max_bonus, margin))


def calc_big_bpr_bonus(team_a, team_b, config=DEFAULT_CONFIG):
    """Margin bonus from frontcourt quality differential (em_big_bpr).

    Sum of BPR for top-4 players with position >= 3.0 (forwards/centers).
    Analysis of 941 tournament matchups shows +0.093 residual correlation,
    56.7% close-game win rate for the team with the frontcourt advantage.
    Typical range ~5-30; a 10-pt diff maps to ±max_bonus.
    """
    ia = team_a.get("em_big_bpr")
    ib = team_b.get("em_big_bpr")
    if ia is None or ib is None:
        return 0.0
    diff = ia - ib
    margin = diff * (config.big_bpr_max_bonus / 10.0)
    return max(-config.big_bpr_max_bonus, min(config.big_bpr_max_bonus, margin))


def calc_guard_bpr_bonus(team_a, team_b, config=DEFAULT_CONFIG):
    """Margin bonus from backcourt quality differential (em_guard_bpr).

    Sum of BPR for top-4 players with position < 3.0 (guards).
    Analysis shows +0.076 residual correlation, 54.5% close-game win rate.
    Guards drive tempo, handle pressure, and create in late-game situations.
    Typical range ~5-25; a 10-pt diff maps to ±max_bonus.
    """
    ga = team_a.get("em_guard_bpr")
    gb = team_b.get("em_guard_bpr")
    if ga is None or gb is None:
        return 0.0
    diff = ga - gb
    margin = diff * (config.guard_bpr_max_bonus / 10.0)
    return max(-config.guard_bpr_max_bonus, min(config.guard_bpr_max_bonus, margin))


def calc_creator_count_bonus(team_a, team_b, config=DEFAULT_CONFIG):
    """Margin bonus from creator/role balance differential (em_creator_count).

    Count of ball-dominant creators (role < 2.0) among top-8 by BPR.
    FEWER creators is better: teams with clear creator/finisher role definition
    outperform in tournament (55.4% vs 41.2% in close games, -0.081 residual).
    Role 1 = offensive creator, Role 5 = receiver/finisher.
    Typical range 1-6; a 2-player diff maps to ±max_bonus.
    """
    ca = team_a.get("em_creator_count")
    cb = team_b.get("em_creator_count")
    if ca is None or cb is None:
        return 0.0
    # Fewer creators is BETTER, so we negate: team with fewer creators gets positive bonus
    diff = cb - ca  # positive when A has fewer (better)
    margin = diff * (config.creator_count_max_bonus / 2.0)
    return max(-config.creator_count_max_bonus, min(config.creator_count_max_bonus, margin))


def calc_size_bonus(team, config=DEFAULT_CONFIG):
    if "size_score" in team:
        return team["size_score"] * config.size_max_bonus
    two_pt = max(0, min(1, (team.get("two_pt_pct", 0.50) - 0.45) / 0.12))
    block = max(0, min(1, (team.get("block_rate", 8.0) - 5) / 10))
    orb = max(0, min(1, (team.get("orb_rate", 28.0) - 22) / 16))
    return (two_pt * 0.4 + block * 0.3 + orb * 0.3) * config.size_max_bonus

# Status -> severity multiplier for injury penalty.
# College basketball has no mandatory injury reporting (unlike NBA), so coaches routinely
# list starters as "questionable" for gamesmanship.  Only "out" and "doubtful" reliably
# indicate a player will miss the game.  "Day-to-day" is treated separately from
# "questionable" since ESPN maps it that way and it's even softer.
_INJURY_SEVERITY = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.20,
    "day-to-day": 0.05,
    "probable": 0.0,
}
# Minimum BPR share to include a player in the injury penalty calculation.
_MIN_BPR_SHARE = 0.05

# Empirical dampening coefficient for college basketball injury impact.
# Absorbs: replacement quality, coaching adaptation, team depth effects, gameplan adjustments.
# NBA studies (FiveThirtyEight RAPTOR, ESPN BPI) use ~0.5; college basketball uses 0.6
# because teams have less depth and fewer tactical options to compensate.
# Result: star out ≈ 4-6 pts, rotation player ≈ 1-2 pts, bench ≈ negligible.
INJURY_DAMPENING = 0.6


_ROUND_INDEX = {
    "Round of 64": 0, "Round of 32": 1, "Sweet 16": 2,
    "Elite 8": 3, "Final Four": 4, "Championship": 5,
}


def calc_injury_penalty(team, config=DEFAULT_CONFIG, round_name=None):
    """Compute penalty (pts of margin) from injuries.

    Primary path — BPR × playing time × empirical dampening:
      penalty = sum(max(0, bpr_i) × poss_per_game_i / 100 × INJURY_DAMPENING × severity_i)
      where poss_per_game_i = player_poss × 5 × adj_tempo / total_team_player_poss

    If a player has return_round set (e.g. "Sweet 16"), their penalty is zeroed
    from that round onward (they're back and healthy).

    Result is capped at 10 pts.
    """
    if "injury_impact" in team and team["injury_impact"] is not None and "injuries" not in team:
        # Trust precomputed value only when no raw injuries list exists to recalculate from.
        # When injuries=[] (e.g. user set all to healthy via override), we must recalc to 0.
        return min(team["injury_impact"], 10.0)
    if "injury_level" in team and config.injury_penalty_per_level > 0:
        return team["injury_level"] * config.injury_penalty_per_level

    injuries = team.get("injuries", [])
    if not injuries:
        return 0.0

    adj_tempo = float(team.get("adj_tempo") or 70.0)
    roster = team.get("roster", [])
    total_team_poss = sum(float(p.get("poss", 0)) for p in roster) or 0.0

    # --- Per-player penalty: use best available formula for each player ---
    # Primary formula (preferred): BPR × playing-time-fraction × dampening.
    #   Requires roster poss context AND per-player poss.
    # Fallback formula: BPR × (tempo/100) × dampening.
    #   Used when roster poss is unavailable OR player has no poss data.
    # Legacy formula: bpr_share × injury_penalty_per_level (old data format).
    #
    # IMPORTANT: we evaluate per-player, not per-list, to avoid a path-switch
    # bug where removing one player with poss > 0 causes all remaining
    # poss=0 players to jump from "skipped in primary" to "counted in fallback".
    current_round_idx = _ROUND_INDEX.get(round_name, -1) if round_name else -1

    total = 0.0
    for i in injuries:
        # Skip player if they're expected back by this round
        ret = i.get("return_round")
        if ret and current_round_idx >= 0:
            ret_idx = _ROUND_INDEX.get(ret, 99)
            if current_round_idx >= ret_idx:
                continue  # Player is back
        severity = _INJURY_SEVERITY.get(str(i.get("status", "out")).lower(), 1.0)
        if severity <= 0:
            continue
        bpr_val = float(i.get("bpr") or 0.0)
        poss = float(i.get("poss") or 0.0)

        if bpr_val > 0 and poss > 0 and total_team_poss > 0:
            # Primary: BPR × playing time fraction × dampening
            poss_per_game = poss * 5.0 * adj_tempo / total_team_poss
            total += severity * bpr_val * poss_per_game / 100.0 * INJURY_DAMPENING
        elif bpr_val > 0:
            # Fallback: absolute BPR without poss weighting
            total += severity * bpr_val * (adj_tempo / 100.0) * INJURY_DAMPENING
        else:
            # Legacy: bpr_share × injury_penalty_per_level
            if config.injury_penalty_per_level == 0:
                continue
            bpr_share = float(i.get("bpr_share") or i.get("importance") or 0.0)
            if bpr_share < _MIN_BPR_SHARE:
                continue
            total += severity * bpr_share * config.injury_penalty_per_level
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
    """Teams from stronger conferences get small bonus.

    Prefers conf_strength_score (0-1 continuous from Torvik conf_adj) when present.
    Otherwise uses conf_rating (1=top, 2=mid, 3=lower tier from EvanMiya).
    """
    conf_strength = team.get("conf_strength_score")
    if conf_strength is not None and 0 <= conf_strength <= 1:
        return conf_strength * config.conf_rating_max_bonus
    conf_rating = team.get("conf_rating")
    if conf_rating is None:
        return 0.0
    # Lower conf_rating = stronger conference (1 = top, 2 = mid, 3 = lower)
    if conf_rating <= 1.0:
        return config.conf_rating_max_bonus
    if conf_rating <= 2.0:
        return config.conf_rating_max_bonus * 0.5
    return 0.0


def calc_conf_tourney_bonus(team, config=DEFAULT_CONFIG):
    """Bonus for conference tournament momentum (won conf tourney, made final, etc)."""
    m = team.get("conf_tourney_momentum")
    if m is None:
        return 0.0
    return m * config.conf_tourney_max_bonus


# ===========================================================================
# CORE PREDICTION
# ===========================================================================

def _get_round_score_scale(round_name, config):
    """Map round name to per-round score_scale config field."""
    mapping = {
        "Round of 64": config.score_scale_r64,
        "Round of 32": config.score_scale_r32,
        "Sweet 16": config.score_scale_s16,
        "Elite 8": config.score_scale_e8,
        "Final Four": config.score_scale_ff,
        "Championship": config.score_scale_ff,
    }
    return mapping.get(round_name, config.score_scale)


def _get_round_stdev_inflation(round_name, config):
    """Map round name to per-round stdev inflation factor."""
    mapping = {
        "Round of 64": config.round_stdev_inflation_r64,
        "Round of 32": config.round_stdev_inflation_r32,
        "Sweet 16": config.round_stdev_inflation_s16,
        "Elite 8": config.round_stdev_inflation_e8,
        "Final Four": config.round_stdev_inflation_ff,
        "Championship": config.round_stdev_inflation_ff,
    }
    return mapping.get(round_name, 1.0)


def predict_game(team_a, team_b, game_site=None, config=DEFAULT_CONFIG, round_name=None):
    score_a, score_b, poss = _predict_base_score(team_a, team_b, config)
    base_margin = score_a - score_b

    # Possession, FT, and schedule-strength driven margin tweaks
    possession_margin = _calc_possession_edge_margin(team_a, team_b, poss, config)
    ft_margin = _calc_ft_edge_margin(team_a, team_b, base_margin, config)
    foul_rate_margin = _calc_foul_rate_margin(team_a, team_b, poss, config)
    sos_margin = _calc_sos_edge_margin(team_a, team_b, config)
    runs_margin = calc_runs_margin_bonus(team_a, team_b, config)
    big_bpr_margin = calc_big_bpr_bonus(team_a, team_b, config)
    guard_bpr_margin = calc_guard_bpr_bonus(team_a, team_b, config)
    creator_margin = calc_creator_count_bonus(team_a, team_b, config)

    # All factor adjustments
    # Injury decay: if players have return_round set, calc_injury_penalty skips them.
    # For players without return_round, apply a generic decay for later rounds
    # (most "out" players return within 2-4 weeks).
    _INJ_ROUND_DECAY = {
        "Round of 64": 1.0,
        "Round of 32": 0.85,
        "Sweet 16": 0.6,
        "Elite 8": 0.4,
        "Final Four": 0.25,
        "Championship": 0.2,
    }
    inj_decay = _INJ_ROUND_DECAY.get(round_name, 1.0)
    _rn = round_name  # capture for lambda

    factor_names = ["experience", "coach", "pedigree", "preseason", "proximity",
                    "momentum", "star_player", "depth", "size", "injuries",
                    "luck_regression", "win_pct", "conf", "conf_tourney"]
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
        lambda t: -calc_injury_penalty(t, config, round_name=_rn) * inj_decay,
        lambda t: calc_luck_regression(t, config),
        lambda t: calc_win_pct_bonus(t, config),
        lambda t: calc_conf_bonus(t, config),
        lambda t: calc_conf_tourney_bonus(t, config),
    ]

    factors_a, factors_b = {}, {}
    for name, fn in zip(factor_names, calc_fns):
        factors_a[name] = fn(team_a)
        factors_b[name] = fn(team_b)

    factor_margin = sum(factors_a.values()) - sum(factors_b.values())

    # Soft-cap factor margin to prevent intangible stacking
    cap = config.factor_margin_cap
    factor_margin = math.tanh(factor_margin / cap) * cap

    extra_margin = possession_margin + ft_margin + foul_rate_margin + sos_margin + runs_margin + big_bpr_margin + guard_bpr_margin + creator_margin
    adjusted_margin = base_margin + factor_margin + extra_margin

    # Close-game upset tolerance: shift toward underdog when game is close and underdog has favorable indicators
    upset_bonus = _calc_upset_tolerance_bonus(team_a, team_b, adjusted_margin, config, round_name)
    adjusted_margin += upset_bonus

    # Variance (3PT volatility)
    vol = (calc_game_volatility(team_a, config) + calc_game_volatility(team_b, config)) / 2
    round_inflation = _get_round_stdev_inflation(round_name, config)
    game_stdev = config.base_scoring_stdev * math.sqrt(poss / config.national_avg_tempo) * vol * round_inflation
    close_boost = getattr(config, "close_game_stdev_boost", 0.0)
    if close_boost > 0 and abs(adjusted_margin) < 2:
        game_stdev *= (1.0 + close_boost)

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

    # Late-round dampening: shift toward 0.5 (models overconfident in Sweet 16+)
    late_rounds = ("Sweet 16", "Elite 8", "Final Four", "Championship")
    pick_dampened = False
    if round_name in late_rounds:
        damp = getattr(config, "late_round_dampening", 0.0)
        if damp > 0:
            final_prob = 0.5 + (final_prob - 0.5) * (1.0 - damp)
            final_prob = max(0.001, min(0.999, final_prob))
            # Pick can differ from margin direction when dampening flips the favorite
            pick_dampened = (eff_prob >= 0.5) != (final_prob >= 0.5)

    # Detect if both teams are using default/fallback stats
    a_default = (team_a.get("adj_o") == 85 and team_a.get("adj_d") == 112)
    b_default = (team_b.get("adj_o") == 85 and team_b.get("adj_d") == 112)
    warning = None
    if a_default and b_default:
        _logger.warning(
            f"Both teams ({team_a.get('team')} and {team_b.get('team')}) have default stats — "
            "returning 50/50"
        )
        final_prob = 0.5
        warning = "both teams have default stats"

    # Per-round score scale with tempo adjustment
    round_scale = _get_round_score_scale(round_name, config)
    avg_tempo = (team_a.get("adj_tempo", config.national_avg_tempo) +
                 team_b.get("adj_tempo", config.national_avg_tempo)) / 2
    tempo_ratio = avg_tempo / config.national_avg_tempo
    tempo_adjust = 0.5 * (tempo_ratio - 1.0)
    effective_scale = round_scale * (1.0 + tempo_adjust)
    effective_scale = max(0.88, min(1.0, effective_scale))

    # Derive scores so predicted_score_a - predicted_score_b = predicted_margin
    scaled_total = (score_a + score_b) * effective_scale
    predicted_score_a = round((scaled_total + adjusted_margin) / 2, 1)
    predicted_score_b = round((scaled_total - adjusted_margin) / 2, 1)

    result = {
        "team_a": team_a["team"], "team_b": team_b["team"],
        "seed_a": seed_a, "seed_b": seed_b,
        "win_prob_a": round(final_prob, 4),
        "win_prob_b": round(1 - final_prob, 4),
        "predicted_score_a": predicted_score_a,
        "predicted_score_b": predicted_score_b,
        "predicted_margin": round(adjusted_margin, 1),
        "base_margin": round(base_margin, 1),
        "factor_margin": round(factor_margin, 1),
        "possession_margin": round(possession_margin, 2),
        "ft_margin": round(ft_margin, 2),
        "sos_margin": round(sos_margin, 2),
        "runs_margin": round(runs_margin, 2),
        "big_bpr_margin": round(big_bpr_margin, 2),
        "guard_bpr_margin": round(guard_bpr_margin, 2),
        "creator_margin": round(creator_margin, 2),
        "foul_rate_margin": round(foul_rate_margin, 2),
        "upset_tolerance_bonus": round(upset_bonus, 2),
        "efficiency_prob": round(eff_prob, 4),
        "seed_prob": round(seed_prob, 4),
        "game_stdev": round(game_stdev, 1),
        "volatility": round(vol, 3),
        "factors_a": {k: round(v, 2) for k, v in factors_a.items()},
        "factors_b": {k: round(v, 2) for k, v in factors_b.items()},
        "pick_dampened": pick_dampened,
    }
    if warning:
        result["warning"] = warning
    return result

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


def _calc_upset_tolerance_bonus(team_a, team_b, margin, config, round_name=None):
    """Shift margin toward underdog when game is close and underdog has favorable indicators.

    Applies only when 0 < |margin| < threshold. Uses 10 binary indicators (dog vs fav)
    plus configurable proximity curve. Optional seed gate: only apply when underdog is lower seed.
    """
    threshold = getattr(config, "upset_spread_threshold", 6.0)
    threshold_r64 = getattr(config, "upset_spread_threshold_r64", 0.0)
    if round_name == "Round of 64" and threshold_r64 > 0:
        threshold = threshold_r64
    max_bonus = getattr(config, "upset_tolerance_max_bonus", 3.0)
    if max_bonus <= 0 or threshold <= 0:
        return 0.0
    dog_margin = abs(margin)
    if dog_margin <= 0 or dog_margin >= threshold:
        return 0.0

    seed_gate = getattr(config, "upset_seed_gate", False)
    if seed_gate:
        seed_a, seed_b = team_a.get("seed", 8), team_b.get("seed", 8)
        # Only apply when model favorite is higher seed (lower number) = true upset scenario
        model_fav_higher_seed = (margin > 0 and seed_a < seed_b) or (margin < 0 and seed_b < seed_a)
        if not model_fav_higher_seed:
            return 0.0

    fav = team_a if margin > 0 else team_b
    dog = team_b if margin > 0 else team_a

    indicators = 0
    total = 0

    # 1. barthag: dog > fav
    if dog.get("barthag") is not None and fav.get("barthag") is not None:
        total += 1
        if dog["barthag"] > fav["barthag"]:
            indicators += 1

    # 2. sos: dog > fav (tougher schedule)
    if dog.get("sos") is not None and fav.get("sos") is not None:
        total += 1
        if dog["sos"] > fav["sos"]:
            indicators += 1

    # 3. wab: dog > fav
    if dog.get("wab") is not None and fav.get("wab") is not None:
        total += 1
        if dog["wab"] > fav["wab"]:
            indicators += 1

    # 4. adj_o: dog > fav
    if dog.get("adj_o") is not None and fav.get("adj_o") is not None:
        total += 1
        if dog["adj_o"] > fav["adj_o"]:
            indicators += 1

    # 5. efg_pct: dog > fav
    if dog.get("efg_pct") is not None and fav.get("efg_pct") is not None:
        total += 1
        if dog["efg_pct"] > fav["efg_pct"]:
            indicators += 1

    # 6. three_rate / three_pt_rate: dog > fav
    tr_d = dog.get("three_rate") or dog.get("three_pt_rate")
    tr_f = fav.get("three_rate") or fav.get("three_pt_rate")
    if tr_d is not None and tr_f is not None:
        total += 1
        if tr_d > tr_f:
            indicators += 1

    # 7. ft_rate or ft_pct: dog > fav
    ft_d = dog.get("ft_rate") or dog.get("ft_pct")
    ft_f = fav.get("ft_rate") or fav.get("ft_pct")
    if ft_d is not None and ft_f is not None:
        total += 1
        if ft_d > ft_f:
            indicators += 1

    # 8. adj_d: dog < fav (lower = better defense)
    if dog.get("adj_d") is not None and fav.get("adj_d") is not None:
        total += 1
        if dog["adj_d"] < fav["adj_d"]:
            indicators += 1

    # 9. to_rate: dog < fav (lower = fewer TOs, protects ball better)
    if dog.get("to_rate") is not None and fav.get("to_rate") is not None:
        total += 1
        if dog["to_rate"] < fav["to_rate"]:
            indicators += 1

    # 10. orb_rate: dog > fav (second-chance points create upset variance)
    if dog.get("orb_rate") is not None and fav.get("orb_rate") is not None:
        total += 1
        if dog["orb_rate"] > fav["orb_rate"]:
            indicators += 1

    # 11. coach_tourney_score: dog > fav (March-proven coach)
    if getattr(config, "coach_tourney_max_bonus", 0.0) > 0:
        cd = dog.get("coach_tourney_score")
        cf = fav.get("coach_tourney_score")
        if cd is not None and cf is not None:
            total += 1
            if cd > cf:
                indicators += 1

    # 12. experience: dog > fav (tournament-tested roster)
    ed = dog.get("experience")
    ef = fav.get("experience")
    if ed is not None and ef is not None:
        total += 1
        if ed > ef:
            indicators += 1

    # 13. pedigree_score: dog > fav (program history)
    if getattr(config, "pedigree_max_bonus", 0.0) > 0:
        pd = dog.get("pedigree_score")
        pf = fav.get("pedigree_score")
        if pd is not None and pf is not None:
            total += 1
            if pd > pf:
                indicators += 1

    # 14. three_pct: dog > fav (better 3PT shooting)
    tp_d = dog.get("three_pct") or dog.get("three_pt_pct")
    tp_f = fav.get("three_pct") or fav.get("three_pt_pct")
    if tp_d is not None and tp_f is not None:
        _d = tp_d / 100.0 if tp_d > 1.5 else tp_d
        _f = tp_f / 100.0 if tp_f > 1.5 else tp_f
        total += 1
        if _d > _f:
            indicators += 1

    # 15. three_pct_d: dog < fav (better 3PT defense, lower opp 3PT%)
    tpd_d = dog.get("three_pct_d") or dog.get("three_pt_pct_d")
    tpd_f = fav.get("three_pct_d") or fav.get("three_pt_pct_d")
    if tpd_d is not None and tpd_f is not None:
        _d = tpd_d / 100.0 if tpd_d > 1.5 else tpd_d
        _f = tpd_f / 100.0 if tpd_f > 1.5 else tpd_f
        total += 1
        if _d < _f:
            indicators += 1

    # 16. underseeded: dog eff_rank < dog seed (efficiency says dog is better than seed suggests)
    ed = dog.get("eff_rank")
    sd = dog.get("seed")
    if ed is not None and sd is not None:
        total += 1
        if ed < sd:
            indicators += 1

    if total == 0:
        return 0.0

    indicator_score = indicators / total
    power = getattr(config, "upset_proximity_power", 0.5)
    proximity = max(0.0, 1.0 - dog_margin / threshold) ** power
    bonus = max_bonus * indicator_score * proximity
    return -bonus if margin > 0 else bonus


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


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_school_locations: dict = {}
_venues_cache: dict = {}


def _load_school_locations() -> dict:
    global _school_locations
    if _school_locations:
        return _school_locations
    path = os.path.join(_DATA_DIR, "school_locations.json")
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        _school_locations = {k: v for k, v in data.items() if not k.startswith("_")}
    return _school_locations


_historical_venues: dict = {}

def _load_venues(year: int) -> dict:
    global _venues_cache, _historical_venues
    if year in _venues_cache:
        return _venues_cache[year]
    # Try year-specific file first (e.g. venues_2026.json)
    path = os.path.join(_DATA_DIR, f"venues_{year}.json")
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        _venues_cache[year] = data
        return data
    # Fall back to venues_historical.json which covers 2008-2025
    if not _historical_venues:
        hist_path = os.path.join(_DATA_DIR, "venues_historical.json")
        if os.path.isfile(hist_path):
            with open(hist_path) as f:
                raw = json.load(f)
            for k, v in raw.items():
                if not k.startswith("_"):
                    try:
                        _historical_venues[int(k)] = v
                    except ValueError:
                        pass
    result = _historical_venues.get(year, {})
    _venues_cache[year] = result
    return result


def _get_game_site(venues: dict, region: str, round_name: str,
                    seed_a: int = None, seed_b: int = None) -> list | None:
    """Return [lat, lon] for the venue where this round/region game is played.
    For R64/R32, pass seed_a/seed_b to resolve pod-level venues."""
    if not venues:
        return None
    round_map = {"Round of 64": "R64", "Round of 32": "R32",
                 "Sweet 16": "S16", "Elite 8": "E8",
                 "Final Four": "F4", "Championship": "Championship"}
    rkey = round_map.get(round_name, round_name)
    if rkey in ("F4", "Championship"):
        return venues.get(rkey) or venues.get("F4")
    region_venues = venues.get("regions", {}).get(region, {})
    if rkey in ("R64", "R32") and seed_a is not None:
        pod = SEED_TO_POD.get(seed_a)
        if pod:
            pod_key = f"pod_{pod}"
            if pod_key in region_venues:
                return region_venues[pod_key]
    return region_venues.get(rkey)


def _get_venue_city(venues: dict, region: str, round_name: str,
                    seed_a: int = None, seed_b: int = None) -> str | None:
    """Return human-readable city label for the game venue.
    For R64/R32, pass seed_a/seed_b to resolve pod-level city labels."""
    if not venues:
        return None
    round_map = {"Round of 64": "R64", "Round of 32": "R32",
                 "Sweet 16": "S16", "Elite 8": "E8",
                 "Final Four": "F4", "Championship": "Championship"}
    rkey = round_map.get(round_name, round_name)
    cities = venues.get("city_labels", {})
    if rkey in ("F4", "Championship"):
        return cities.get(rkey) or cities.get("F4")
    region_cities = cities.get(region, {})
    if rkey in ("R64", "R32") and seed_a is not None:
        pod = SEED_TO_POD.get(seed_a)
        if pod:
            pod_key = f"pod_{pod}"
            if pod_key in region_cities:
                return region_cities[pod_key]
    return region_cities.get(rkey)

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
    # Tier 1: All-time blue bloods (multiple titles, perennial contenders)
    "Kansas": 0.95, "Duke": 0.95, "North Carolina": 0.95, "Kentucky": 0.95,
    # Tier 2: Historic powerhouses
    "UCLA": 0.90, "UConn": 0.90, "Connecticut": 0.90,
    "Villanova": 0.85, "Indiana": 0.80, "Michigan St": 0.85, "Michigan State": 0.85,
    "Louisville": 0.80, "Syracuse": 0.75,
    # Tier 3: Consistent tournament programs
    "Ohio St": 0.70, "Ohio State": 0.70, "Michigan": 0.70, "Florida": 0.75,
    "Arizona": 0.70, "Georgetown": 0.65, "Wisconsin": 0.65, "Gonzaga": 0.65,
    "Purdue": 0.65, "Virginia": 0.65,
    "NC State": 0.60, "North Carolina St": 0.60,  # 2 titles, 2024 F4
    "Texas": 0.55, "Illinois": 0.55, "Oklahoma": 0.50, "LSU": 0.45,
    # Tier 4: Strong recent programs
    "Baylor": 0.60, "Houston": 0.60, "Alabama": 0.55, "Tennessee": 0.55,
    "Arkansas": 0.60, "Iowa St": 0.50, "Iowa State": 0.50,
    "Marquette": 0.55, "Xavier": 0.50, "Creighton": 0.45,
    "Memphis": 0.55, "St. John's": 0.55, "Cincinnati": 0.55,
    "Maryland": 0.55, "Oregon": 0.50, "Texas Tech": 0.45,
    "Auburn": 0.50, "San Diego St": 0.40, "San Diego State": 0.40,
    # Tier 5: Moderate tournament history
    "Texas A&M": 0.40, "SMU": 0.35, "Clemson": 0.35, "BYU": 0.35,
    "Iowa": 0.40, "Missouri": 0.40, "Pittsburgh": 0.45, "Wake Forest": 0.35,
    "Minnesota": 0.30, "Notre Dame": 0.40, "West Virginia": 0.40,
    "Utah": 0.35, "Colorado": 0.30, "Washington": 0.30,
    "Saint Mary's": 0.30, "VCU": 0.30, "Butler": 0.40,
}

_NORMALIZED_PEDIGREE = None


def _get_pedigree_score(team_name):
    """Return pedigree score using both exact and normalized team-name lookup."""
    global _NORMALIZED_PEDIGREE
    if team_name in PEDIGREE:
        return PEDIGREE[team_name]
    if _NORMALIZED_PEDIGREE is None:
        _NORMALIZED_PEDIGREE = {
            _normalize_team_for_match(name): score
            for name, score in PEDIGREE.items()
        }
    return _NORMALIZED_PEDIGREE.get(_normalize_team_for_match(team_name), 0.15)

def enrich_team(team):
    t = dict(team)
    coach_name = t.get("coach", "")
    static_coach_score = COACH_SCORES.get(coach_name)
    if static_coach_score is not None:
        # User-authored coach constants are the primary signal when available.
        t["coach_tourney_score"] = static_coach_score
    else:
        t["coach_tourney_score"] = 0.3
    if "pedigree_score" not in t:
        t["pedigree_score"] = _get_pedigree_score(t["team"])
    # Attach school lat/lon for proximity calculations (needed by /analyze endpoint)
    if "location" not in t:
        locs = _load_school_locations()
        tname = t.get("team", "")
        if tname in locs:
            t["location"] = locs[tname]
        else:
            # Fallback: try normalized or case-insensitive match
            tname_lower = tname.lower()
            for k, v in locs.items():
                if k.lower() == tname_lower or _normalize_team_for_match(k) == _normalize_team_for_match(tname):
                    t["location"] = v
                    break
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
    # Seed-based efficiency fallback: use historical per-seed averages when
    # Torvik adj_o/adj_d are missing (common for 12-16 seeds from small programs).
    # Averages derived from teams_merged data across 2010-2025 tournaments.
    _SEED_EFF_DEFAULTS = {
        1:  (120.9, 91.5),
        2:  (118.1, 93.8),
        3:  (115.6, 95.7),
        4:  (113.8, 96.9),
        5:  (112.6, 97.5),
        6:  (112.0, 98.2),
        7:  (111.4, 98.7),
        8:  (111.0, 99.3),
        9:  (110.8, 99.6),
        10: (111.2, 99.0),
        11: (110.9, 99.1),
        12: (110.5, 97.9),
        13: (109.0, 100.2),
        14: (107.2, 101.4),
        15: (105.3, 102.9),
        16: (102.2, 104.9),
    }
    adj_o = t.get("adj_o")
    adj_d = t.get("adj_d")
    if adj_o is None or adj_d is None or (adj_o == 85 and adj_d == 112):
        seed = t.get("seed")
        defaults = _SEED_EFF_DEFAULTS.get(int(seed)) if seed is not None else None
        if defaults:
            if adj_o is None or adj_o == 85:
                t["adj_o"] = defaults[0]
            if adj_d is None or adj_d == 112:
                t["adj_d"] = defaults[1]
            _logger.debug(
                f"Team '{t.get('team', 'UNKNOWN')}' (seed {seed}): using seed-based "
                f"efficiency fallback adj_o={t['adj_o']}, adj_d={t['adj_d']}"
            )
        else:
            _logger.warning(
                f"Team '{t.get('team', 'UNKNOWN')}' has no real efficiency data and "
                "no seed for fallback — prediction will be unreliable."
            )
    return t

# ===========================================================================
# TOURNAMENT SIMULATION
# ===========================================================================

FIRST_ROUND_MATCHUPS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
REGIONS = ["South", "East", "Midwest", "West"]
SEED_TO_POD = {1:'A',16:'A',8:'A',9:'A', 5:'B',12:'B',4:'B',13:'B',
               6:'C',11:'C',3:'C',14:'C', 7:'D',10:'D',2:'D',15:'D'}

def simulate_region(teams_by_seed, config=DEFAULT_CONFIG, region_name=None, venues=None):
    enriched = {s: enrich_team(t) for s, t in teams_by_seed.items()}
    matchups = [(enriched[a], enriched[b]) for a, b in FIRST_ROUND_MATCHUPS if a in enriched and b in enriched]
    results = {"Round of 64": [], "Round of 32": [], "Sweet 16": [], "Elite 8": []}

    def play_round(teams, round_name):
        winners = []
        for i in range(0, len(teams), 2):
            sa, sb = teams[i].get("seed"), teams[i+1].get("seed")
            site = _get_game_site(venues, region_name, round_name, seed_a=sa, seed_b=sb) if venues else None
            w = simulate_game(teams[i], teams[i+1], game_site=site, config=config)
            results[round_name].append({
                "team_a": teams[i]["team"], "seed_a": teams[i]["seed"],
                "team_b": teams[i+1]["team"], "seed_b": teams[i+1]["seed"],
                "winner": w["team"]
            })
            winners.append(w)
        return winners

    r64_winners = []
    for a, b in matchups:
        r64_site = _get_game_site(venues, region_name, "Round of 64", seed_a=a.get("seed"), seed_b=b.get("seed")) if venues else None
        w = simulate_game(a, b, game_site=r64_site, config=config)
        results["Round of 64"].append({"team_a": a["team"], "seed_a": a["seed"],
            "team_b": b["team"], "seed_b": b["seed"], "winner": w["team"]})
        r64_winners.append(w)

    r32_winners = play_round(r64_winners, "Round of 32")
    s16_winners = play_round(r32_winners, "Sweet 16")
    e8_winners = play_round(s16_winners, "Elite 8")
    return e8_winners[0], results

def simulate_tournament(bracket, config=DEFAULT_CONFIG, venues=None):
    results = {}
    final_four = []
    for region in REGIONS:
        if region not in bracket: continue
        winner, rr = simulate_region(bracket[region], config, region_name=region, venues=venues)
        results[region] = rr
        final_four.append((region, winner))

    results["Final Four"] = []
    champ_teams = []
    f4_site = _get_game_site(venues, None, "Final Four") if venues else None
    for i, j in [(0,1),(2,3)]:
        if i < len(final_four) and j < len(final_four):
            _, a = final_four[i]; _, b = final_four[j]
            w = simulate_game(a, b, game_site=f4_site, config=config)
            results["Final Four"].append({"team_a": a["team"], "seed_a": a["seed"],
                "team_b": b["team"], "seed_b": b["seed"], "winner": w["team"]})
            champ_teams.append(w)

    results["Championship"] = []
    champ_site = _get_game_site(venues, None, "Championship") if venues else None
    if len(champ_teams) == 2:
        w = simulate_game(champ_teams[0], champ_teams[1], game_site=champ_site, config=config)
        results["Championship"].append({"team_a": champ_teams[0]["team"], "seed_a": champ_teams[0]["seed"],
            "team_b": champ_teams[1]["team"], "seed_b": champ_teams[1]["seed"], "winner": w["team"]})
        return w, results
    return None, results

def run_monte_carlo(bracket, config=DEFAULT_CONFIG, year=None):
    counts = {k: defaultdict(int) for k in ["champ","ff","e8","s16","r32"]}
    game_results = defaultdict(lambda: defaultdict(int))

    # Load venue data for proximity bonus (uses year from bracket if not provided)
    if year is None:
        year = next((v.get("year") for v in bracket.values()
                     if isinstance(v, dict) and "year" in v), None)
    venues = _load_venues(year) if year else {}

    print(f"Running {config.num_sims:,} simulations...")
    print(f"  Factors: efficiency, seed, experience, coach, pedigree, preseason,")
    print(f"           momentum, star player, size, injuries, luck, 3PT volatility, proximity")

    for sim in range(config.num_sims):
        if (sim+1) % 2000 == 0:
            print(f"  {sim+1:,}/{config.num_sims:,}")
        champ, results = simulate_tournament(bracket, config, venues=venues)
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
    "mcneese": "mcneese st",
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
    # Historical naming variants (important for 2008/2009 joins)
    "wku": "western kentucky",
    "texas arlington": "ut arlington",
    "texasarlington": "ut arlington",
    "cal state fullerton": "cal st fullerton",
    "cal state northridge": "cal st northridge",
    "miss valley st": "mississippi valley st",
    # Sports Reference / conf tourney variants
    "ualr": "arkansas little rock",
    "college of charleston": "charleston",
    "pennsylvania": "penn",
    "pennsylvania quakers": "penn",
    "mass lowell": "massachusetts lowell",
    "masslowell": "massachusetts lowell",
    "s dakota st": "south dakota st",
    "uta": "ut arlington",
    # Odds API / ESPN full-name variants
    "queens university": "queens",
}

# Mascots used by Odds API / ESPN that should be stripped for matching.
_MASCOT_SUFFIXES = {
    "blue devils", "tar heels", "bulldogs", "wildcats", "hoosiers", "boilermakers",
    "jayhawks", "bears", "cougars", "longhorns", "tigers", "aggies", "seminoles",
    "cavaliers", "huskies", "volunteers", "razorbacks", "fighting irish",
    "golden eagles", "bruins", "trojans", "beavers", "ducks", "cardinals",
    "red raiders", "cyclones", "sooners", "mountaineers", "cowboys", "panthers",
    "demon deacons", "wolfpack", "yellow jackets", "orange", "hokies",
    "commodores", "crimson tide", "gators", "rebels", "gamecocks", "hurricanes",
    "owls", "spartans", "wolverines", "hawkeyes", "badgers", "cornhuskers",
    "golden gophers", "nittany lions", "fighting illini", "terrapins", "scarlet knights",
    "buckeyes", "redbirds", "golden flashes", "braves", "flyers", "mustangs",
    "redhawks", "billikens", "saints", "sharks", "royals", "bison", "peacocks",
    "gaels", "friars", "musketeers", "blue jays", "pirates", "explorers",
    "eagles", "raiders", "broncos", "rams", "bobcats", "catamounts", "phoenix",
    "anteaters", "gauchos", "matadors", "tritons", "highlanders", "titans",
    "lumberjacks", "flames", "paladins", "thundering herd", "red storm",
    "racers", "colonels", "penguins", "bearcats", "shockers", "salukis",
    "leathernecks", "jackrabbits", "coyotes", "norse", "chippewas",
    "rockets", "falcons", "zips", "hilltoppers", "jaguars",
    "knights", "49ers", "monarchs", "dukes", "keydets", "retrievers",
    "terriers", "greyhounds", "spiders", "pilots", "toreros", "dons",
    "waves", "lions", "leopards", "big green", "crimson", "quakers",
    "engineers", "bantams", "mean green", "roadrunners",
    "chanticleers", "thunderbirds", "aces", "sycamores",
    "mountain hawks", "pride",
}


def _strip_mascot(name):
    """Strip trailing mascot from Odds API / ESPN team names.

    'Duke Blue Devils' -> 'Duke', 'Michigan St Spartans' -> 'Michigan St'
    """
    if not name:
        return name
    low = name.lower().strip()
    for mascot in sorted(_MASCOT_SUFFIXES, key=len, reverse=True):
        if low.endswith(" " + mascot):
            return name[: len(name) - len(mascot) - 1].strip()
    return name


def is_ncaa_tournament_game(home_team, away_team, year=2026):
    """Check if a game is an NCAA tournament matchup by matching both teams against the bracket.

    Includes First Four play-in games — checks bracket.first_four[] entries in addition
    to the 64-team bracket.  Uses mascot stripping + normalization to match Odds API
    names ('Duke Blue Devils') against bracket short names ('Duke').
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    bracket_path = os.path.join(data_dir, f"bracket_{year}.json")
    if not os.path.isfile(bracket_path):
        return False
    try:
        with open(bracket_path) as f:
            bracket = json.load(f)
    except Exception:
        return False

    bracket_teams = set()
    for region_entries in bracket.get("regions", {}).values():
        if isinstance(region_entries, list):
            for entry in region_entries:
                t = entry.get("team", "") if isinstance(entry, dict) else ""
                if t:
                    bracket_teams.add(_normalize_team_for_match(t))
        elif isinstance(region_entries, dict):
            for entry in region_entries.values():
                t = entry.get("team", "") if isinstance(entry, dict) else ""
                if t:
                    bracket_teams.add(_normalize_team_for_match(t))

    # Include First Four play-in teams (both sides) so play-in games are recognized
    for ff in bracket.get("first_four", []):
        for key in ("team_a", "team_b"):
            t = ff.get(key, "")
            if t:
                bracket_teams.add(_normalize_team_for_match(t))

    h_norm = _normalize_team_for_match(_strip_mascot(home_team))
    a_norm = _normalize_team_for_match(_strip_mascot(away_team))
    return h_norm in bracket_teams and a_norm in bracket_teams


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
            canonical = {"team": tname, "seed": team_obj.get("seed")}
            for k, v in merged.items():
                if k in ("team", "seed"):
                    continue
                canonical[k] = v
            team_obj.clear()
            team_obj.update(canonical)
            adj_o = team_obj.get("adj_o")
            adj_d = team_obj.get("adj_d")
            team_obj["adj_o"] = adj_o if adj_o is not None else _BAD_DEFAULTS["adj_o"]
            team_obj["adj_d"] = adj_d if adj_d is not None else _BAD_DEFAULTS["adj_d"]
            team_obj["adj_tempo"] = team_obj.get("adj_tempo") or _BAD_DEFAULTS["adj_tempo"]
            team_obj["barthag"] = team_obj.get("barthag") if team_obj.get("barthag") is not None else _BAD_DEFAULTS["barthag"]
            # Attach school lat/lon for proximity calculations
            locs = _load_school_locations()
            if tname in locs and not team_obj.get("location"):
                team_obj["location"] = locs[tname]
            enriched += 1
            enriched_names.add(tname)

    # Compute eff_rank (1-indexed barthag rank among all bracket teams)
    all_teams = []
    for region in bracket.values():
        for team_obj in region.values():
            b = team_obj.get("barthag")
            if b is not None:
                all_teams.append((team_obj, b))
    all_teams.sort(key=lambda x: -x[1])
    for i, (team_obj, _) in enumerate(all_teams):
        team_obj["eff_rank"] = i + 1

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
    base_margin = result.get("base_margin", margin)
    if abs(base_margin) >= 3:
        better = team_a["team"] if base_margin > 0 else team_b["team"]
        factors.append(f"{better}: {abs(base_margin):.0f}-pt matchup efficiency advantage")
    elif abs(margin) >= 3:
        better = team_a["team"] if margin > 0 else team_b["team"]
        factors.append(f"{better}: {abs(margin):.0f}-pt projected advantage")

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


def get_matchup_analysis_display(team_a, team_b, data_dir=None, year=None, config=DEFAULT_CONFIG,
                                  game_site=None, region=None, round_name=None, venue_city=None):
    """Return matchup analysis in pick-compatible format for API/frontend display.
    team_a, team_b can be team names (str) or team dicts. Returns dict with stats_a,
    stats_b, key_factors, insight, head_to_head, upset_alert, win_prob_a, etc.
    game_site: [lat, lon] of game location for proximity calculation.
    venue_city: human-readable city name for display.
    """
    a = enrich_team(team_a) if isinstance(team_a, dict) else enrich_team({"team": team_a})
    b = enrich_team(team_b) if isinstance(team_b, dict) else enrich_team({"team": team_b})
    a.setdefault("seed", 8)
    b.setdefault("seed", 8)
    # If game_site not provided, try to infer from venues
    if game_site is None and year and round_name:
        venues = _load_venues(year)
        sa, sb = a.get("seed"), b.get("seed")
        game_site = _get_game_site(venues, region, round_name, seed_a=sa, seed_b=sb)
        if venue_city is None:
            venue_city = _get_venue_city(venues, region, round_name, seed_a=sa, seed_b=sb)
    result = predict_game(a, b, game_site=game_site, config=config, round_name=round_name or "Round of 64")
    hist = get_seed_matchup_history(a.get("seed", 8), b.get("seed", 8))
    h2h = get_head_to_head(a["team"], b["team"], data_dir=data_dir, current_year=year or 2026)
    upset_alert = _compute_upset_alert(
        a["seed"], b["seed"], result["predicted_margin"],
        result["win_prob_a"], hist)
    pick_team = a if result["win_prob_a"] >= 0.5 else b
    rnd_name = round_name or "Round of 64"
    rnd_of = {"Round of 64": 64, "Round of 32": 32, "Sweet 16": 16, "Elite 8": 8,
              "Final Four": 4, "Championship": 2}.get(rnd_name, 64)
    pick_dict = _make_pick_dict(0, rnd_of, rnd_name, region, a, b, result, pick_team,
                                data_dir=data_dir, year=year or 2026)
    pick_dict["upset_alert"] = upset_alert
    pick_dict["venue_city"] = venue_city
    return pick_dict


def _game_id_for_pick(region, round_of, game_index, quadrant_order=None):
    """Map (region, round, index) to frontend game_id.
    Region games: {region}-{round}-{gi}
    Final Four: FF-4-0, FF-4-1 (order from quadrant_order)
    Championship: FF-2-0
    """
    if round_of == 4:
        return f"FF-4-{game_index}"
    if round_of == 2:
        return "FF-2-0"
    return f"{region}-{round_of}-{game_index}"


def resolve_ff_pairs(quadrant_order, ff_matchups=None):
    """Resolve Final Four region pairings from bracket layout metadata.

    `quadrant_order` is [TL, TR, BR, BL]. `ff_matchups` is expected to be a
    pair of index pairs referencing `quadrant_order`, e.g. [[0, 3], [1, 2]].
    Falls back to the legacy TL-vs-BL and TR-vs-BR mapping when layout metadata
    is missing or malformed.
    """
    if not quadrant_order or len(quadrant_order) < 4:
        return []
    if ff_matchups:
        pairs = []
        for pair in ff_matchups[:2]:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                a_idx, b_idx = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                continue
            if 0 <= a_idx < len(quadrant_order) and 0 <= b_idx < len(quadrant_order):
                pairs.append((quadrant_order[a_idx], quadrant_order[b_idx]))
        if len(pairs) == 2:
            return pairs
    return [(quadrant_order[0], quadrant_order[3]), (quadrant_order[1], quadrant_order[2])]


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
                    data_dir=None, year=None, game_id=None):
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
    def _team_stats(t):
        def _r(v, d=1): return round(v, d) if v is not None else None
        wins  = t.get("wins")
        games = t.get("games")
        losses = (games - wins) if (wins is not None and games is not None) else None
        return {
            # Core efficiency (Torvik)
            "adj_o":        _r(t.get("adj_o"), 1),
            "adj_d":        _r(t.get("adj_d"), 1),
            "adj_tempo":    _r(t.get("adj_tempo"), 1),
            "win_pct":      _r(t.get("win_pct"), 3),
            "wins":         wins,
            "losses":       losses,
            "barthag":      _r(t.get("barthag"), 3),
            "sos":          _r(t.get("sos"), 3),
            "elite_sos":    _r(t.get("elite_sos"), 3),
            "wab":          _r(t.get("wab"), 1),
            "conf_rating":  t.get("conf_rating"),
            "conf_strength": _r(t.get("conf_strength_score"), 3),
            "conf_win_pct": _r(t.get("conf_win_pct"), 3),
            # EvanMiya ratings
            "em_adj_o":     _r(t.get("em_adj_o"), 1),
            "em_adj_d":     _r(t.get("em_adj_d"), 1),
            "em_off_rank":  t.get("em_off_rank"),
            "em_def_rank":  t.get("em_def_rank"),
            "em_opponent_adjust": _r(t.get("em_opponent_adjust"), 3),
            "em_tempo":     _r(t.get("em_tempo"), 1),
            "em_bpr":       _r(t.get("em_bpr"), 1),
            "em_obpr":      _r(t.get("em_obpr"), 1),
            "em_dbpr":      _r(t.get("em_dbpr"), 1),
            "em_runs_per_game": _r(t.get("em_runs_per_game"), 2),
            "em_runs_conceded": _r(t.get("em_runs_conceded"), 2),
            "em_depth_score": _r(t.get("em_depth_score"), 3),
            "em_big_bpr":   _r(t.get("em_big_bpr"), 2),
            "em_guard_bpr": _r(t.get("em_guard_bpr"), 2),
            "em_top5_bpr":  _r(t.get("em_top5_bpr"), 1),
            "em_creator_count": t.get("em_creator_count"),
            "em_star_concentration": _r(t.get("em_star_concentration"), 3),
            # Scoring
            "ppg":          _r(t.get("ppg"), 1),
            "opp_ppg":      _r(t.get("opp_ppg"), 1),
            "ppp_off":      _r(t.get("ppp_off"), 3),
            "ppp_def":      _r(t.get("ppp_def"), 3),
            # Shooting
            "efg_pct":      _r(t.get("efg_pct"), 3),
            "efg_d":        _r(t.get("efg_d"), 3),
            "three_pt_pct": _r(t.get("three_pt_pct"), 3),
            "three_pt_pct_d": _r(t.get("three_pt_pct_d"), 3),
            "three_pt_rate": _r(t.get("three_pt_rate") or t.get("three_rate"), 3),
            "three_pt_rate_d": _r(t.get("three_pt_rate_d"), 3),
            "two_pt_pct":   _r(t.get("two_pt_pct"), 3),
            "two_pt_pct_d": _r(t.get("two_pt_pct_d"), 3),
            "ft_pct":       _r(t.get("ft_pct"), 3),
            "ft_rate":      _r(t.get("ft_rate"), 3),
            "ft_rate_d":    _r(t.get("ft_rate_d"), 3),
            # Rebounding & ball control
            "orb_rate":     _r(t.get("orb_rate"), 3),
            "opp_orb_rate": _r(t.get("opp_orb_rate"), 3),
            "to_rate":      _r(t.get("to_rate"), 3),
            "to_rate_d":    _r(t.get("to_rate_d"), 3),
            # Defense
            "blk_rate":     _r(t.get("blk_rate"), 3),
            "ast_rate":     _r(t.get("ast_rate"), 3),
            "opp_ast_rate": _r(t.get("opp_ast_rate"), 3),
            # Roster
            "avg_experience": _r(t.get("avg_experience"), 2),
            "experience":   _r(t.get("experience"), 3),
            "top_player":   t.get("top_player"),
            "top_player_bpr": _r(t.get("top_player_bpr"), 2),
            "star_score":   _r(t.get("star_score"), 2),
            # Momentum
            "momentum":     _r(t.get("momentum"), 2),
            "conf_tourney_momentum": _r(t.get("conf_tourney_momentum"), 2),
            # Quality metrics
            "qual_o":       _r(t.get("qual_o"), 1),
            "qual_d":       _r(t.get("qual_d"), 1),
            "qual_barthag": _r(t.get("qual_barthag"), 3),
            # Injuries
            "injuries":     t.get("injuries"),
            "injury_impact": _r(t.get("injury_impact"), 1),
        }

    d = {
        "game_num": game_num,
        "round": round_of,
        "round_name": round_name,
        "region": region,
        "game_id": game_id,
        "team_a": a["team"], "seed_a": a["seed"],
        "team_b": b["team"], "seed_b": b["seed"],
        "pick": pick_team["team"],
        "pick_seed": pick_team["seed"],
        # Win probabilities (both sides, not just the higher)
        "win_prob":   round(max(result["win_prob_a"], result["win_prob_b"]), 4),
        "win_prob_a": round(result["win_prob_a"], 4),
        "win_prob_b": round(result["win_prob_b"], 4),
        # Score prediction
        "projected_spread": spread_amt,
        "spread_fav": f"{fav_team} -{spread_amt}",
        "spread_dog": f"{dog_team} +{spread_amt}",
        "projected_score": f"{result['predicted_score_a']:.0f}-{result['predicted_score_b']:.0f}",
        "predicted_score_a": round(result["predicted_score_a"], 1),
        "predicted_score_b": round(result["predicted_score_b"], 1),
        "predicted_total":   round(result["predicted_score_a"] + result["predicted_score_b"], 1),
        # Margin decomposition
        "base_margin":        round(result["base_margin"], 1),
        "factor_margin":      round(result["factor_margin"], 1),
        "possession_margin":  round(result.get("possession_margin", 0), 2),
        "ft_margin":          round(result.get("ft_margin", 0), 2),
        "sos_margin":         round(result.get("sos_margin", 0), 2),
        "runs_margin":        round(result.get("runs_margin", 0), 2),
        "big_bpr_margin":     round(result.get("big_bpr_margin", 0), 2),
        "guard_bpr_margin":   round(result.get("guard_bpr_margin", 0), 2),
        "creator_margin":     round(result.get("creator_margin", 0), 2),
        "foul_rate_margin":   round(result.get("foul_rate_margin", 0), 2),
        "upset_tolerance_bonus": round(result.get("upset_tolerance_bonus", 0), 2),
        "predicted_margin":   round(result["predicted_margin"], 1),
        # Model signal probabilities
        "efficiency_prob":    round(result["efficiency_prob"], 4),
        "seed_prob":          round(result["seed_prob"], 4),
        "volatility":         round(result.get("volatility", 1.0), 3),
        "game_stdev":         round(result.get("game_stdev", 11.0), 1),
        # Per-factor bonuses
        "factors_a": result["factors_a"],
        "factors_b": result["factors_b"],
        # Team efficiency stats (for display in panel)
        "stats_a": _team_stats(a),
        "stats_b": _team_stats(b),
        # Labels / narrative
        "confidence": _confidence_tier(result["win_prob_a"]),
        "upset_rating": _upset_rating(a["seed"], b["seed"], result["win_prob_a"], result.get("volatility", 1.0)),
        "variability": _variability_label(result.get("volatility", 1.0), abs(a["seed"] - b["seed"]), result["win_prob_a"]),
        "key_factors": _generate_key_factors(result, a, b),
        "insight": _generate_insight(result, hist, a, b),
        "historical": f"{hist['higher_seed_wins']}-{hist['lower_seed_wins']}" if hist else None,
        "historical_win_pct": round(hist["higher_seed_win_pct"], 3) if hist else None,
        "historical_avg_margin": round(hist["avg_margin"], 1) if hist else None,
        "head_to_head": h2h,
        "upset_alert": upset_alert,
        "pick_dampened": result.get("pick_dampened", False),
    }
    d["stats_a"]["injury_impact"] = round(abs(result["factors_a"].get("injuries", 0)), 1)
    d["stats_b"]["injury_impact"] = round(abs(result["factors_b"].get("injuries", 0)), 1)
    return d


def generate_bracket_picks(bracket, config=DEFAULT_CONFIG, upset_aggression=0.0, quadrant_order=None,
                          ff_matchups=None, data_dir=None, year=None, locked_picks=None):
    """Generate a complete 63-game bracket with analysis for every pick.

    Args:
        bracket: dict of region -> {seed: team_dict}
        config: ModelConfig
        upset_aggression: 0.0 (always pick favorite) to 1.0 (chaos mode)
        quadrant_order: [TL, TR, BR, BL] region names for bracket layout
        ff_matchups: optional pairings as quadrant index pairs, e.g. [[0, 3], [1, 2]]
        data_dir: path to data dir for head-to-head lookup
        year: current bracket year for head-to-head
        locked_picks: optional dict of game_id -> team_name for user-locked picks

    Returns dict with 'picks', 'champion', 'final_four', 'biggest_upsets', 'most_uncertain_games'.
    """
    if quadrant_order is None:
        quadrant_order = REGIONS[:4]
    locked_picks = locked_picks or {}
    _h2h_kw = {"data_dir": data_dir, "year": year or 2026}
    picks = []
    game_num = 0

    venues = _load_venues(year) if year else {}

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
        for gi, (seed_a, seed_b) in enumerate(FIRST_ROUND_MATCHUPS):
            a = teams.get(seed_a)
            b = teams.get(seed_b)
            if not a or not b:
                continue
            game_num += 1
            gid = _game_id_for_pick(region, 64, gi)
            game_site = _get_game_site(venues, region, "Round of 64", seed_a=seed_a, seed_b=seed_b)
            venue_city = _get_venue_city(venues, region, "Round of 64", seed_a=seed_a, seed_b=seed_b)
            result = predict_game(a, b, game_site=game_site, config=config, round_name="Round of 64")
            locked_team = locked_picks.get(gid)
            if locked_team in (a["team"], b["team"]):
                pick_team = a if locked_team == a["team"] else b
            else:
                pick_a = _should_pick_upset(result["win_prob_a"], a["seed"], b["seed"], upset_aggression)
                pick_team = a if pick_a else b
            pd = _make_pick_dict(game_num, 64, "Round of 64", region, a, b, result, pick_team, game_id=gid, **_h2h_kw)
            pd["venue_city"] = venue_city
            picks.append(pd)
            r64_winners.append(pick_team)

        def _play_round(teams_in, round_of, round_name):
            n = len(teams_in) // 2
            winners = []
            for i in range(0, len(teams_in), 2):
                if i + 1 >= len(teams_in):
                    winners.append(teams_in[i])
                    continue
                gi = i // 2
                a, b = teams_in[i], teams_in[i + 1]
                nonlocal game_num
                game_num += 1
                gid = _game_id_for_pick(region, round_of, gi)
                sa, sb = a.get("seed"), b.get("seed")
                game_site = _get_game_site(venues, region, round_name, seed_a=sa, seed_b=sb)
                venue_city = _get_venue_city(venues, region, round_name, seed_a=sa, seed_b=sb)
                result = predict_game(a, b, game_site=game_site, config=config, round_name=round_name)
                locked_team = locked_picks.get(gid)
                if locked_team in (a["team"], b["team"]):
                    pick_team = a if locked_team == a["team"] else b
                else:
                    pick_a = _should_pick_upset(result["win_prob_a"], a["seed"], b["seed"], upset_aggression)
                    pick_team = a if pick_a else b
                pd = _make_pick_dict(game_num, round_of, round_name, region, a, b, result, pick_team, game_id=gid, **_h2h_kw)
                pd["venue_city"] = venue_city
                picks.append(pd)
                winners.append(pick_team)
            return winners

        r32_winners = _play_round(r64_winners, 32, "Round of 32")
        s16_winners = _play_round(r32_winners, 16, "Sweet 16")
        e8_winners = _play_round(s16_winners, 8, "Elite 8")
        if e8_winners:
            region_winners[region] = e8_winners[0]

    ff_pairs = resolve_ff_pairs(quadrant_order, ff_matchups)
    ff_winners = []
    for ff_gi, (r_a, r_b) in enumerate(ff_pairs):
        a = region_winners.get(r_a)
        b = region_winners.get(r_b)
        if not a and not b:
            continue
        if not a or not b:
            ff_winners.append(a or b)
            continue
        game_num += 1
        gid = _game_id_for_pick(None, 4, ff_gi)
        f4_site = _get_game_site(venues, None, "Final Four")
        f4_city = _get_venue_city(venues, None, "Final Four")
        result = predict_game(a, b, game_site=f4_site, config=config, round_name="Final Four")
        locked_team = locked_picks.get(gid)
        if locked_team in (a["team"], b["team"]):
            pick_team = a if locked_team == a["team"] else b
        else:
            pick_a = _should_pick_upset(result["win_prob_a"], a["seed"], b["seed"], upset_aggression)
            pick_team = a if pick_a else b
        pd = _make_pick_dict(game_num, 4, "Final Four", None, a, b, result, pick_team, game_id=gid)
        pd["venue_city"] = f4_city
        picks.append(pd)
        ff_winners.append(pick_team)

    # Championship
    champion = None
    if len(ff_winners) >= 2:
        a, b = ff_winners[0], ff_winners[1]
        game_num += 1
        gid = "FF-2-0"
        champ_site = _get_game_site(venues, None, "Championship")
        champ_city = _get_venue_city(venues, None, "Championship")
        result = predict_game(a, b, game_site=champ_site, config=config, round_name="Championship")
        locked_team = locked_picks.get(gid)
        if locked_team in (a["team"], b["team"]):
            pick_team = a if locked_team == a["team"] else b
        else:
            pick_a = _should_pick_upset(result["win_prob_a"], a["seed"], b["seed"], upset_aggression)
            pick_team = a if pick_a else b
        pd = _make_pick_dict(game_num, 2, "Championship", None, a, b, result, pick_team, game_id=gid, **_h2h_kw)
        pd["venue_city"] = champ_city
        picks.append(pd)
        champion = pick_team["team"]

    final_four = []
    for r_a, r_b in ff_pairs:
        if r_a in region_winners:
            final_four.append(region_winners[r_a]["team"])
        if r_b in region_winners:
            final_four.append(region_winners[r_b]["team"])
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
