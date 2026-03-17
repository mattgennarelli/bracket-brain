"""
best_bets.py — Compare our model's predictions against today's Vegas lines.

Fetches live NCAAB odds (spread, moneyline, over/under) from an odds provider.
Providers: The Odds API (default) or BetStack. Set ODDS_PROVIDER=odds_api|betstack.

Requirements:
  pip install requests
  ODDS_API_KEY for The Odds API (https://the-odds-api.com, 500 req/month free)
  BETSTACK_API_KEY for BetStack (https://betstack.dev, free forever, 1 req/60s)

Usage:
  export ODDS_API_KEY=your_key_here
  python scripts/best_bets.py

  # Use BetStack instead:
  export ODDS_PROVIDER=betstack
  export BETSTACK_API_KEY=your_betstack_key
  python scripts/best_bets.py

  python scripts/best_bets.py --api-key YOUR_KEY --year 2026
"""
import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

try:
    import requests
except ImportError:
    # Don't sys.exit at import time — that crashes pytest collection for ALL tests.
    # Instead, defer the error to runtime so tests can safely import this module.
    requests = None  # type: ignore[assignment]

from engine import predict_game, ModelConfig, _normalize_team_for_match, enrich_team, load_teams_merged
from odds_provider import get_provider, get_api_key

# Default edge thresholds to flag as a "bet"
DEFAULT_ML_EDGE = 0.10      # 10% win probability edge over implied odds (raised from 7%)
DEFAULT_SPREAD_EDGE = 7.0   # model cover margin must exceed 7 pts
DEFAULT_TOTAL_EDGE = 10.0   # model total differs from Vegas total by 10+ pts
DEFAULT_MIN_MODEL_CONFIDENCE = 0.58  # skip ML bets when model probability < 58%
DEFAULT_MIN_ML_PROB_FOR_DOG = 0.40   # skip ML when betting underdog with model prob < 40%
DEFAULT_MIN_SPREAD_COVER_MARGIN = 3.0  # skip spread when cover margin < 3 pts
DEFAULT_MAX_3STAR_PICKS = 0   # 3-star picks disabled: 0/3 hit rate (0% ROI). Re-enable when positive.
DEFAULT_MAX_2STAR_PICKS = 8   # cap 2-star picks per day

# Maximum moneyline price to bet — don't chase heavy chalk beyond this.
# At -200, break-even accuracy is 66.7%. Our model rarely beats that on big favorites.
DEFAULT_MAX_ML_PRICE = -200  # skip ML bets with juice worse than -200

# Bias correction: model consistently predicts totals ~1.8 pts too LOW vs actuals.
# Applied before edge calculation: model_total_corrected = model_total + TOTAL_BIAS_CORRECTION
# Calibrated on 187 tournament games 2023-2025 (total_bias = -1.8 pts).
TOTAL_BIAS_CORRECTION = 1.8


# ---------------------------------------------------------------------------
# ODDS FETCHING (via odds_provider)
# ---------------------------------------------------------------------------

def _require_requests():
    """Raise a clear error if requests wasn't installed. Called lazily before any network call."""
    if requests is None:
        raise ImportError("'requests' not installed. Run: pip install requests")


def fetch_today_odds(api_key):
    """Fetch today's NCAAB games from the active provider (ODDS_PROVIDER)."""
    _require_requests()
    provider = get_provider()
    try:
        return provider.fetch_games(api_key)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def parse_game(game):
    """Parse raw game from active provider into common dict shape."""
    return get_provider().parse_game(game)


def _american_to_prob(odds):
    """Convert American moneyline odds to raw implied probability (not devigged)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def _devig(p1, p2):
    """Remove bookmaker margin (vig) — normalize two raw probabilities to sum=1."""
    total = p1 + p2
    if total <= 0:
        return 0.5, 0.5
    return p1 / total, p2 / total


def _american_to_decimal(odds):
    """Convert American odds to decimal (European) format."""
    if odds < 0:
        return 1 + 100 / abs(odds)
    else:
        return 1 + odds / 100


# ---------------------------------------------------------------------------
# MODEL LOOKUP
# ---------------------------------------------------------------------------

# Manual overrides for Odds API names that can't be resolved by mascot stripping
_ODDS_MANUAL = {
    "gw revolutionaries": "george washington",
    "gw":                 "george washington",
    "loyola chi":         "loyola chicago",
    "loyola chi ramblers": "loyola chicago",
    "loyola (chi)":       "loyola chicago",
    "boston univ":        "boston university",
    "miami oh":           "miami oh",
    "miami (oh)":         "miami oh",
    "uc san diego":       "uc san diego",
    "stephen f austin":   "stephen f austin",
    "st bonaventure":     "st bonaventure",
    "st johns red storm": "St. John's",   # Odds API uses mascot; _prep expands St->State wrongly
    "state johns red storm": "St. John's",
    "texas am":           "texas a&m",
    "texas a m":          "texas a&m",
    "vmi":                "virginia military",
    "iupui":              "iupui",
}


def _prep_odds_name(name):
    """Normalize an Odds API team name: expand abbreviations, strip unicode."""
    # Unicode → ASCII
    name = name.replace("é", "e").replace("É", "E")
    # Expand abbreviations (word-boundary safe)
    name = re.sub(r"\bSt\b\.?", "State", name)
    name = re.sub(r"\bUniv\b\.?", "University", name)
    name = re.sub(r"\bNo\b\.", "North", name)
    # Remove parentheticals: "Miami (OH)" → "Miami OH"
    name = re.sub(r"\((\w+)\)", r"\1", name)
    return name.strip()


def load_team_stats(year):
    """Load teams_merged for the given year."""
    teams = load_teams_merged(DATA_DIR, year)
    if not teams:
        for y in range(year - 1, year - 4, -1):
            teams = load_teams_merged(DATA_DIR, y)
            if teams:
                print(f"  Warning: using {y} team data (no {year} data found)")
                break
    return teams


def lookup_team(name, teams_merged):
    """Match an Odds API team name (with mascot) to our teams_merged dict.

    Strategy:
      1. Direct normalized match
      2. After abbreviation expansion (St→State etc.)
      3. Progressive mascot stripping — drop 1, 2, 3 trailing words
      4. Manual overrides for Odds API quirks
      5. Conservative substring match (≥80% length overlap)
    """
    def _try(key):
        return teams_merged.get(key)

    # Step 1: as-is
    key = _normalize_team_for_match(name)
    if _try(key):
        return dict(_try(key))

    # Step 2: with abbreviation expansion
    prepped = _prep_odds_name(name)
    key2 = _normalize_team_for_match(prepped)
    if key2 != key and _try(key2):
        return dict(_try(key2))

    # Step 3: progressive mascot stripping on the expanded name
    words = prepped.split()
    for n in range(1, min(4, len(words))):
        shorter = " ".join(words[:-n])
        if not shorter:
            break
        k = _normalize_team_for_match(shorter)
        if _try(k):
            return dict(_try(k))

    # Step 4: manual overrides — check all candidate keys
    for candidate in (key, key2):
        manual_key = _ODDS_MANUAL.get(candidate)
        if manual_key:
            mk = _normalize_team_for_match(manual_key)
            if _try(mk):
                return dict(_try(mk))
    # Also try stripping words then checking manual map
    words2 = prepped.split()
    for n in range(1, min(4, len(words2))):
        shorter_key = _normalize_team_for_match(" ".join(words2[:-n]))
        manual_key = _ODDS_MANUAL.get(shorter_key)
        if manual_key:
            mk = _normalize_team_for_match(manual_key)
            if _try(mk):
                return dict(_try(mk))

    # Step 5: conservative substring match — both keys must be ≥80% of longer
    for k, v in teams_merged.items():
        shorter, longer = (key2, k) if len(key2) <= len(k) else (k, key2)
        if len(shorter) >= 6 and shorter in longer and len(shorter) >= len(longer) * 0.80:
            return dict(v)

    return None


def run_model(home_stats, away_stats, config):
    """Run predict_game and return result dict."""
    home = enrich_team(dict(home_stats))
    away = enrich_team(dict(away_stats))
    home.setdefault("seed", 8)
    away.setdefault("seed", 8)
    return predict_game(home, away, config=config)


# ---------------------------------------------------------------------------
# EDGE CALCULATIONS
# ---------------------------------------------------------------------------

def ml_edge(model_prob_home, ml_home, ml_away):
    """Return (home_edge, away_edge) in probability points vs. devigged implied odds."""
    if ml_home is None or ml_away is None:
        return None, None
    raw_home = _american_to_prob(ml_home)
    raw_away = _american_to_prob(ml_away)
    impl_home, impl_away = _devig(raw_home, raw_away)
    return model_prob_home - impl_home, (1 - model_prob_home) - impl_away


def spread_edge(model_margin, vegas_spread_home):
    """Return cover margin from home team's perspective.

    Positive = home team covers their spread.
    Negative = away team covers their spread.

    Formula: model_margin + vegas_spread_home
      e.g. Iowa State -30.5 (spread_home=-30.5), model +3.8:
           3.8 + (-30.5) = -26.7 → away (Arizona State) covers +30.5 by 26.7 pts.
      e.g. Auburn -7.5 (spread_home=-7.5), model +14:
           14 + (-7.5) = +6.5 → home (Auburn) covers -7.5 by 6.5 pts.
    """
    if vegas_spread_home is None:
        return None
    return model_margin + vegas_spread_home


def total_edge(model_total, vegas_total):
    """Return model_total - vegas_total. Positive = lean Over."""
    if vegas_total is None:
        return None
    return model_total - vegas_total


def kelly_fraction(model_prob, decimal_odds, fraction=0.10):
    """Compute fractional Kelly bet size.

    Args:
        model_prob: our estimated probability of winning
        decimal_odds: payout in decimal format (e.g., 2.0 for even money)
        fraction: Kelly fraction (0.10 = tenth Kelly — conservative; quarter-Kelly
                  almost always hits the 5% cap given our high-edge thresholds)

    Returns:
        Fraction of bankroll to bet (0.0 if no edge), capped at 5%.
    """
    if decimal_odds <= 1.0:
        return 0.0
    edge = model_prob * decimal_odds - 1.0
    if edge <= 0:
        return 0.0
    full_kelly = edge / (decimal_odds - 1.0)
    sized = full_kelly * fraction
    return min(sized, 0.05)  # hard cap at 5% of bankroll


def cover_prob(cover_margin, stdev=11.0):
    """Probability of covering given cover margin (positive = covers) and game stdev."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(cover_margin / (stdev * sqrt(2))))


def payout_str(american_odds):
    """Format American odds with sign."""
    if american_odds is None:
        return "N/A"
    return f"{'+' if american_odds > 0 else ''}{int(american_odds)}"


def get_best_bets_json(api_key, year=None, ml_min=None, spread_min=None, total_min=None,
                       min_ml_prob_for_dog=None, min_spread_cover=None,
                       max_3star=None, max_2star=None):
    """Fetch today's odds, run model, return bets as JSON-serializable list sorted by start time.

    Returns list of dicts, each with: commence_time, away_team, home_team, bet_type, bet_side/bet_team,
    bet_odds, edge, stars, model_prob, implied_prob, etc. Empty list if no API key or no games.
    """
    if not api_key:
        return []
    year = year or datetime.now().year
    min_ml = ml_min if ml_min is not None else DEFAULT_ML_EDGE
    min_spread = spread_min if spread_min is not None else DEFAULT_SPREAD_EDGE
    min_total = total_min if total_min is not None else DEFAULT_TOTAL_EDGE
    min_ml_dog = min_ml_prob_for_dog if min_ml_prob_for_dog is not None else DEFAULT_MIN_ML_PROB_FOR_DOG
    min_cover = min_spread_cover if min_spread_cover is not None else DEFAULT_MIN_SPREAD_COVER_MARGIN
    cap_3 = max_3star if max_3star is not None else DEFAULT_MAX_3STAR_PICKS
    cap_2 = max_2star if max_2star is not None else DEFAULT_MAX_2STAR_PICKS

    raw_games = fetch_today_odds(api_key)
    if not raw_games:
        return []

    teams = load_team_stats(year)
    config = ModelConfig()
    bets = []
    now_utc = datetime.now(timezone.utc)

    for raw in raw_games:
        game = parse_game(raw)
        try:
            commence = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
            if commence <= now_utc:
                continue
        except Exception:
            pass

        home_name = game["home_team"]
        away_name = game["away_team"]
        home_stats = lookup_team(home_name, teams)
        away_stats = lookup_team(away_name, teams)
        if home_stats is None or away_stats is None:
            continue
        # Skip if either team has no efficiency data — model output would be fabricated
        if home_stats.get("adj_o") is None or home_stats.get("adj_d") is None:
            continue
        if away_stats.get("adj_o") is None or away_stats.get("adj_d") is None:
            continue

        home_stats["team"] = home_name
        away_stats["team"] = away_name
        result = run_model(home_stats, away_stats, config)
        model_prob_home = result["win_prob_a"]
        model_margin = result["predicted_margin"]
        model_total_raw = result["predicted_score_a"] + result["predicted_score_b"]
        # Apply bias correction: model under-predicts totals by ~1.8 pts on avg
        model_total = model_total_raw + TOTAL_BIAS_CORRECTION

        base = {
            "home_team": home_name,
            "away_team": away_name,
            "commence_time": game["commence_time"],
            "model_prob_home": model_prob_home,
            "model_margin": model_margin,
            "model_total": model_total,
        }

        min_conf = DEFAULT_MIN_MODEL_CONFIDENCE

        # ML bets (with underdog gating + max price filter)
        h_edge, a_edge = ml_edge(model_prob_home, game["ml_home"], game["ml_away"])
        if h_edge is not None:
            raw_h = _american_to_prob(game["ml_home"])
            raw_a = _american_to_prob(game["ml_away"])
            ih, ia = _devig(raw_h, raw_a)
            # Home: skip if we're the dog (ih < 0.5) and model prob < min_ml_dog
            # Also skip if price is worse than max ML price (too much chalk juice)
            home_price_ok = game["ml_home"] is None or game["ml_home"] >= DEFAULT_MAX_ML_PRICE
            home_ok = h_edge >= min_ml and model_prob_home >= min_conf and home_price_ok
            if ih < 0.5 and model_prob_home < min_ml_dog:
                home_ok = False  # underdog ML gating
            if home_ok:
                dec_h = _american_to_decimal(game["ml_home"])
                k_h = kelly_fraction(model_prob_home, dec_h)
                bets.append(({**base, "bet_type": "ml", "bet_side": home_name,
                             "bet_odds": game["ml_home"], "implied_prob": ih,
                             "model_prob": model_prob_home,
                             "kelly_size": round(k_h, 4),
                             "kelly_units": round(k_h * 100, 2)}, h_edge))
            # Away: skip if we're the dog (ia < 0.5) and model prob < min_ml_dog
            away_price_ok = game["ml_away"] is None or game["ml_away"] >= DEFAULT_MAX_ML_PRICE
            away_ok = a_edge >= min_ml and (1 - model_prob_home) >= min_conf and away_price_ok
            if ia < 0.5 and (1 - model_prob_home) < min_ml_dog:
                away_ok = False  # underdog ML gating
            if away_ok:
                dec_a = _american_to_decimal(game["ml_away"])
                k_a = kelly_fraction(1 - model_prob_home, dec_a)
                bets.append(({**base, "bet_type": "ml", "bet_side": away_name,
                             "bet_odds": game["ml_away"], "implied_prob": ia,
                             "model_prob": 1 - model_prob_home,
                             "kelly_size": round(k_a, 4),
                             "kelly_units": round(k_a * 100, 2)}, a_edge))

        # Spread bets (skip when cover margin too slim — no-pick filter)
        sp_edge_val = spread_edge(model_margin, game["spread_home"])
        if sp_edge_val is not None and abs(sp_edge_val) >= min_spread and abs(sp_edge_val) >= min_cover:
            if sp_edge_val > 0:
                bet_team = home_name
                bet_spread = game["spread_home"]
                cp = cover_prob(sp_edge_val)
            else:
                bet_team = away_name
                bet_spread = -game["spread_home"]
                cp = cover_prob(abs(sp_edge_val))
            dec_sp = _american_to_decimal(game["spread_line"]) if game["spread_line"] else 1.909
            k_sp = kelly_fraction(cp, dec_sp)
            bets.append(({**base, "bet_type": "spread",
                         "bet_team": bet_team, "bet_spread": bet_spread,
                         "bet_odds": game["spread_line"],
                         "vegas_spread": game["spread_home"],
                         "cover_margin": abs(sp_edge_val),
                         "model_margin": model_margin,
                         "kelly_size": round(k_sp, 4),
                         "kelly_units": round(k_sp * 100, 2)}, abs(sp_edge_val)))

        # Total bets — calibrated for NCAA tournament games (bias-corrected by +1.8 pts).
        # Do NOT run against regular-season or conference tournament games:
        # mid-major adj_o/d produces +14.7pt OVER bias outside the tourney.
        tot_e = total_edge(model_total, game["total_line"])
        if tot_e is not None and abs(tot_e) >= min_total:
            side = "OVER" if tot_e > 0 else "UNDER"
            # cover_margin for total: how far our prediction exceeds the line
            cp_tot = cover_prob(abs(tot_e))
            dec_tot = 1.909  # typical -110 for totals
            k_tot = kelly_fraction(cp_tot, dec_tot)
            bets.append(({**base, "bet_type": "total", "bet_side": side,
                         "vegas_total": game["total_line"],
                         "model_total": model_total,
                         "kelly_size": round(k_tot, 4),
                         "kelly_units": round(k_tot * 100, 2)}, tot_e))

    # Sort by commence_time, then by edge magnitude
    bets.sort(key=lambda x: (x[0]["commence_time"], -abs(x[1])))

    # Build JSON-serializable output with star ratings
    thresh_ml = [min_ml, min_ml * 1.5, min_ml * 2.5]
    thresh_sp = [min_spread, min_spread * 1.5, min_spread * 2.0]
    thresh_tot = [min_total, min_total * 1.4, min_total * 2.0]
    rated = []
    for g, edge in bets:
        rec = dict(g)
        rec["edge"] = float(edge)
        thresh = thresh_ml if g["bet_type"] == "ml" else thresh_sp if g["bet_type"] == "spread" else thresh_tot
        rec["stars"] = star_rating(abs(edge), thresh)
        rated.append(rec)

    # Apply star-tier issuance caps (3-star underperforming per audit)
    count_3, count_2 = 0, 0
    out = []
    for rec in rated:
        s = rec.get("stars", "")
        if s == "★★★":
            if count_3 >= cap_3:
                continue
            count_3 += 1
        elif s == "★★":
            if count_2 >= cap_2:
                continue
            count_2 += 1
        out.append(rec)
    return out


def get_full_card_json(api_key, year=None):
    """Return every game today with the model's best lean for each market.

    Unlike get_best_bets_json, NO edge threshold is applied — every game is
    included, and the model's preferred side for ML/spread/total is always
    returned.  Stars are assigned by the same threshold rules so the UI can
    highlight high-confidence leans.

    Returns a list of game dicts, each with:
        home_team, away_team, commence_time, model_prob_home,
        model_margin, model_total, data_available, picks (list of 1-3)
    """
    if not api_key:
        return []
    year = year or datetime.now().year

    raw_games = fetch_today_odds(api_key)
    if not raw_games:
        return []

    teams = load_team_stats(year)
    config = ModelConfig()
    now_utc = datetime.now(timezone.utc)
    out = []

    for raw in raw_games:
        game = parse_game(raw)
        try:
            commence = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
            if commence <= now_utc:
                continue
        except Exception:
            pass

        home_name = game["home_team"]
        away_name = game["away_team"]
        home_stats = lookup_team(home_name, teams)
        away_stats = lookup_team(away_name, teams)

        # Mark games where we lack efficiency data
        data_ok = (
            home_stats is not None and away_stats is not None
            and home_stats.get("adj_o") is not None and home_stats.get("adj_d") is not None
            and away_stats.get("adj_o") is not None and away_stats.get("adj_d") is not None
        )

        game_rec = {
            "home_team": home_name,
            "away_team": away_name,
            "commence_time": game["commence_time"],
            "data_available": data_ok,
            "picks": [],
        }

        if not data_ok:
            out.append(game_rec)
            continue

        home_stats["team"] = home_name
        away_stats["team"] = away_name
        result = run_model(home_stats, away_stats, config)
        model_prob_home = result["win_prob_a"]
        model_margin = result["predicted_margin"]
        model_total = result["predicted_score_a"] + result["predicted_score_b"]

        game_rec["model_prob_home"] = round(model_prob_home, 4)
        game_rec["model_margin"] = round(model_margin, 1)
        game_rec["model_total"] = round(model_total, 1)

        # ML pick — always take the side with higher model probability
        if game["ml_home"] is not None and game["ml_away"] is not None:
            raw_h = _american_to_prob(game["ml_home"])
            raw_a = _american_to_prob(game["ml_away"])
            ih, ia = _devig(raw_h, raw_a)
            h_edge, a_edge = ml_edge(model_prob_home, game["ml_home"], game["ml_away"])
            if model_prob_home >= 0.5:
                bet_side, bet_odds, model_p, implied_p, edge_val = \
                    home_name, game["ml_home"], model_prob_home, ih, (h_edge or 0)
            else:
                bet_side, bet_odds, model_p, implied_p, edge_val = \
                    away_name, game["ml_away"], 1 - model_prob_home, ia, (a_edge or 0)
            dec = _american_to_decimal(bet_odds)
            k = kelly_fraction(model_p, dec)
            stars = star_rating(abs(edge_val), [DEFAULT_ML_EDGE, DEFAULT_ML_EDGE * 1.5, DEFAULT_ML_EDGE * 2.5])
            game_rec["picks"].append({
                "bet_type": "ml",
                "bet_side": bet_side,
                "bet_odds": bet_odds,
                "model_prob": round(model_p, 4),
                "implied_prob": round(implied_p, 4),
                "edge": round(edge_val, 4),
                "stars": stars,
                "kelly_units": round(k * 100, 2),
                "model_margin": round(model_margin, 1),
                "model_total": round(model_total, 1),
                "vegas_spread": game["spread_home"],
                "vegas_total": game["total_line"],
            })

        # Spread pick — take the side the model favors vs the line
        if game["spread_home"] is not None:
            sp_edge_val = spread_edge(model_margin, game["spread_home"])
            if sp_edge_val is not None:
                if sp_edge_val > 0:
                    bet_team, bet_spread = home_name, game["spread_home"]
                else:
                    bet_team, bet_spread = away_name, -game["spread_home"]
                cp = cover_prob(abs(sp_edge_val))
                dec_sp = _american_to_decimal(game["spread_line"]) if game["spread_line"] else 1.909
                k_sp = kelly_fraction(cp, dec_sp)
                stars = star_rating(abs(sp_edge_val), [DEFAULT_SPREAD_EDGE, DEFAULT_SPREAD_EDGE * 1.5, DEFAULT_SPREAD_EDGE * 2.0])
                game_rec["picks"].append({
                    "bet_type": "spread",
                    "bet_team": bet_team,
                    "bet_spread": round(bet_spread, 1),
                    "bet_odds": game["spread_line"],
                    "edge": round(sp_edge_val, 2),
                    "stars": stars,
                    "kelly_units": round(k_sp * 100, 2),
                    "cover_margin": round(abs(sp_edge_val), 1),
                    "model_margin": round(model_margin, 1),
                    "vegas_spread": game["spread_home"],
                    "model_total": round(model_total, 1),
                    "vegas_total": game["total_line"],
                })

        # Total pick — over if model_total > vegas, under if model_total < vegas
        if game["total_line"] is not None:
            tot_e = total_edge(model_total, game["total_line"])
            if tot_e is not None:
                side = "OVER" if tot_e > 0 else "UNDER"
                cp_tot = cover_prob(abs(tot_e))
                k_tot = kelly_fraction(cp_tot, 1.909)
                stars = star_rating(abs(tot_e), [DEFAULT_TOTAL_EDGE, DEFAULT_TOTAL_EDGE * 1.4, DEFAULT_TOTAL_EDGE * 2.0])
                game_rec["picks"].append({
                    "bet_type": "total",
                    "bet_side": side,
                    "bet_odds": -110,
                    "edge": round(tot_e, 2),
                    "stars": stars,
                    "kelly_units": round(k_tot * 100, 2),
                    "model_total": round(model_total, 1),
                    "vegas_total": game["total_line"],
                    "model_margin": round(model_margin, 1),
                    "vegas_spread": game["spread_home"],
                })

        out.append(game_rec)

    out.sort(key=lambda g: g.get("commence_time", ""))
    return out


def star_rating(edge_abs, thresholds):
    """Return ★ rating string based on edge magnitude."""
    if edge_abs >= thresholds[2]:
        return "★★★"
    if edge_abs >= thresholds[1]:
        return "★★"
    if edge_abs >= thresholds[0]:
        return "★"
    return ""


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def format_time(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        local = dt.astimezone()
        return local.strftime("%-I:%M%p").lower()
    except Exception:
        return iso_str


def print_report(bets, thresholds):
    ml_thresh, sp_thresh, tot_thresh = thresholds

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'=' * 65}")
    print(f"  BRACKET BRAIN — BEST BETS   {today}")
    print(f"{'=' * 65}")

    ml_bets   = [(g, e) for g, e in bets if g["bet_type"] == "ml"     and abs(e) >= ml_thresh]
    sp_bets   = [(g, e) for g, e in bets if g["bet_type"] == "spread" and abs(e) >= sp_thresh]
    tot_bets  = [(g, e) for g, e in bets if g["bet_type"] == "total"  and abs(e) >= tot_thresh]

    ml_bets.sort(key=lambda x: -abs(x[1]))
    sp_bets.sort(key=lambda x: -abs(x[1]))
    tot_bets.sort(key=lambda x: -abs(x[1]))

    def header(title):
        print(f"\n  {title}")
        print(f"  {'-' * 60}")

    # --- Moneyline ---
    header("MONEYLINE")
    if not ml_bets:
        print("  No qualifying ML edges today.")
    for g, edge in ml_bets:
        side = g["bet_side"]
        odds = payout_str(g["bet_odds"])
        impl = g["implied_prob"]
        model_p = g["model_prob"]
        kelly_u = g.get("kelly_units", 0)
        stars = star_rating(abs(edge), [ml_thresh, ml_thresh * 1.5, ml_thresh * 2.5])
        print(f"  {stars:4s} {side} ({odds})")
        print(f"       {g['away_team']} @ {g['home_team']}  {format_time(g['commence_time'])}")
        print(f"       Model: {model_p:.1%}  Vegas implied: {impl:.1%}  Edge: {edge:+.1%}  Kelly: {kelly_u:.1f} units")

    # --- Spread ---
    header("SPREAD")
    if not sp_bets:
        print("  No qualifying spread edges today.")
    for g, edge in sp_bets:
        team = g["bet_team"]
        spread = g["bet_spread"]
        model_m = g["model_margin"]
        odds = payout_str(g["bet_odds"])
        cover = g["cover_margin"]
        kelly_u = g.get("kelly_units", 0)
        stars = star_rating(edge, [sp_thresh, sp_thresh * 1.5, sp_thresh * 2.0])
        print(f"  {stars:4s} {team} {spread:+.1f} ({odds})")
        print(f"       {g['away_team']} @ {g['home_team']}  {format_time(g['commence_time'])}")
        print(f"       Model margin: {model_m:+.1f}  Spread: {spread:+.1f}  Covers by: {cover:.1f} pts  Kelly: {kelly_u:.1f} units")

    # --- Totals ---
    header("OVER / UNDER")
    if not tot_bets:
        print("  No qualifying total edges today.")
    for g, edge in tot_bets:
        side = g["bet_side"]
        line = g["vegas_total"]
        model_t = g["model_total"]
        kelly_u = g.get("kelly_units", 0)
        stars = star_rating(abs(edge), [tot_thresh, tot_thresh * 1.4, tot_thresh * 2.0])
        print(f"  {stars:4s} {side} {line:.1f}")
        print(f"       {g['away_team']} @ {g['home_team']}  {format_time(g['commence_time'])}")
        print(f"       Model total: {model_t:.1f}  Vegas: {line:.1f}  Edge: {edge:+.1f} pts  Kelly: {kelly_u:.1f} units")

    total_flagged = len(ml_bets) + len(sp_bets) + len(tot_bets)
    print(f"\n  {total_flagged} qualifying bet(s) found.")
    print(f"{'=' * 65}\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Find best bets vs. Vegas for today's NCAAB games")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: ODDS_API_KEY or BETSTACK_API_KEY per ODDS_PROVIDER)")
    parser.add_argument("--year", type=int, default=datetime.now().year,
                        help="Team stats year (default: current year)")
    parser.add_argument("--ml-edge", type=float, default=DEFAULT_ML_EDGE,
                        help=f"Min ML edge to flag (default: {DEFAULT_ML_EDGE:.0%})")
    parser.add_argument("--spread-edge", type=float, default=DEFAULT_SPREAD_EDGE,
                        help=f"Min spread edge in pts (default: {DEFAULT_SPREAD_EDGE})")
    parser.add_argument("--total-edge", type=float, default=DEFAULT_TOTAL_EDGE,
                        help=f"Min total edge in pts (default: {DEFAULT_TOTAL_EDGE})")
    parser.add_argument("--all", action="store_true",
                        help="Print all games, not just qualifying bets")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_MODEL_CONFIDENCE,
                        help=f"Min model probability for ML bets (default: {DEFAULT_MIN_MODEL_CONFIDENCE:.0%})")
    args = parser.parse_args()

    api_key = args.api_key or get_api_key()
    if not api_key:
        provider = (os.environ.get("ODDS_PROVIDER") or "odds_api").strip().lower()
        print("ERROR: No API key provided.")
        if provider == "betstack":
            print("  Get a free key at https://betstack.dev")
            print("  Then: export BETSTACK_API_KEY=your_key  OR  --api-key YOUR_KEY")
        else:
            print("  Get a free key at https://the-odds-api.com (500 req/month)")
            print("  Then: export ODDS_API_KEY=your_key  OR  --api-key YOUR_KEY")
        sys.exit(1)

    print(f"Fetching today's NCAAB odds...")
    raw_games = fetch_today_odds(api_key)
    if not raw_games:
        print("No games found today.")
        return
    print(f"  {len(raw_games)} games found")

    print(f"Loading team stats (year={args.year})...")
    teams = load_team_stats(args.year)
    print(f"  {len(teams)} teams loaded")

    config = ModelConfig()

    bets = []       # list of (game_info_dict, edge_value)
    unmatched = []

    now_utc = datetime.now(timezone.utc)
    started = 0

    print(f"\nRunning model on {len(raw_games)} games...")
    for raw in raw_games:
        game = parse_game(raw)

        # Skip games that have already started
        try:
            commence = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
            if commence <= now_utc:
                started += 1
                continue
        except Exception:
            pass

        home_name = game["home_team"]
        away_name = game["away_team"]

        home_stats = lookup_team(home_name, teams)
        away_stats = lookup_team(away_name, teams)

        if home_stats is None or away_stats is None:
            missing = []
            if home_stats is None: missing.append(home_name)
            if away_stats is None: missing.append(away_name)
            unmatched.append(f"{away_name} @ {home_name} (no data: {', '.join(missing)})")
            continue

        home_stats["team"] = home_name
        away_stats["team"] = away_name

        result = run_model(home_stats, away_stats, config)
        model_prob_home = result["win_prob_a"]   # home = team_a
        model_margin    = result["predicted_margin"]
        model_total     = result["predicted_score_a"] + result["predicted_score_b"]

        base = {
            "home_team": home_name,
            "away_team": away_name,
            "commence_time": game["commence_time"],
            "model_prob_home": model_prob_home,
            "model_margin": model_margin,
            "model_total": model_total,
        }

        if args.all:
            print(f"  {away_name} @ {home_name}")
            print(f"    Model: {home_name} {model_prob_home:.1%}  margin {model_margin:+.1f}  total {model_total:.1f}")
            if game["ml_home"]:
                raw_h = _american_to_prob(game["ml_home"])
                raw_a = _american_to_prob(game["ml_away"])
                ih, ia = _devig(raw_h, raw_a)
                print(f"    Vegas ML: {home_name} {payout_str(game['ml_home'])} (impl {ih:.1%})  "
                      f"{away_name} {payout_str(game['ml_away'])} (impl {ia:.1%})")
            if game["spread_home"] is not None:
                print(f"    Vegas spread: {home_name} {game['spread_home']:+.1f}  model: {model_margin:+.1f}")
            if game["total_line"]:
                print(f"    Vegas total: {game['total_line']:.1f}  model: {model_total:.1f}")

        min_conf = getattr(args, "min_confidence", DEFAULT_MIN_MODEL_CONFIDENCE)

        # --- ML bets ---
        h_edge, a_edge = ml_edge(model_prob_home, game["ml_home"], game["ml_away"])
        if h_edge is not None:
            raw_h = _american_to_prob(game["ml_home"])
            raw_a = _american_to_prob(game["ml_away"])
            ih, ia = _devig(raw_h, raw_a)
            if h_edge >= args.ml_edge and model_prob_home >= min_conf:
                dec_h = _american_to_decimal(game["ml_home"])
                k_h = kelly_fraction(model_prob_home, dec_h)
                bets.append(({**base, "bet_type": "ml", "bet_side": home_name,
                               "bet_odds": game["ml_home"], "implied_prob": ih,
                               "model_prob": model_prob_home,
                               "kelly_size": round(k_h, 4),
                               "kelly_units": round(k_h * 100, 2)}, h_edge))
            if a_edge >= args.ml_edge and (1 - model_prob_home) >= min_conf:
                dec_a = _american_to_decimal(game["ml_away"])
                k_a = kelly_fraction(1 - model_prob_home, dec_a)
                bets.append(({**base, "bet_type": "ml", "bet_side": away_name,
                               "bet_odds": game["ml_away"], "implied_prob": ia,
                               "model_prob": 1 - model_prob_home,
                               "kelly_size": round(k_a, 4),
                               "kelly_units": round(k_a * 100, 2)}, a_edge))

        # --- Spread bets ---
        sp_edge = spread_edge(model_margin, game["spread_home"])
        if sp_edge is not None and abs(sp_edge) >= args.spread_edge:
            if sp_edge > 0:
                # Home team covers their spread
                bet_team = home_name
                bet_spread = game["spread_home"]          # e.g. -7.5 (home favored)
                cp_sp = cover_prob(sp_edge)
            else:
                # Away team covers their spread
                bet_team = away_name
                bet_spread = -game["spread_home"]         # flip: e.g. +30.5 if home is -30.5
                cp_sp = cover_prob(abs(sp_edge))
            dec_sp = _american_to_decimal(game["spread_line"]) if game["spread_line"] else 1.909
            k_sp = kelly_fraction(cp_sp, dec_sp)
            bets.append(({**base, "bet_type": "spread",
                           "bet_team": bet_team, "bet_spread": bet_spread,
                           "bet_odds": game["spread_line"],
                           "vegas_spread": game["spread_home"],
                           "cover_margin": abs(sp_edge),  # pts by which bet team beats spread
                           "model_margin": model_margin,
                           "kelly_size": round(k_sp, 4),
                           "kelly_units": round(k_sp * 100, 2)}, abs(sp_edge)))

        # --- Total bets ---
        tot_e = total_edge(model_total, game["total_line"])
        if tot_e is not None and abs(tot_e) >= args.total_edge:
            side = "OVER" if tot_e > 0 else "UNDER"
            cp_tot = cover_prob(abs(tot_e))
            k_tot = kelly_fraction(cp_tot, 1.909)
            bets.append(({**base, "bet_type": "total", "bet_side": side,
                           "vegas_total": game["total_line"],
                           "model_total": model_total,
                           "kelly_size": round(k_tot, 4),
                           "kelly_units": round(k_tot * 100, 2)}, tot_e))

    if started:
        print(f"  Skipped {started} game(s) already in progress.")
    if unmatched:
        print(f"\n  Skipped {len(unmatched)} games (no team stats):")
        for u in unmatched:
            print(f"    - {u}")

    thresholds = (args.ml_edge, args.spread_edge, args.total_edge)
    print_report(bets, thresholds)


if __name__ == "__main__":
    main()
