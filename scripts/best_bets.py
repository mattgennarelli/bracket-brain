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
from functools import lru_cache
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
ET_TZ = ZoneInfo("America/New_York")

try:
    import requests
except ImportError:
    # Don't sys.exit at import time — that crashes pytest collection for ALL tests.
    # Instead, defer the error to runtime so tests can safely import this module.
    requests = None  # type: ignore[assignment]

from engine import (
    predict_game, ModelConfig, DEFAULT_CONFIG, _normalize_team_for_match,
    _strip_mascot, enrich_team, load_teams_merged, is_ncaa_tournament_game,
    get_matchup_analysis_display, resolve_ff_pairs,
)
from odds_provider import get_provider, get_api_key

# Default edge thresholds to flag as a "bet"
DEFAULT_ML_EDGE = 0.07      # 7% win probability edge over implied odds
DEFAULT_SPREAD_EDGE = 4.0   # model cover margin must exceed 4 pts
DEFAULT_TOTAL_EDGE = 12.0   # require a larger total gap before issuing a totals bet
DEFAULT_MIN_MODEL_CONFIDENCE = 0.55  # skip ML bets when model probability < 55%
DEFAULT_MIN_ML_PROB_FOR_DOG = 0.40   # skip ML when betting underdog with model prob < 40%
DEFAULT_MIN_SPREAD_COVER_MARGIN = 3.0  # skip spread when cover margin < 3 pts
DEFAULT_MAX_3STAR_PICKS = 0   # 3-star picks disabled: 0/3 hit rate (0% ROI). Re-enable when positive.
DEFAULT_MAX_2STAR_PICKS = 8   # cap 2-star picks per day
DEFAULT_SIDE_PREFERENCE_RATIO = 0.80  # prefer a side when it's close to the best total on the same game

# Maximum moneyline price to bet — don't chase heavy chalk beyond this.
# At -250, break-even accuracy is 71.4%. Our model can occasionally justify that
# on tournament favorites; tighter than this was filtering out too many viable MLs.
DEFAULT_MAX_ML_PRICE = -250  # skip ML bets with juice worse than -250

# Betting-path total correction. Totals are materially better when we shrink our
# raw projection back toward market instead of trusting the full raw edge 1:1.
# Keep a modest intercept to handle the remaining global high-total bias.
TOTAL_MARKET_SHRINK = 0.60
TOTAL_MARKET_INTERCEPT = -2.5
TOTAL_FALLBACK_CORRECTION = -5.5
ROUND_NAME_BY_SIZE = {
    68: "First Four",
    64: "Round of 64",
    32: "Round of 32",
    16: "Sweet 16",
    8: "Elite 8",
    4: "Final Four",
    2: "Championship",
}
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

# Historical rescue lines for snapshots that were saved before team matching was fixed.
# These are only used when the daily card snapshot has no market rows for a matchup.
_MANUAL_HISTORICAL_LINES = {
    (
        _normalize_team_for_match("Vanderbilt Commodores"),
        _normalize_team_for_match("McNeese Cowboys"),
        "2026-03-19T19:15:00Z",
    ): {
        "ml_home": -800,
        "ml_away": 525,
        "spread_home": -11.5,
        "spread_line": -110,
        "total_line": 149.5,
    },
}


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


def _prob_to_american(prob):
    """Convert implied probability to approximate American odds."""
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return None
    if p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


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
    "mcneese":            "mcneese st.",
    "mcneese cowboys":    "mcneese st.",
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


def _contract_state_suffix(name):
    """Convert trailing 'State' to 'St.' for our canonical team keys."""
    words = str(name or "").split()
    if words and words[-1].lower() == "state":
        words[-1] = "St."
        return " ".join(words)
    return ""


def _candidate_lookup_keys(name):
    """Yield normalized lookup keys for a raw odds name and stripped variants."""
    seen = set()

    def add(candidate):
        if not candidate:
            return
        for variant in (candidate, _contract_state_suffix(candidate)):
            key = _normalize_team_for_match(variant)
            if key and key not in seen:
                seen.add(key)
                yield key

    yield from add(name)
    words = str(name or "").split()
    for n in range(1, min(4, len(words))):
        shorter = " ".join(words[:-n])
        if not shorter:
            break
        yield from add(shorter)


def _matchup_key(team_a, team_b):
    a = _normalize_team_for_match(_strip_mascot(team_a))
    b = _normalize_team_for_match(_strip_mascot(team_b))
    if not a or not b or a == b:
        return None
    return tuple(sorted((a, b)))


def _manual_historical_lines(home_name, away_name, commence_time):
    return _MANUAL_HISTORICAL_LINES.get((
        _normalize_team_for_match(home_name),
        _normalize_team_for_match(away_name),
        commence_time or "",
    ))


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


@lru_cache(maxsize=8)
def _load_bracket_context(year):
    path = os.path.join(DATA_DIR, f"bracket_{year}.json")
    if not os.path.isfile(path):
        return {}, {}
    with open(path) as f:
        bracket = json.load(f)
    seed_map = {}
    region_map = {}
    regions = bracket.get("regions", {})
    if isinstance(regions, dict):
        for region, teams in regions.items():
            for team in teams or []:
                name = team.get("team") or team.get("name")
                seed = team.get("seed")
                if not name:
                    continue
                for variant in (name, _strip_mascot(name)):
                    norm = _normalize_team_for_match(variant)
                    if not norm:
                        continue
                    seed_map[norm] = seed
                    region_map[norm] = region
    return seed_map, region_map


@lru_cache(maxsize=8)
def _exact_tournament_matchups(year):
    path = os.path.join(DATA_DIR, f"bracket_{year}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        raw = json.load(f)

    regions = {}
    for region, teams in (raw.get("regions") or {}).items():
        seed_map = {}
        for entry in teams or []:
            try:
                seed = int(entry.get("seed"))
            except (TypeError, ValueError, AttributeError):
                continue
            if entry.get("team"):
                seed_map[seed] = entry.get("team")
        regions[region] = seed_map

    def matchup_key(team_a, team_b):
        a = _normalize_team_for_match(_strip_mascot(team_a))
        b = _normalize_team_for_match(_strip_mascot(team_b))
        if not a or not b or a == b:
            return None
        return tuple(sorted((a, b)))

    def add_matchups(target, teams_a, teams_b):
        for team_a in teams_a or []:
            for team_b in teams_b or []:
                key = matchup_key(team_a, team_b)
                if key:
                    target.add(key)

    first_four_slots = {}
    matchups = {}
    for ff in raw.get("first_four", []):
        key = matchup_key(ff.get("team_a"), ff.get("team_b"))
        if key:
            matchups.setdefault(68, set()).add(key)
        region = ff.get("region")
        seed = ff.get("seed")
        if region and seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                continue
            first_four_slots[(region, seed)] = [t for t in (ff.get("team_a"), ff.get("team_b")) if t]

    region_winners = {}
    for region, seed_map in regions.items():
        def slot_teams(seed):
            out = set(first_four_slots.get((region, seed), []))
            if seed_map.get(seed):
                out.add(seed_map[seed])
            return out

        round_slots = []
        for seed_a, seed_b in REGION_R64_SEED_PAIRS:
            teams_a = slot_teams(seed_a)
            teams_b = slot_teams(seed_b)
            add_matchups(matchups.setdefault(64, set()), teams_a, teams_b)
            round_slots.append(set(teams_a) | set(teams_b))

        next_slots = []
        for idx in range(0, len(round_slots), 2):
            left = round_slots[idx]
            right = round_slots[idx + 1]
            add_matchups(matchups.setdefault(32, set()), left, right)
            next_slots.append(set(left) | set(right))
        round_slots = next_slots

        next_slots = []
        for idx in range(0, len(round_slots), 2):
            left = round_slots[idx]
            right = round_slots[idx + 1]
            add_matchups(matchups.setdefault(16, set()), left, right)
            next_slots.append(set(left) | set(right))
        round_slots = next_slots

        if len(round_slots) == 2:
            add_matchups(matchups.setdefault(8, set()), round_slots[0], round_slots[1])
            region_winners[region] = set(round_slots[0]) | set(round_slots[1])

    quadrant_order = raw.get("quadrant_order") or list(regions.keys())
    semifinal_winners = []
    for region_a, region_b in resolve_ff_pairs(quadrant_order, raw.get("final_four_matchups")):
        winners_a = region_winners.get(region_a, set())
        winners_b = region_winners.get(region_b, set())
        add_matchups(matchups.setdefault(4, set()), winners_a, winners_b)
        semifinal_winners.append(set(winners_a) | set(winners_b))
    if len(semifinal_winners) == 2:
        add_matchups(matchups.setdefault(2, set()), semifinal_winners[0], semifinal_winners[1])

    return matchups


def _nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int):
    d = datetime(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    d += timedelta(days=7 * (occurrence - 1))
    return d.date()


def _infer_tournament_round(year: int, scheduled_at: str):
    if not scheduled_at:
        return None
    try:
        dt = datetime.fromisoformat(str(scheduled_at).replace("Z", "+00:00")).astimezone(ET_TZ).date()
    except ValueError:
        return None
    r64_start = _nth_weekday_of_month(year, 3, 3, 3)
    first_four_start = r64_start - timedelta(days=2)
    windows = {
        68: (first_four_start, first_four_start + timedelta(days=1)),
        64: (r64_start, r64_start + timedelta(days=1)),
        32: (r64_start + timedelta(days=2), r64_start + timedelta(days=3)),
        16: (r64_start + timedelta(days=7), r64_start + timedelta(days=8)),
        8: (r64_start + timedelta(days=9), r64_start + timedelta(days=10)),
        4: (_nth_weekday_of_month(year, 4, 5, 1), _nth_weekday_of_month(year, 4, 5, 1)),
        2: (_nth_weekday_of_month(year, 4, 0, 1), _nth_weekday_of_month(year, 4, 0, 1)),
    }
    for round_of, (start, end) in windows.items():
        if start <= dt <= end:
            return round_of
    return None


def _exact_tournament_round_for_matchup(home_name, away_name, year, scheduled_at=None):
    round_of = _infer_tournament_round(year, scheduled_at)
    if round_of is None:
        return None
    key = _matchup_key(home_name, away_name)
    if key is None:
        return None
    return round_of if key in _exact_tournament_matchups(year).get(round_of, set()) else None


def _tournament_context(home_name, away_name, year, scheduled_at=None):
    seed_map, region_map = _load_bracket_context(year)
    home_keys = list(_candidate_lookup_keys(_prep_odds_name(home_name))) + list(_candidate_lookup_keys(home_name))
    away_keys = list(_candidate_lookup_keys(_prep_odds_name(away_name))) + list(_candidate_lookup_keys(away_name))
    home_seed = next((seed_map[k] for k in home_keys if k in seed_map), None)
    away_seed = next((seed_map[k] for k in away_keys if k in seed_map), None)
    home_region = next((region_map[k] for k in home_keys if k in region_map), None)
    away_region = next((region_map[k] for k in away_keys if k in region_map), None)
    round_of = _exact_tournament_round_for_matchup(home_name, away_name, year, scheduled_at=scheduled_at)
    region = home_region if home_region and home_region == away_region else None
    if home_region and home_region == away_region:
        return {
            "seed_home": home_seed,
            "seed_away": away_seed,
            "region": home_region,
            "round_name": ROUND_NAME_BY_SIZE.get(round_of),
            "round_of": round_of,
        }
    return {
        "seed_home": home_seed,
        "seed_away": away_seed,
        "region": region,
        "round_name": ROUND_NAME_BY_SIZE.get(round_of),
        "round_of": round_of,
    }


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

    # Step 1: as-is + mascot stripping on the raw provider name
    raw_keys = list(_candidate_lookup_keys(name))
    for key in raw_keys:
        if _try(key):
            return dict(_try(key))

    # Step 2: with abbreviation expansion + stripped variants
    prepped = _prep_odds_name(name)
    prepped_keys = list(_candidate_lookup_keys(prepped))
    for key in prepped_keys:
        if key not in raw_keys and _try(key):
            return dict(_try(key))

    # Step 4: manual overrides — check all candidate keys
    for candidate in raw_keys + prepped_keys:
        manual_key = _ODDS_MANUAL.get(candidate)
        if manual_key:
            mk = _normalize_team_for_match(manual_key)
            if _try(mk):
                return dict(_try(mk))

    # Step 5: conservative substring match — both keys must be ≥80% of longer
    for candidate in prepped_keys or raw_keys:
        for k, v in teams_merged.items():
            shorter, longer = (candidate, k) if len(candidate) <= len(k) else (k, candidate)
            if len(shorter) >= 6 and shorter in longer and len(shorter) >= len(longer) * 0.80:
                return dict(v)

    return None


def run_model(home_stats, away_stats, config=DEFAULT_CONFIG, round_name=None, region=None, year=None):
    """Run the same matchup pipeline used by the bracket for tournament games."""
    home = enrich_team(dict(home_stats))
    away = enrich_team(dict(away_stats))
    home.setdefault("seed", 8)
    away.setdefault("seed", 8)
    if round_name and year:
        analysis = get_matchup_analysis_display(
            home,
            away,
            data_dir=DATA_DIR,
            year=year,
            region=region,
            round_name=round_name,
            config=config,
        )
        return {
            "win_prob_a": analysis["win_prob_a"],
            "predicted_margin": analysis["predicted_margin"],
            "predicted_score_a": analysis["predicted_score_a"],
            "predicted_score_b": analysis["predicted_score_b"],
        }
    return predict_game(home, away, config=config, round_name=round_name)


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


def corrected_total_projection(raw_model_total, vegas_total=None):
    """Shrink raw totals toward market before evaluating a total bet.

    The model already captures tempo and efficiency, so a direct tempo multiplier
    would double-count pace.  Empirically, shrinking toward the posted market and
    then applying a small downward intercept is a better correction.
    """
    if raw_model_total is None:
        return None
    if vegas_total is None:
        return raw_model_total + TOTAL_FALLBACK_CORRECTION
    return vegas_total + TOTAL_MARKET_SHRINK * (raw_model_total - vegas_total) + TOTAL_MARKET_INTERCEPT


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
    config = DEFAULT_CONFIG
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

        round_name = None
        if is_ncaa_tournament_game(home_name, away_name, year=year):
            ctx = _tournament_context(home_name, away_name, year, scheduled_at=game["commence_time"])
            if ctx["seed_home"] is not None:
                home_stats["seed"] = ctx["seed_home"]
            if ctx["seed_away"] is not None:
                away_stats["seed"] = ctx["seed_away"]
            round_name = ctx["round_name"]
        result = run_model(
            home_stats,
            away_stats,
            config,
            round_name=round_name,
            region=ctx["region"] if round_name else None,
            year=year if round_name else None,
        )
        model_prob_home = result["win_prob_a"]
        model_margin = result["predicted_margin"]
        model_total_raw = result["predicted_score_a"] + result["predicted_score_b"]
        model_total = corrected_total_projection(model_total_raw, game["total_line"])

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

    return _curate_best_bets(rated, min_ml, min_spread, min_total, cap_3, cap_2)


def extract_best_bets_from_games(games, ml_min=None, spread_min=None, total_min=None,
                                 min_ml_prob_for_dog=None, min_spread_cover=None,
                                 max_3star=None, max_2star=None):
    """Filter a refreshed full-card game list down to the same best-bet format as get_best_bets_json."""
    min_ml = ml_min if ml_min is not None else DEFAULT_ML_EDGE
    min_spread = spread_min if spread_min is not None else DEFAULT_SPREAD_EDGE
    min_total = total_min if total_min is not None else DEFAULT_TOTAL_EDGE
    min_ml_dog = min_ml_prob_for_dog if min_ml_prob_for_dog is not None else DEFAULT_MIN_ML_PROB_FOR_DOG
    min_cover = min_spread_cover if min_spread_cover is not None else DEFAULT_MIN_SPREAD_COVER_MARGIN
    cap_3 = max_3star if max_3star is not None else DEFAULT_MAX_3STAR_PICKS
    cap_2 = max_2star if max_2star is not None else DEFAULT_MAX_2STAR_PICKS
    min_conf = DEFAULT_MIN_MODEL_CONFIDENCE

    bets = []
    for game in games or []:
        if not game.get("data_available"):
            continue
        base = {
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "commence_time": game.get("commence_time"),
            "model_prob_home": game.get("model_prob_home"),
            "model_margin": game.get("model_margin"),
            "model_total": game.get("model_total"),
        }
        for pick in game.get("picks", []):
            bt = pick.get("bet_type")
            if bt == "ml":
                edge = float(pick.get("edge") or 0)
                model_prob = float(pick.get("model_prob") or 0)
                implied_prob = float(pick.get("implied_prob") or 0)
                odds = pick.get("bet_odds")
                price_ok = odds is None or odds >= DEFAULT_MAX_ML_PRICE
                if implied_prob < 0.5 and model_prob < min_ml_dog:
                    continue
                if edge < min_ml or model_prob < min_conf or not price_ok:
                    continue
                rec = {**base, **pick}
                rec["edge"] = edge
                rec["stars"] = star_rating(edge, [min_ml, min_ml * 1.5, min_ml * 2.5])
                bets.append((rec, edge))
            elif bt == "spread":
                edge = abs(float(pick.get("edge") or 0))
                if edge < min_spread or edge < min_cover:
                    continue
                rec = {**base, **pick}
                rec["edge"] = float(pick.get("edge") or 0)
                rec["stars"] = star_rating(edge, [min_spread, min_spread * 1.5, min_spread * 2.0])
                bets.append((rec, edge))
            elif bt == "total":
                edge = abs(float(pick.get("edge") or 0))
                if edge < min_total:
                    continue
                rec = {**base, **pick}
                rec["edge"] = float(pick.get("edge") or 0)
                rec["stars"] = star_rating(edge, [min_total, min_total * 1.4, min_total * 2.0])
                bets.append((rec, edge))

    bets.sort(key=lambda x: (x[0].get("commence_time", ""), -abs(x[1])))
    rated = [rec for rec, _ in bets]
    return _curate_best_bets(rated, min_ml, min_spread, min_total, cap_3, cap_2)


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
    config = DEFAULT_CONFIG
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

        round_name = None
        if is_ncaa_tournament_game(home_name, away_name, year=year):
            ctx = _tournament_context(home_name, away_name, year, scheduled_at=game["commence_time"])
            if ctx["seed_home"] is not None:
                home_stats["seed"] = ctx["seed_home"]
            if ctx["seed_away"] is not None:
                away_stats["seed"] = ctx["seed_away"]
            round_name = ctx["round_name"]
        result = run_model(
            home_stats,
            away_stats,
            config,
            round_name=round_name,
            region=ctx["region"] if round_name else None,
            year=year if round_name else None,
        )
        model_prob_home = result["win_prob_a"]
        model_margin = result["predicted_margin"]
        model_total_raw = result["predicted_score_a"] + result["predicted_score_b"]
        model_total = corrected_total_projection(model_total_raw, game["total_line"])

        game_rec["model_prob_home"] = round(model_prob_home, 4)
        game_rec["model_margin"] = round(model_margin, 1)
        game_rec["model_total"] = round(model_total, 1)

        # ML pick — always surface the stronger lean, even if edge is negative
        if game["ml_home"] is not None and game["ml_away"] is not None:
            raw_h = _american_to_prob(game["ml_home"])
            raw_a = _american_to_prob(game["ml_away"])
            ih, ia = _devig(raw_h, raw_a)
            h_edge, a_edge = ml_edge(model_prob_home, game["ml_home"], game["ml_away"])
            ml_options = [
                (h_edge, home_name, game["ml_home"], model_prob_home, ih),
                (a_edge, away_name, game["ml_away"], 1 - model_prob_home, ia),
            ]
            edge_val, bet_side, bet_odds, model_p, implied_p = max(ml_options, key=lambda x: x[0])
            dec = _american_to_decimal(bet_odds)
            k = kelly_fraction(model_p, dec) if edge_val > 0 else 0.0
            stars = star_rating(edge_val, [DEFAULT_ML_EDGE, DEFAULT_ML_EDGE * 1.5, DEFAULT_ML_EDGE * 2.5])
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


def refresh_saved_card_games(games, year=None):
    """Re-score a saved card snapshot against the current model/team matcher.

    This lets the API serve corrected model leans even when we only have a saved
    daily card file and no live odds pull available.
    """
    year = year or datetime.now().year
    teams = load_team_stats(year)
    config = DEFAULT_CONFIG
    out = []

    for game in games or []:
        rec = {
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "commence_time": game.get("commence_time"),
            "data_available": False,
            "ncaa_tournament": game.get("ncaa_tournament", True),
            "round_of": game.get("round_of"),
            "round_name": game.get("round_name"),
            "picks": [],
        }

        home_name = rec["home_team"]
        away_name = rec["away_team"]
        old_picks = {p.get("bet_type"): dict(p) for p in game.get("picks", [])}
        line_override = _manual_historical_lines(home_name, away_name, rec["commence_time"])

        home_stats = lookup_team(home_name, teams)
        away_stats = lookup_team(away_name, teams)
        data_ok = (
            home_stats is not None and away_stats is not None
            and home_stats.get("adj_o") is not None and home_stats.get("adj_d") is not None
            and away_stats.get("adj_o") is not None and away_stats.get("adj_d") is not None
        )
        rec["data_available"] = data_ok
        if not data_ok:
            out.append(rec)
            continue

        round_name = None
        if is_ncaa_tournament_game(home_name, away_name, year=year):
            ctx = _tournament_context(home_name, away_name, year, scheduled_at=rec["commence_time"])
            if ctx["seed_home"] is not None:
                home_stats["seed"] = ctx["seed_home"]
            if ctx["seed_away"] is not None:
                away_stats["seed"] = ctx["seed_away"]
            round_name = ctx["round_name"]
        result = run_model(
            home_stats,
            away_stats,
            config,
            round_name=round_name,
            region=ctx["region"] if round_name else None,
            year=year if round_name else None,
        )
        model_prob_home = result["win_prob_a"]
        model_margin = result["predicted_margin"]
        model_total_raw = result["predicted_score_a"] + result["predicted_score_b"]
        market_total = None
        if old_picks.get("total") and old_picks["total"].get("vegas_total") is not None:
            market_total = old_picks["total"]["vegas_total"]
        elif line_override and line_override.get("total_line") is not None:
            market_total = line_override["total_line"]
        model_total = corrected_total_projection(model_total_raw, market_total)

        rec["model_prob_home"] = round(model_prob_home, 4)
        rec["model_margin"] = round(model_margin, 1)
        rec["model_total"] = round(model_total, 1)

        ml_old = old_picks.get("ml")
        if ml_old and ml_old.get("implied_prob") is not None:
            old_side = ml_old.get("bet_side")
            old_implied = float(ml_old.get("implied_prob") or 0)
            home_implied = old_implied if old_side == home_name else 1 - old_implied
            away_implied = 1 - home_implied
            home_edge = model_prob_home - home_implied
            away_edge = (1 - model_prob_home) - away_implied

            if home_edge >= away_edge:
                bet_side = home_name
                implied_prob = home_implied
                model_prob = model_prob_home
                edge_val = home_edge
            else:
                bet_side = away_name
                implied_prob = away_implied
                model_prob = 1 - model_prob_home
                edge_val = away_edge

            if old_side == bet_side and ml_old.get("bet_odds") is not None:
                bet_odds = ml_old.get("bet_odds")
            else:
                bet_odds = _prob_to_american(implied_prob)

            dec = _american_to_decimal(bet_odds) if bet_odds is not None else None
            kelly_units = 0.0
            if dec is not None and edge_val > 0:
                kelly_units = round(kelly_fraction(model_prob, dec) * 100, 2)

            rec["picks"].append({
                "bet_type": "ml",
                "bet_side": bet_side,
                "bet_odds": bet_odds,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge_val, 4),
                "stars": star_rating(edge_val, [DEFAULT_ML_EDGE, DEFAULT_ML_EDGE * 1.5, DEFAULT_ML_EDGE * 2.5]),
                "kelly_units": kelly_units,
                "model_margin": round(model_margin, 1),
                "model_total": round(model_total, 1),
                "vegas_spread": ml_old.get("vegas_spread"),
                "vegas_total": ml_old.get("vegas_total"),
            })
        elif line_override and line_override.get("ml_home") is not None and line_override.get("ml_away") is not None:
            raw_h = _american_to_prob(line_override["ml_home"])
            raw_a = _american_to_prob(line_override["ml_away"])
            ih, ia = _devig(raw_h, raw_a)
            h_edge, a_edge = ml_edge(model_prob_home, line_override["ml_home"], line_override["ml_away"])
            if h_edge >= a_edge:
                bet_side = home_name
                bet_odds = line_override["ml_home"]
                model_prob = model_prob_home
                implied_prob = ih
                edge_val = h_edge
            else:
                bet_side = away_name
                bet_odds = line_override["ml_away"]
                model_prob = 1 - model_prob_home
                implied_prob = ia
                edge_val = a_edge
            rec["picks"].append({
                "bet_type": "ml",
                "bet_side": bet_side,
                "bet_odds": bet_odds,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge_val, 4),
                "stars": star_rating(edge_val, [DEFAULT_ML_EDGE, DEFAULT_ML_EDGE * 1.5, DEFAULT_ML_EDGE * 2.5]),
                "kelly_units": round(kelly_fraction(model_prob, _american_to_decimal(bet_odds)) * 100, 2) if edge_val > 0 else 0.0,
                "model_margin": round(model_margin, 1),
                "model_total": round(model_total, 1),
                "vegas_spread": line_override.get("spread_home"),
                "vegas_total": line_override.get("total_line"),
            })

        sp_old = old_picks.get("spread")
        if sp_old and sp_old.get("vegas_spread") is not None:
            sp_edge_val = spread_edge(model_margin, sp_old["vegas_spread"])
            if sp_edge_val is not None:
                if sp_edge_val > 0:
                    bet_team, bet_spread = home_name, sp_old["vegas_spread"]
                else:
                    bet_team, bet_spread = away_name, -sp_old["vegas_spread"]
                cp = cover_prob(abs(sp_edge_val))
                dec_sp = _american_to_decimal(sp_old["bet_odds"]) if sp_old.get("bet_odds") else 1.909
                rec["picks"].append({
                    "bet_type": "spread",
                    "bet_team": bet_team,
                    "bet_spread": round(bet_spread, 1),
                    "bet_odds": sp_old.get("bet_odds"),
                    "edge": round(sp_edge_val, 2),
                    "stars": star_rating(abs(sp_edge_val), [DEFAULT_SPREAD_EDGE, DEFAULT_SPREAD_EDGE * 1.5, DEFAULT_SPREAD_EDGE * 2.0]),
                    "kelly_units": round(kelly_fraction(cp, dec_sp) * 100, 2),
                    "cover_margin": round(abs(sp_edge_val), 1),
                    "model_margin": round(model_margin, 1),
                    "vegas_spread": sp_old["vegas_spread"],
                    "model_total": round(model_total, 1),
                    "vegas_total": sp_old.get("vegas_total"),
                })
        elif line_override and line_override.get("spread_home") is not None:
            sp_edge_val = spread_edge(model_margin, line_override["spread_home"])
            if sp_edge_val is not None:
                if sp_edge_val > 0:
                    bet_team, bet_spread = home_name, line_override["spread_home"]
                else:
                    bet_team, bet_spread = away_name, -line_override["spread_home"]
                cp = cover_prob(abs(sp_edge_val))
                dec_sp = _american_to_decimal(line_override.get("spread_line", -110))
                rec["picks"].append({
                    "bet_type": "spread",
                    "bet_team": bet_team,
                    "bet_spread": round(bet_spread, 1),
                    "bet_odds": line_override.get("spread_line", -110),
                    "edge": round(sp_edge_val, 2),
                    "stars": star_rating(abs(sp_edge_val), [DEFAULT_SPREAD_EDGE, DEFAULT_SPREAD_EDGE * 1.5, DEFAULT_SPREAD_EDGE * 2.0]),
                    "kelly_units": round(kelly_fraction(cp, dec_sp) * 100, 2),
                    "cover_margin": round(abs(sp_edge_val), 1),
                    "model_margin": round(model_margin, 1),
                    "vegas_spread": line_override["spread_home"],
                    "model_total": round(model_total, 1),
                    "vegas_total": line_override.get("total_line"),
                })

        tot_old = old_picks.get("total")
        if tot_old and tot_old.get("vegas_total") is not None:
            tot_e = total_edge(model_total, tot_old["vegas_total"])
            if tot_e is not None:
                cp_tot = cover_prob(abs(tot_e))
                rec["picks"].append({
                    "bet_type": "total",
                    "bet_side": "OVER" if tot_e > 0 else "UNDER",
                    "bet_odds": tot_old.get("bet_odds", -110),
                    "edge": round(tot_e, 2),
                    "stars": star_rating(abs(tot_e), [DEFAULT_TOTAL_EDGE, DEFAULT_TOTAL_EDGE * 1.4, DEFAULT_TOTAL_EDGE * 2.0]),
                    "kelly_units": round(kelly_fraction(cp_tot, 1.909) * 100, 2),
                    "model_total": round(model_total, 1),
                    "vegas_total": tot_old["vegas_total"],
                    "model_margin": round(model_margin, 1),
                    "vegas_spread": tot_old.get("vegas_spread"),
                })
        elif line_override and line_override.get("total_line") is not None:
            tot_e = total_edge(model_total, line_override["total_line"])
            if tot_e is not None:
                cp_tot = cover_prob(abs(tot_e))
                rec["picks"].append({
                    "bet_type": "total",
                    "bet_side": "OVER" if tot_e > 0 else "UNDER",
                    "bet_odds": -110,
                    "edge": round(tot_e, 2),
                    "stars": star_rating(abs(tot_e), [DEFAULT_TOTAL_EDGE, DEFAULT_TOTAL_EDGE * 1.4, DEFAULT_TOTAL_EDGE * 2.0]),
                    "kelly_units": round(kelly_fraction(cp_tot, 1.909) * 100, 2),
                    "model_total": round(model_total, 1),
                    "vegas_total": line_override["total_line"],
                    "model_margin": round(model_margin, 1),
                    "vegas_spread": line_override.get("spread_home"),
                })

        out.append(rec)

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


def _pick_threshold(pick, min_ml, min_spread, min_total):
    bt = pick.get("bet_type")
    if bt == "ml":
        return min_ml
    if bt == "spread":
        return min_spread
    return min_total


def _normalized_pick_score(pick, min_ml, min_spread, min_total):
    threshold = _pick_threshold(pick, min_ml, min_spread, min_total)
    edge = abs(float(pick.get("edge") or 0))
    if threshold <= 0:
        return edge
    return edge / threshold


def _pick_game_key(pick):
    return (
        pick.get("commence_time", ""),
        pick.get("home_team", ""),
        pick.get("away_team", ""),
    )


def _curate_best_bets(records, min_ml, min_spread, min_total, cap_3, cap_2):
    grouped = {}
    for rec in records:
        grouped.setdefault(_pick_game_key(rec), []).append(rec)

    curated = []
    for group in grouped.values():
        ranked = sorted(
            group,
            key=lambda rec: (
                _normalized_pick_score(rec, min_ml, min_spread, min_total),
                abs(float(rec.get("edge") or 0)),
            ),
            reverse=True,
        )
        best_total = next((rec for rec in ranked if rec.get("bet_type") == "total"), None)
        best_side = next((rec for rec in ranked if rec.get("bet_type") in {"ml", "spread"}), None)
        chosen = ranked[0]
        if best_total and best_side:
            total_score = _normalized_pick_score(best_total, min_ml, min_spread, min_total)
            side_score = _normalized_pick_score(best_side, min_ml, min_spread, min_total)
            if side_score >= total_score * DEFAULT_SIDE_PREFERENCE_RATIO:
                chosen = best_side
        curated.append(chosen)

    curated.sort(
        key=lambda rec: (
            rec.get("commence_time", ""),
            -_normalized_pick_score(rec, min_ml, min_spread, min_total),
            -abs(float(rec.get("edge") or 0)),
        )
    )

    out = []
    count_3 = count_2 = 0
    for rec in curated:
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
