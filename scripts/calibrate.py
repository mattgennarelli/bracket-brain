"""
Calibrate the prediction model against historical tournament results.

Loads results_all.json + teams_merged_YYYY.json for each year, runs
predict_game for every historical matchup, and optimizes ModelConfig
parameters to minimize Brier score.

Uses walk-forward cross-validation: train on years N-k, test on year N.
Drops params with |calibrated - default| < 0.05 across folds; target ≤ 12 params.

Output: data/calibrated_config.json

Usage:
  python scripts/calibrate.py                # optimize + report (walk-forward, CV-based)
  python scripts/calibrate.py --report-only   # just score current params
  python scripts/calibrate.py --phase2-only   # optimize only Phase 2 + upset params (fast)
  python scripts/calibrate.py --priors-only   # optimize only coach + pedigree weights (very fast)
  python scripts/calibrate.py --holdout 2025  # hold out 2025 for final eval
  python scripts/calibrate.py --exclude-years 2008,2009  # exclude years for data validation
  python scripts/calibrate.py --min-floor 0.1  # M1.1: force possession/conf >= 0.1
  python scripts/calibrate.py --objective brier  # brier|logloss|brier+f4|brier-acc|brier-close|brier-upset|composite
  python scripts/calibrate.py --objective brier+f4 --brier-f4-weights 0.6,0.4  # Brier + bracket quality
  python scripts/calibrate.py --no-cv         # legacy: per-fold train optimization (may overfit)
  python scripts/calibrate.py --maxiter 100 --popsize 16  # longer optimization (default: 100, 16)
  python scripts/calibrate.py --multi-start 3  # run 3 times, pick best (default: 2)
  python scripts/calibrate.py --recency-weight  # weight recent years more
  python scripts/calibrate.py --round-weight    # weight late rounds more
  python scripts/calibrate.py --save-report docs/cal_report.json  # write JSON report
"""
import json
import math
import os
import sys
from dataclasses import replace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from engine import predict_game, ModelConfig, DEFAULT_CONFIG, _normalize_team_for_match, enrich_team
from engine import run_monte_carlo, load_bracket, _load_venues, _get_game_site, _load_school_locations


def load_results():
    """Load all per-year results files. Returns list of games with 'year' field."""
    all_games = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith("results_") and fname.endswith(".json") and fname != "results_all.json":
            year = int(fname.replace("results_", "").replace(".json", ""))
            path = os.path.join(DATA_DIR, fname)
            with open(path) as f:
                data = json.load(f)
            for g in data.get("games", []):
                g["year"] = year
                all_games.append(g)
    if not all_games:
        print("ERROR: No results files found. Run: python scripts/extract_results.py")
        sys.exit(1)
    return all_games


def load_all_teams():
    """Load teams_merged for each year. Returns {year: {normalized_name: stats_dict}}."""
    teams_by_year = {}
    for fname in os.listdir(DATA_DIR):
        if fname.startswith("teams_merged_") and fname.endswith(".json"):
            year = int(fname.replace("teams_merged_", "").replace(".json", ""))
            path = os.path.join(DATA_DIR, fname)
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                teams_by_year[year] = {}
                for r in data:
                    t = r.get("team", "")
                    if not t or t.startswith("Unknown"):
                        continue
                    key = _normalize_team_for_match(t)
                    if key:
                        teams_by_year[year][key] = dict(r)
    return teams_by_year


def _lookup_team(name, seed, year_teams):
    """Look up team stats, returning enriched dict or placeholder."""
    key = _normalize_team_for_match(name)
    if key in year_teams:
        t = dict(year_teams[key])
        t["seed"] = seed
        t["team"] = name
        return t
    for k, v in year_teams.items():
        if key in k or k in key:
            t = dict(v)
            t["seed"] = seed
            t["team"] = name
            return t
    return {"team": name, "seed": seed, "adj_o": 85, "adj_d": 112,
            "adj_tempo": 64, "barthag": 0.05}


def build_game_pairs(games, teams_by_year):
    """Build list of (team_a_dict, team_b_dict, actual_winner_is_a, margin, year, game_info) tuples.

    Also reconstructs tournament path data (path_avg_barthag, path_rounds) for each team
    by replaying prior rounds in chronological order within each year.
    """
    ROUND_ORDER = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]

    def round_idx(g):
        rn = g.get("round_name", "")
        try:
            return ROUND_ORDER.index(rn)
        except ValueError:
            return 99

    # Group and sort games by year and round order
    by_year = {}
    for g in games:
        yr = g.get("year")
        by_year.setdefault(yr, []).append(g)

    # Precompute eff_rank (1-indexed barthag rank among tournament teams) per year
    eff_rank_by_year = {}
    for yr, yr_games in by_year.items():
        if yr not in teams_by_year:
            continue
        yt = teams_by_year[yr]
        team_barthags = []
        seen_keys = set()
        for g in yr_games:
            for name, seed in [(g["team_a"], g["seed_a"]), (g["team_b"], g["seed_b"])]:
                t = _lookup_team(name, seed, yt)
                if t.get("barthag") is not None:
                    key = _normalize_team_for_match(name)
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        team_barthags.append((key, t["barthag"]))
        team_barthags.sort(key=lambda x: -x[1])
        eff_rank_by_year[yr] = {key: i + 1 for i, (key, _) in enumerate(team_barthags)}

    # Replay rounds in order, capturing path state BEFORE each game is processed.
    # game_path_snapshot[(yr, game_key)] = {team_name: path_dict_snapshot}
    game_path_snapshot = {}
    for yr, yr_games in by_year.items():
        if yr not in teams_by_year:
            continue
        yt = teams_by_year[yr]
        state = {}  # team_name -> {"path_opponents_barthag": [...]}
        for g in sorted(yr_games, key=round_idx):
            # Snapshot BEFORE processing this game
            game_key = (g["team_a"], g["team_b"], g.get("round_name", ""))
            snap = {}
            for name in (g["team_a"], g["team_b"]):
                if name in state:
                    snap[name] = dict(state[name])
            game_path_snapshot[(yr, game_key)] = snap

            # Now update winner's path
            winner = g["winner"]
            loser = g["team_b"] if winner == g["team_a"] else g["team_a"]
            loser_seed = g["seed_b"] if winner == g["team_a"] else g["seed_a"]
            loser_team = _lookup_team(loser, loser_seed, yt)
            loser_barthag = loser_team.get("barthag", 0.5)
            if winner not in state:
                state[winner] = {"path_opponents_barthag": []}
            state[winner]["path_opponents_barthag"] = state[winner]["path_opponents_barthag"] + [loser_barthag]

    # Preload school locations once for efficiency
    school_locs = _load_school_locations()

    pairs = []
    skipped = 0
    for g in games:
        yr = g.get("year")
        if yr not in teams_by_year:
            skipped += 1
            continue
        yt = teams_by_year[yr]
        a = _lookup_team(g["team_a"], g["seed_a"], yt)
        b = _lookup_team(g["team_b"], g["seed_b"], yt)
        # Attach eff_rank (efficiency rank by barthag among tournament teams)
        rank_map = eff_rank_by_year.get(yr, {})
        for team_dict, name in ((a, g["team_a"]), (b, g["team_b"])):
            key = _normalize_team_for_match(name)
            if key and key in rank_map:
                team_dict["eff_rank"] = rank_map[key]
        # Attach school location for proximity calculation (normalize name first)
        for team_dict, name in ((a, g["team_a"]), (b, g["team_b"])):
            if "location" not in team_dict:
                loc = school_locs.get(name)
                if not loc:
                    nk = _normalize_team_for_match(name)
                    for sname, scoords in school_locs.items():
                        if _normalize_team_for_match(sname) == nk:
                            loc = scoords
                            break
                if loc:
                    team_dict["location"] = loc
        # Attach pre-game path snapshot
        game_key = (g["team_a"], g["team_b"], g.get("round_name", ""))
        snap = game_path_snapshot.get((yr, game_key), {})
        for team_dict, name in ((a, g["team_a"]), (b, g["team_b"])):
            ps = snap.get(name, {})
            prev = ps.get("path_opponents_barthag", [])
            if prev:
                team_dict["path_avg_barthag"] = sum(prev) / len(prev)
                team_dict["path_rounds"] = len(prev)
        a_won = 1 if g["winner"] == g["team_a"] else 0
        pairs.append((a, b, a_won, g["margin"], yr, g))
    if skipped:
        print(f"  Skipped {skipped} games (no Torvik data for that year)")
    return pairs


# Set by main() when --recency-weight or --round-weight passed
SCORING_OPTIONS = {"recency_weight": False, "round_weight": False}

ROUND_WEIGHTS = {
    "Round of 64": 1.0,
    "Round of 32": 1.2,
    "Sweet 16": 1.5,
    "Elite 8": 2.0,
    "Final Four": 2.5,
    "Championship": 3.0,
}

FULL_TOURNAMENT_GAMES = 63


def _partial_year_weight_cap(n_games):
    """Return the maximum effective weight for a partial tournament year."""
    if n_games < 16:
        return 16
    if n_games < 32:
        return 32
    if n_games < 48:
        return 48
    return FULL_TOURNAMENT_GAMES


def build_partial_year_weight_overrides(pairs):
    """Return {year: per-game-weight} for any partial tournament seasons."""
    counts = {}
    for _, _, _, _, yr, _ in pairs:
        counts[yr] = counts.get(yr, 0) + 1

    overrides = {}
    for yr, n_games in counts.items():
        if 0 < n_games < FULL_TOURNAMENT_GAMES:
            overrides[yr] = _partial_year_weight_cap(n_games) / n_games
    return overrides


def partial_years_from_pairs(pairs):
    """Return partial tournament years present in pairs."""
    return sorted(build_partial_year_weight_overrides(pairs))


def score_model(pairs, config, recency_weight=None, round_weight=None, year_weights=None):
    """Evaluate model on all game pairs. Returns metrics dict.

    Also returns brier_close (Brier on games with |pred_margin| < 3), brier_upset
    (Brier on games where lower seed won), n_close, n_upset for multi-objective optimization.

    If recency_weight: weight game by 1 + 0.05 * (year - 2010).
    If round_weight: weight by R64=1, R32=1.2, S16=1.5, E8=2, FF=2.5, Champ=3.
    """
    recency_weight = recency_weight if recency_weight is not None else SCORING_OPTIONS.get("recency_weight", False)
    round_weight = round_weight if round_weight is not None else SCORING_OPTIONS.get("round_weight", False)
    year_weights = year_weights if year_weights is not None else build_partial_year_weight_overrides(pairs)
    brier_sum = 0.0
    log_loss_sum = 0.0
    correct = 0
    margin_errors = []
    total_errors = []
    round_stats = {}
    brier_close_sum = 0.0
    brier_upset_sum = 0.0
    weight_close_sum = 0.0
    weight_upset_sum = 0.0
    n_close = 0
    n_upset = 0
    weight_sum = 0.0
    n = len(pairs)

    for a, b, a_won, actual_margin, yr, g in pairs:
        rname = g.get("round_name", "Unknown")
        w = 1.0
        w *= year_weights.get(yr, 1.0)
        if recency_weight:
            w *= 1.0 + 0.05 * (yr - 2010)
        if round_weight:
            w *= ROUND_WEIGHTS.get(rname, 1.0)
        weight_sum += w

        try:
            venues = _load_venues(yr)
            sa, sb = a.get("seed"), b.get("seed")
            game_site = _get_game_site(venues, g.get("region"), rname, seed_a=sa, seed_b=sb) if venues else None
            result = predict_game(enrich_team(a), enrich_team(b), game_site=game_site,
                                  config=config, round_name=rname)
        except Exception:
            continue
        prob_a = result["win_prob_a"]
        prob_a = max(0.001, min(0.999, prob_a))

        brier_contrib = (prob_a - a_won) ** 2
        brier_sum += w * brier_contrib
        log_loss_sum -= w * (a_won * math.log(prob_a) + (1 - a_won) * math.log(1 - prob_a))

        predicted_winner_is_a = prob_a >= 0.5
        if predicted_winner_is_a == bool(a_won):
            correct += 1

        pred_margin = result["predicted_margin"]
        actual_signed = actual_margin if a_won else -actual_margin
        margin_errors.append(abs(pred_margin - actual_signed))

        # Close-game and upset tracking
        if abs(pred_margin) < 3:
            brier_close_sum += w * brier_contrib
            weight_close_sum += w
            n_close += 1
        seed_a, seed_b = a.get("seed", 8), b.get("seed", 8)
        is_upset = g.get("upset", (a_won and seed_a > seed_b) or (not a_won and seed_b > seed_a))
        if is_upset:
            brier_upset_sum += w * brier_contrib
            weight_upset_sum += w
            n_upset += 1

        # Total prediction tracking
        score_a = g.get("score_a")
        score_b = g.get("score_b")
        if score_a is not None and score_b is not None:
            predicted_total = result["predicted_score_a"] + result["predicted_score_b"]
            total_errors.append(predicted_total - (score_a + score_b))

        if rname not in round_stats:
            round_stats[rname] = {"correct": 0, "total": 0, "brier": 0.0}
        round_stats[rname]["total"] += 1
        round_stats[rname]["brier"] += w * brier_contrib
        if predicted_winner_is_a == bool(a_won):
            round_stats[rname]["correct"] += 1

    denom = weight_sum if (recency_weight or round_weight) and weight_sum > 0 else n
    return {
        "brier_score": brier_sum / denom if denom else 1.0,
        "log_loss": log_loss_sum / denom if denom else 10.0,
        "accuracy": correct / n if n else 0.0,
        "spread_mae": sum(margin_errors) / len(margin_errors) if margin_errors else 99,
        "n_games": n,
        "correct": correct,
        "round_stats": round_stats,
        "total_bias": sum(total_errors) / len(total_errors) if total_errors else 0,
        "total_mae": sum(abs(e) for e in total_errors) / len(total_errors) if total_errors else 0,
        "brier_close": brier_close_sum / weight_close_sum if n_close and weight_close_sum > 0 else (brier_close_sum / n_close if n_close else 0.0),
        "brier_upset": brier_upset_sum / weight_upset_sum if n_upset and weight_upset_sum > 0 else (brier_upset_sum / n_upset if n_upset else 0.0),
        "n_close": n_close,
        "n_upset": n_upset,
    }


PRIOR_GATE_RECENT_YEARS = (2023, 2024, 2025)


def _resolve_prior_gate_recent_years(pairs, recent_years=None):
    """Use the standard recent slice plus any newer available seasons."""
    if recent_years is not None:
        return tuple(recent_years)
    years = sorted({p[4] for p in pairs})
    resolved = list(PRIOR_GATE_RECENT_YEARS)
    resolved.extend(y for y in years if y > PRIOR_GATE_RECENT_YEARS[-1])
    return tuple(dict.fromkeys(resolved))


def _filter_pairs_by_years(pairs, years):
    """Return only pairs whose season is in years."""
    allowed = set(years)
    return [p for p in pairs if p[4] in allowed]


def _build_prior_variants(config):
    """Build the four launch-gate variants for coach/pedigree."""
    return {
        "base": replace(config, num_sims=1),
        "coach_zero": replace(config, num_sims=1, coach_tourney_max_bonus=0.0),
        "pedigree_zero": replace(config, num_sims=1, pedigree_max_bonus=0.0),
        "both_zero": replace(config, num_sims=1, coach_tourney_max_bonus=0.0, pedigree_max_bonus=0.0),
    }


def evaluate_prior_gate(pairs, config, recent_years=None):
    """Evaluate whether coach/pedigree should survive launch gating.

    Keep a prior only if the base config beats the corresponding zeroed variant on
    both all-years and the recent-year slice.
    """
    recent_years = _resolve_prior_gate_recent_years(pairs, recent_years)
    recent_pairs = _filter_pairs_by_years(pairs, recent_years)
    variants = _build_prior_variants(config)
    report = {}
    for name, variant in variants.items():
        report[name] = {
            "all": score_model(pairs, variant),
            "recent": score_model(recent_pairs, variant) if recent_pairs else None,
        }

    base_all = report["base"]["all"]["brier_score"]
    base_recent = report["base"]["recent"]["brier_score"] if report["base"]["recent"] else None
    coach_all = report["coach_zero"]["all"]["brier_score"]
    coach_recent = report["coach_zero"]["recent"]["brier_score"] if report["coach_zero"]["recent"] else None
    pedigree_all = report["pedigree_zero"]["all"]["brier_score"]
    pedigree_recent = report["pedigree_zero"]["recent"]["brier_score"] if report["pedigree_zero"]["recent"] else None

    keep_coach = base_all < coach_all and (base_recent < coach_recent if base_recent is not None and coach_recent is not None else True)
    keep_pedigree = base_all < pedigree_all and (base_recent < pedigree_recent if base_recent is not None and pedigree_recent is not None else True)

    gated = replace(config, num_sims=1)
    if not keep_coach:
        gated.coach_tourney_max_bonus = 0.0
    if not keep_pedigree:
        gated.pedigree_max_bonus = 0.0
    return report, keep_coach, keep_pedigree, gated


def print_prior_gate_report(report, keep_coach, keep_pedigree, recent_years=None):
    """Print a concise coach/pedigree ablation report."""
    recent_years = tuple(recent_years or PRIOR_GATE_RECENT_YEARS)
    recent_label = f"{recent_years[0]}-{recent_years[-1]}"
    print(f"\n{'=' * 60}")
    print("  COACH / PEDIGREE ABLATION")
    print(f"{'=' * 60}")
    for name in ("base", "coach_zero", "pedigree_zero", "both_zero"):
        row = report.get(name, {})
        all_metrics = row.get("all") or {}
        recent_metrics = row.get("recent") or {}
        recent_text = "n/a"
        if recent_metrics:
            recent_text = f"Brier={recent_metrics['brier_score']:.4f}  Acc={recent_metrics['accuracy']:.1%}"
        print(
            f"  {name:13s}  "
            f"all: Brier={all_metrics.get('brier_score', 0):.4f}  Acc={all_metrics.get('accuracy', 0):.1%}  "
            f"{recent_label}: {recent_text}"
        )
    print(f"\n  Coach gate:    {'KEEP' if keep_coach else 'ZERO'}")
    print(f"  Pedigree gate: {'KEEP' if keep_pedigree else 'ZERO'}")


def apply_prior_gate(pairs, config, optimized=None, recent_years=None, enabled=True):
    """Run the coach/pedigree ablation gate and return gated config + params."""
    if not enabled:
        return None, None, None, config, dict(optimized or {})

    recent_years = _resolve_prior_gate_recent_years(pairs, recent_years)
    gate_report, keep_coach, keep_pedigree, gated_config = evaluate_prior_gate(
        pairs, config, recent_years=recent_years
    )
    gated_optimized = dict(optimized or {})
    gated_optimized["coach_tourney_max_bonus"] = round(float(gated_config.coach_tourney_max_bonus), 4)
    gated_optimized["pedigree_max_bonus"] = round(float(gated_config.pedigree_max_bonus), 4)
    return gate_report, keep_coach, keep_pedigree, gated_config, gated_optimized


PARAM_SPEC = [
    # seed_weight removed: efficiency model already captures seed info;
    # 0.0 ties or beats 0.18 on 2023-2025 folds (Brier diff < 0.002)
    ("seed_weight", 0.0, 8.0),
    ("base_scoring_stdev", 8.0, 18.0),
    ("sos_max_bonus", 0.0, 8.0),
    ("possession_edge_max_bonus", 0.0, 8.0),
    ("ft_clutch_max_bonus", 0.0, 6.0),
    ("experience_max_bonus", 0.0, 8.0),  # widened: optimizer hit 4.0 ceiling
    ("coach_tourney_max_bonus", 0.0, 6.0),
    ("pedigree_max_bonus", 0.0, 6.0),
    ("three_pt_volatility_factor", 0.0, 3.0),
    ("tempo_volatility_weight", 0.0, 8.0),  # widened: optimizer consistently hits 4.0 ceiling
    ("star_player_max_bonus", 0.0, 8.0),
    ("proximity_max_bonus", 0.0, 5.0),
    ("proximity_neutral_distance", 600, 1800),
    ("proximity_home_threshold_mi", 30, 200),
    ("proximity_home_bonus", 0.0, 4.0),
    ("momentum_max_bonus", 0.0, 5.0),
    ("win_pct_max_bonus", 0.0, 5.0),
    ("conf_rating_max_bonus", 0.0, 4.0),
    ("conf_tourney_max_bonus", 0.0, 2.0),
    ("depth_max_bonus", 0.0, 8.0),
    ("em_opp_adjust_max_bonus", 0.0, 8.0),
    ("em_adj_o_weight", 0.0, 1.0),
    ("em_runs_margin_max_bonus", 0.0, 6.0),
    ("big_bpr_max_bonus", 0.0, 4.0),
    ("guard_bpr_max_bonus", 0.0, 4.0),
    ("creator_count_max_bonus", 0.0, 4.0),
    ("ft_foul_rate_max_bonus", 0.0, 6.0),
    # Per-round score scaling (Phase 1)
    ("score_scale", 0.88, 1.00),
    ("score_scale_r64", 0.90, 1.00),
    ("score_scale_r32", 0.90, 1.00),
    ("score_scale_s16", 0.88, 0.98),
    ("score_scale_e8", 0.86, 0.96),
    ("score_scale_ff", 0.85, 0.95),
    # Per-round stdev inflation (Phase 2)
    ("round_stdev_inflation_r64", 1.0, 1.15),
    ("round_stdev_inflation_r32", 1.0, 1.15),
    ("round_stdev_inflation_s16", 1.0, 1.20),
    ("round_stdev_inflation_e8", 1.0, 1.25),
    ("round_stdev_inflation_ff", 1.0, 1.30),
    # Late-round dampening: pull win-probs toward 0.5 in Sweet 16+ (reduces overconfidence)
    ("late_round_dampening", 0.0, 0.35),
    # Close-game upset tolerance
    ("upset_spread_threshold", 2.0, 8.0),
    ("upset_tolerance_max_bonus", 0.0, 8.0),  # widened: optimizer hit 5.0 ceiling
    ("close_game_stdev_boost", 0.0, 0.3),
    # NOTE: injury_penalty_per_level is intentionally NOT calibrated here.
    # We have no historical injury data to train against (only 2026 injuries exist),
    # so calibration would zero it out. Instead it's treated as an expert-set prior
    # in ModelConfig (default=3.0) and applied using live injuries_YYYY.json at
    # prediction time. This is correct: it's a current-year signal, not a
    # historical pattern the optimizer can learn.
]

# Phase 2 only: score scaling, stdev inflation, late-round dampening, close-game upset tolerance
PARAM_SPEC_PHASE2 = [
    ("score_scale", 0.88, 1.00),
    ("score_scale_r64", 0.90, 1.00),
    ("score_scale_r32", 0.90, 1.00),
    ("score_scale_s16", 0.88, 0.98),
    ("score_scale_e8", 0.86, 0.96),
    ("score_scale_ff", 0.85, 0.95),
    ("round_stdev_inflation_r64", 1.0, 1.15),
    ("round_stdev_inflation_r32", 1.0, 1.15),
    ("round_stdev_inflation_s16", 1.0, 1.20),
    ("round_stdev_inflation_e8", 1.0, 1.25),
    ("round_stdev_inflation_ff", 1.0, 1.30),
    ("late_round_dampening", 0.0, 0.35),
    ("upset_spread_threshold", 2.0, 8.0),
    ("upset_tolerance_max_bonus", 0.0, 8.0),  # widened: optimizer hit 5.0 ceiling
    ("close_game_stdev_boost", 0.0, 0.3),
]

PARAM_SPEC_PRIORS = [
    ("coach_tourney_max_bonus", 0.0, 6.0),
    ("pedigree_max_bonus", 0.0, 6.0),
]


def _pairs_by_year(pairs):
    """Group pairs by year. Returns {year: [(a,b,a_won,margin,yr,g), ...]}."""
    by_year = {}
    for p in pairs:
        yr = p[4]
        by_year.setdefault(yr, []).append(p)
    return by_year


def _fold_pairs(pairs_by_year, test_year):
    """Split into train (all years before test_year) and test (test_year only)."""
    train = []
    test = []
    for yr, yr_pairs in sorted(pairs_by_year.items()):
        if yr < test_year:
            train.extend(yr_pairs)
        elif yr == test_year:
            test.extend(yr_pairs)
    return train, test


def _get_actual_champion_and_ff(games, year):
    """Extract actual champion and Final Four teams from results for a year."""
    year_games = [g for g in games if g.get("year") == year]
    champ = None
    ff_teams = set()
    for g in year_games:
        rn = g.get("round_name", "")
        if rn == "Championship":
            champ = g.get("winner")
        elif rn == "Final Four":
            ff_teams.add(g.get("team_a"))
            ff_teams.add(g.get("team_b"))
    return champ, ff_teams


def _normalize_team_for_bq(name):
    """Normalize for bracket-quality matching."""
    if not name:
        return ""
    n = name.strip().lower()
    aliases = {"uconn": "connecticut", "unc": "north carolina", "st. mary's": "saint mary's"}
    return aliases.get(n, n)


def _compute_bracket_quality_for_config(config, bracket_years, games, data_dir, num_sims=500):
    """Run Monte Carlo for config on bracket years, return penalty (lower is better).
    Penalty = (avg_champ_rank - 1)/15 + (1 - avg_champ_pct) + (1 - avg_ff_hit).
    """
    import io
    import contextlib

    champ_ranks = []
    champ_pcts = []
    ff_hits = []

    for year in bracket_years:
        path = os.path.join(data_dir, f"bracket_{year}.json")
        if not os.path.isfile(path):
            continue
        try:
            bracket, _, _ = load_bracket(path, data_dir=data_dir, year=year)
        except Exception:
            continue
        if not bracket:
            continue

        champ, actual_ff = _get_actual_champion_and_ff(games, year)
        if not champ:
            continue

        def _match(pred_name, actual_name):
            pa = _normalize_team_for_bq(pred_name)
            pb = _normalize_team_for_bq(actual_name)
            return pa == pb or pa in pb or pb in pa

        cfg = replace(config, num_sims=num_sims)
        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                mc = run_monte_carlo(bracket, config=cfg)
        champ_probs = mc["champion_probs"]
        ff_probs = mc["final_four_probs"]

        champ_rank = len(champ_probs) + 1
        champ_pct = 0.0
        for i, t in enumerate(champ_probs.keys()):
            if _match(t, champ):
                champ_rank = i + 1
                champ_pct = champ_probs.get(t, 0.0)
                break

        top8_ff = list(ff_probs.keys())[:8]
        hits = 0
        for actual in actual_ff:
            for pred in top8_ff:
                if _match(pred, actual):
                    hits += 1
                    break
        ff_hit = hits / 4.0 if actual_ff else 0.0

        champ_ranks.append(champ_rank)
        champ_pcts.append(champ_pct)
        ff_hits.append(ff_hit)

    if not champ_ranks:
        return 0.0

    avg_rank = sum(champ_ranks) / len(champ_ranks)
    avg_pct = sum(champ_pcts) / len(champ_pcts)
    avg_ff = sum(ff_hits) / len(ff_hits)
    penalty = (avg_rank - 1) / 15.0 + (1.0 - avg_pct) + (1.0 - avg_ff)
    return max(0.0, penalty)


def _compute_objective(metrics, objective_name="brier", baseline_accuracy=None, accuracy_penalty=0.05):
    """Compute optimization score from metrics. Lower is better.

    If baseline_accuracy is set and accuracy drops by more than 1%, add accuracy_penalty.
    """
    brier = metrics["brier_score"]
    log_loss = metrics.get("log_loss", brier * 3)  # fallback
    mae = metrics.get("total_mae", 0)
    acc = metrics.get("accuracy", 0)
    brier_close = metrics.get("brier_close", 0)
    brier_upset = metrics.get("brier_upset", 0)
    score = 0.0
    if objective_name == "brier":
        score = brier + 0.001 * mae
    elif objective_name == "logloss":
        score = log_loss + 0.001 * mae
    elif objective_name == "brier-acc":
        score = brier - 0.01 * acc
    elif objective_name == "brier-close":
        score = brier + 0.5 * brier_close
    elif objective_name == "brier-upset":
        score = brier + 0.3 * brier_upset
    elif objective_name == "composite":
        score = 0.7 * brier - 0.02 * acc + 0.2 * brier_close
    elif objective_name == "brier+f4":
        # Proxy for bracket quality: weight upset accuracy more heavily.
        # Full bracket quality (MC sims) is too expensive per evaluation;
        # the CV phase in calibrate_walk_forward handles the real brier+f4
        # weighting. This proxy ensures the final calibrate_single step
        # still biases toward getting key upset/close games right.
        score = brier + 0.3 * brier_upset + 0.001 * mae
    else:
        score = brier + 0.001 * mae
    if baseline_accuracy is not None and acc < baseline_accuracy - 0.01:
        score += accuracy_penalty
    return score


FLOOR_PARAMS = {"possession_edge_max_bonus", "conf_rating_max_bonus"}

# ---------------------------------------------------------------------------
# L2 regularization — penalise parameters that drift far from defaults
# ---------------------------------------------------------------------------

def _compute_l2_penalty(x, param_spec, reg_lambda):
    """L2 penalty: λ × Σ((x_i - default_i) / range_i)².

    Each parameter's deviation is normalised by its allowed range so that
    narrow-range params (score_scale: 0.12 span) and wide-range params
    (experience: 8.0 span) are penalised on the same scale.
    """
    if reg_lambda <= 0:
        return 0.0
    defaults = {n: getattr(ModelConfig(), n) for n, _, _ in param_spec}
    total = 0.0
    for i, (name, lo, hi) in enumerate(param_spec):
        span = hi - lo
        if span <= 0:
            continue
        default_val = defaults.get(name, (lo + hi) / 2)
        deviation = (x[i] - default_val) / span
        total += deviation * deviation
    return reg_lambda * total


def calibrate_single(pairs, param_spec, objective_name="brier", maxiter=60, popsize=12, seed=42,
                    init_values=None, min_floor=0.0, reg_lambda=0.0):
    """Optimize params on given pairs. Returns (optimized_dict, config).

    If init_values is provided (or calibrated_config.json exists), use as first DE population member.
    If min_floor > 0, apply as lower bound for possession_edge_max_bonus and conf_rating_max_bonus.
    If reg_lambda > 0, add L2 regularisation penalty to prevent parameter drift from defaults.
    """
    import random
    from scipy.optimize import differential_evolution

    eval_count = [0]
    best_score = [1e9]
    baseline_acc = [None]

    def objective(x):
        config = ModelConfig(num_sims=1)
        for i, (name, _, _) in enumerate(param_spec):
            setattr(config, name, x[i])
        metrics = score_model(pairs, config)
        if baseline_acc[0] is None:
            baseline_acc[0] = metrics["accuracy"]
        eval_count[0] += 1
        score = _compute_objective(metrics, objective_name, baseline_acc[0])
        score += _compute_l2_penalty(x, param_spec, reg_lambda)
        if score < best_score[0]:
            best_score[0] = score
            if eval_count[0] % 50 == 0:
                print(f"  [{eval_count[0]}] Best score: {best_score[0]:.5f}  Brier={metrics['brier_score']:.4f}")
        return score

    bounds = []
    for name, lo, hi in param_spec:
        if min_floor > 0 and name in FLOOR_PARAMS:
            lo = max(lo, min_floor)
        bounds.append((lo, hi))
    rng = random.Random(seed)

    # Warm start: use calibrated_config or init_values as first population member
    init_pop = None
    if init_values is None:
        cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
        if os.path.isfile(cal_path):
            with open(cal_path) as f:
                init_values = json.load(f)

    if init_values and popsize >= 5:
        first_row = []
        for i, (name, _, _) in enumerate(param_spec):
            lo, hi = bounds[i]
            v = init_values.get(name, getattr(ModelConfig(), name, (lo + hi) / 2))
            first_row.append(max(lo, min(hi, float(v))))
        init_pop = [first_row]
        for _ in range(popsize - 1):
            init_pop.append([rng.uniform(lo, hi) for lo, hi in bounds])
        init_pop = init_pop  # list of lists, scipy accepts it

    kwargs = dict(seed=seed, maxiter=maxiter, tol=1e-6, popsize=popsize,
                  mutation=(0.5, 1.5), recombination=0.8)
    if init_pop is not None:
        kwargs["init"] = init_pop

    result = differential_evolution(objective, bounds, **kwargs)

    optimized = {}
    config = ModelConfig(num_sims=1)
    for i, (name, _, _) in enumerate(param_spec):
        val = float(result.x[i])
        optimized[name] = round(val, 4)
        setattr(config, name, val)
    return optimized, config


def calibrate_phase2_only(pairs, objective_name="brier", maxiter=60, popsize=12, seed=42, reg_lambda=0.0):
    """Optimize only Phase 2 + upset params, starting from current calibrated_config."""
    from scipy.optimize import differential_evolution

    cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
    baseline = ModelConfig(num_sims=1)
    if os.path.isfile(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        for k, v in cal.items():
            if hasattr(baseline, k):
                setattr(baseline, k, v)

    eval_count = [0]
    best_score = [1e9]
    baseline_acc = [None]

    def objective(x):
        config = replace(baseline, num_sims=1)
        for i, (name, _, _) in enumerate(PARAM_SPEC_PHASE2):
            setattr(config, name, x[i])
        metrics = score_model(pairs, config)
        if baseline_acc[0] is None:
            baseline_acc[0] = metrics["accuracy"]
        eval_count[0] += 1
        score = _compute_objective(metrics, objective_name, baseline_acc[0])
        score += _compute_l2_penalty(x, PARAM_SPEC_PHASE2, reg_lambda)
        if score < best_score[0]:
            best_score[0] = score
            if eval_count[0] % 50 == 0:
                print(f"  [{eval_count[0]}] Best score: {best_score[0]:.5f}  Brier={metrics['brier_score']:.4f}")
        return score

    bounds = [(lo, hi) for _, lo, hi in PARAM_SPEC_PHASE2]
    result = differential_evolution(objective, bounds, seed=seed,
                                    maxiter=maxiter, tol=1e-6, popsize=popsize,
                                    mutation=(0.5, 1.5), recombination=0.8)

    # Merge phase2 optimized params into baseline (from calibrated_config or ModelConfig)
    optimized = {}
    if os.path.isfile(cal_path):
        with open(cal_path) as f:
            optimized = dict(json.load(f))
    else:
        all_param_names = list(dict.fromkeys([p[0] for p in PARAM_SPEC] + [p[0] for p in PARAM_SPEC_PHASE2]))
        for name in all_param_names:
            if hasattr(ModelConfig(), name):
                val = getattr(ModelConfig(), name)
                if isinstance(val, (int, float)):
                    optimized[name] = round(float(val), 4)
    for i, (name, _, _) in enumerate(PARAM_SPEC_PHASE2):
        optimized[name] = round(float(result.x[i]), 4)

    config = replace(baseline, num_sims=1)
    for k, v in optimized.items():
        if hasattr(config, k):
            setattr(config, k, v)

    return optimized, config


def calibrate_priors_only(pairs, objective_name="brier", maxiter=60, popsize=12, seed=42, reg_lambda=0.0):
    """Optimize only coach + pedigree weights, starting from current calibrated_config."""
    from scipy.optimize import differential_evolution

    cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
    baseline = ModelConfig(num_sims=1)
    if os.path.isfile(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        for k, v in cal.items():
            if hasattr(baseline, k):
                setattr(baseline, k, v)

    eval_count = [0]
    best_score = [1e9]
    baseline_acc = [None]

    def objective(x):
        config = replace(baseline, num_sims=1)
        for i, (name, _, _) in enumerate(PARAM_SPEC_PRIORS):
            setattr(config, name, x[i])
        metrics = score_model(pairs, config)
        if baseline_acc[0] is None:
            baseline_acc[0] = metrics["accuracy"]
        eval_count[0] += 1
        score = _compute_objective(metrics, objective_name, baseline_acc[0])
        score += _compute_l2_penalty(x, PARAM_SPEC_PRIORS, reg_lambda)
        if score < best_score[0]:
            best_score[0] = score
            if eval_count[0] % 25 == 0:
                print(f"  [{eval_count[0]}] Best score: {best_score[0]:.5f}  Brier={metrics['brier_score']:.4f}")
        return score

    bounds = [(lo, hi) for _, lo, hi in PARAM_SPEC_PRIORS]
    result = differential_evolution(
        objective, bounds, seed=seed,
        maxiter=maxiter, tol=1e-6, popsize=popsize,
        mutation=(0.5, 1.5), recombination=0.8
    )

    optimized = {}
    if os.path.isfile(cal_path):
        with open(cal_path) as f:
            optimized = dict(json.load(f))
    else:
        for name, _, _ in PARAM_SPEC:
            if hasattr(ModelConfig(), name):
                val = getattr(ModelConfig(), name)
                if isinstance(val, (int, float)):
                    optimized[name] = round(float(val), 4)
    for i, (name, _, _) in enumerate(PARAM_SPEC_PRIORS):
        optimized[name] = round(float(result.x[i]), 4)

    config = replace(baseline, num_sims=1)
    for k, v in optimized.items():
        if hasattr(config, k):
            setattr(config, k, v)

    return optimized, config


def _compute_cv_score(pairs_by_year, param_spec, x, objective_name, test_years, baseline_acc_ref,
                      games=None, brier_f4_weights=(0.6, 0.4), reg_lambda=0.0):
    """Evaluate param vector x on walk-forward CV. Returns (score, cv_brier).
    Lower is better.
    For objective brier+f4: score = w1*cv_brier + w2*bracket_quality_penalty.
    """
    config = ModelConfig(num_sims=1)
    for i, (name, _, _) in enumerate(param_spec):
        if i < len(x):
            setattr(config, name, x[i])

    scores = []
    briers = []
    for test_year in test_years:
        _, test_pairs = _fold_pairs(pairs_by_year, test_year)
        if not test_pairs:
            continue
        metrics = score_model(test_pairs, config)
        scores.append(_compute_objective(metrics, objective_name, baseline_acc_ref))
        briers.append(metrics["brier_score"])

    cv_brier = sum(briers) / len(briers) if briers else 1e9
    base_score = sum(scores) / len(scores) if scores else 1e9

    # Add L2 regularization penalty (applied once, not per-fold)
    l2 = _compute_l2_penalty(x, param_spec, reg_lambda)

    if objective_name == "brier+f4" and games:
        w1, w2 = brier_f4_weights[0], brier_f4_weights[1]
        bracket_years = [y for y in test_years if os.path.isfile(os.path.join(DATA_DIR, f"bracket_{y}.json"))]
        if bracket_years:
            bq_penalty = _compute_bracket_quality_for_config(config, bracket_years, games, DATA_DIR, num_sims=500)
            combined = w1 * cv_brier + w2 * bq_penalty
            return combined + l2, cv_brier
    return base_score + l2, cv_brier


def calibrate_walk_forward(pairs, objective_name="brier", maxiter=100, popsize=16, seed=42,
                          use_cv_objective=True, early_stop_iter=0, min_floor=0.0,
                          games=None, brier_f4_weights=(0.6, 0.4), reg_lambda=0.0):
    """Walk-forward: optimize to minimize CV Brier (or train Brier if use_cv_objective=False).
    Reduce params; return final config.
    If min_floor > 0, apply as lower bound for possession_edge_max_bonus and conf_rating_max_bonus.
    """
    from scipy.optimize import differential_evolution

    pairs_by_year = _pairs_by_year(pairs)
    years = sorted(pairs_by_year.keys())
    if len(years) < 3:
        print("  Not enough years for walk-forward; falling back to single calibration.")
        return calibrate_single(pairs, PARAM_SPEC, objective_name, maxiter, popsize, seed, min_floor=min_floor, reg_lambda=reg_lambda)

    partial_years = set(partial_years_from_pairs(pairs))
    test_years = [y for y in sorted(pairs_by_year.keys()) if y >= 2017 and y not in partial_years]
    if partial_years:
        print(f"  Skipping partial years in CV folds: {sorted(partial_years)}")
    defaults = {name: getattr(ModelConfig(), name) for name, _, _ in PARAM_SPEC}
    bounds = []
    for name, lo, hi in PARAM_SPEC:
        if min_floor > 0 and name in FLOOR_PARAMS:
            lo = max(lo, min_floor)
        bounds.append((lo, hi))
    baseline_acc_ref = [None]

    if use_cv_objective:
        # Optimize directly on CV Brier (reduces overfitting)
        eval_count = [0]
        best_score = [1e9]
        best_brier = [1e9]

        def objective(x):
            score, cv_brier = _compute_cv_score(
                pairs_by_year, PARAM_SPEC, x, objective_name, test_years, baseline_acc_ref[0],
                games=games, brier_f4_weights=brier_f4_weights, reg_lambda=reg_lambda
            )
            eval_count[0] += 1
            if baseline_acc_ref[0] is None:
                baseline_acc_ref[0] = 0.74  # placeholder; not used for brier-only
            if score < best_score[0]:
                best_score[0] = score
                best_brier[0] = cv_brier
                if eval_count[0] % 25 == 0:
                    print(f"  [{eval_count[0]}] Best CV score: {best_score[0]:.5f}  CV Brier={cv_brier:.4f}")
            return score

        print(f"\n--- CV-based optimization: {len(test_years)} folds, objective={objective_name} ---")
        result = differential_evolution(
            objective, bounds, seed=seed, maxiter=maxiter, tol=1e-6, popsize=popsize,
            mutation=(0.5, 1.5), recombination=0.8
        )
        optimized = {name: round(float(result.x[i]), 4) for i, (name, _, _) in enumerate(PARAM_SPEC)}
        cv_brier = best_brier[0]
        print(f"\n  Cross-validated Brier: {cv_brier:.4f}  ← honest out-of-sample estimate")
    else:
        # Legacy: per-fold train optimization
        fold_results = []
        for test_year in test_years:
            train_pairs, test_pairs = _fold_pairs(pairs_by_year, test_year)
            if not train_pairs or not test_pairs:
                continue
            print(f"\n--- Fold: train on years < {test_year}, test on {test_year} ({len(test_pairs)} games) ---")
            opt, _ = calibrate_single(train_pairs, PARAM_SPEC, objective_name, maxiter, popsize, seed, reg_lambda=reg_lambda)
            config = ModelConfig(num_sims=1)
            for k, v in opt.items():
                setattr(config, k, v)
            test_metrics = score_model(test_pairs, config)
            fold_results.append((test_year, opt, test_metrics["brier_score"]))
            print(f"  Test Brier on {test_year}: {test_metrics['brier_score']:.4f}")

        fold_briers = [b for _, _, b in fold_results]
        cv_brier = sum(fold_briers) / len(fold_briers)
        print(f"\n  Cross-validated Brier (avg of {len(fold_briers)} folds): {cv_brier:.4f}")

        # Merge fold opts: use median or mean per param (simplified: use last fold's opt)
        optimized = dict(fold_results[-1][1]) if fold_results else {}

    # Param reduction: drop params with |optimized - default| < 0.02
    reduced_spec = [
        (n, lo, hi) for n, lo, hi in PARAM_SPEC
        if abs(optimized.get(n, defaults.get(n, 0)) - defaults.get(n, 0)) >= 0.02
    ]
    if len(reduced_spec) > 20:
        ranked = [(n, abs(optimized.get(n, 0) - defaults.get(n, 0))) for n, _, _ in PARAM_SPEC]
        ranked.sort(key=lambda x: -x[1])
        keep = {n for n, _ in ranked[:20]}
        reduced_spec = [(n, lo, hi) for n, lo, hi in PARAM_SPEC if n in keep]
    print(f"\n  Reduced to {len(reduced_spec)} params (target ≤20): {[p[0] for p in reduced_spec]}")

    # Final calibration on all historical years, plus any partial current-year data if present.
    if partial_years:
        all_train = list(pairs)
        partial_label = ", ".join(str(y) for y in sorted(partial_years))
        print(f"\n--- Final calibration on {len(all_train)} games (including partial years: {partial_label}) ---")
    else:
        all_train = [p for p in pairs if p[4] < max(years)]
        print(f"\n--- Final calibration on {len(all_train)} games (all years except {max(years)}) ---")
    init_vals = optimized if use_cv_objective else None
    optimized, config = calibrate_single(
        all_train, reduced_spec, objective_name, maxiter, popsize, seed, init_values=init_vals, min_floor=min_floor,
        reg_lambda=reg_lambda
    )

    # M1: Drop params that hit bounds
    bounds_map = {n: (lo, hi) for n, lo, hi in PARAM_SPEC}
    for name in list(optimized.keys()):
        lo, hi = bounds_map.get(name, (0, 1))
        val = optimized[name]
        if val <= lo + 0.01:
            optimized[name] = 0.0
            setattr(config, name, 0.0)
            print(f"  Param {name} hit lower bound ({val:.4f}) -> removed (set to 0)")
        elif val >= hi - 0.01:
            # Use the BOUND value (optimizer's best guess given constraints), not the
            # unrelated ModelConfig default. The old behavior (falling back to default)
            # silently discarded the optimizer's finding whenever it hit the ceiling.
            optimized[name] = round(hi, 4)
            setattr(config, name, hi)
            print(f"  Param {name} hit upper bound ({val:.4f}) -> using bound value {hi} "
                  f"(consider widening bounds if this persists)")

    for name, _, _ in PARAM_SPEC:
        if name not in optimized:
            optimized[name] = round(defaults.get(name, getattr(ModelConfig(), name)), 4)
            setattr(config, name, optimized[name])

    return optimized, config


def print_report(metrics, label=""):
    """Print calibration report."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
    print(f"  Games scored: {metrics['n_games']}")
    print(f"  Accuracy:     {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['n_games']})")
    print(f"  Brier score:  {metrics['brier_score']:.4f}")
    print(f"  Log loss:     {metrics['log_loss']:.4f}")
    print(f"  Spread MAE:   {metrics['spread_mae']:.1f} pts")
    if metrics.get("total_mae") and metrics["total_mae"] > 0:
        print(f"  Total bias:   {metrics['total_bias']:+.1f} pts  MAE: {metrics['total_mae']:.1f} pts")

    print(f"\n  Per-round breakdown:")
    round_order = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    for rname in round_order:
        rs = metrics["round_stats"].get(rname)
        if not rs:
            continue
        acc = rs["correct"] / rs["total"] if rs["total"] else 0
        brier = rs["brier"] / rs["total"] if rs["total"] else 0
        print(f"    {rname:16s}: {acc:.1%} ({rs['correct']}/{rs['total']})  Brier={brier:.4f}")


def save_config(optimized):
    """Save optimized config to JSON."""
    out_path = os.path.join(DATA_DIR, "calibrated_config.json")
    with open(out_path, "w") as f:
        json.dump(optimized, f, indent=2)
    print(f"\nSaved calibrated config to {out_path}")


def _metrics_to_json(metrics):
    """Convert metrics dict to JSON-serializable form."""
    out = {
        "n_games": metrics.get("n_games", 0),
        "accuracy": metrics.get("accuracy", 0),
        "brier_score": metrics.get("brier_score", 0),
        "log_loss": metrics.get("log_loss", 0),
        "spread_mae": metrics.get("spread_mae", 0),
        "total_bias": metrics.get("total_bias", 0),
        "total_mae": metrics.get("total_mae", 0),
    }
    if "round_stats" in metrics:
        out["round_stats"] = {
            r: {"correct": rs["correct"], "total": rs["total"], "brier": rs["brier"] / rs["total"] if rs["total"] else 0}
            for r, rs in metrics["round_stats"].items()
        }
    return out


def save_report(out_path, baseline_metrics, optimized_metrics=None, cv_folds=None, params=None):
    """Write calibration report JSON to out_path."""
    report = {"baseline": _metrics_to_json(baseline_metrics)}
    if optimized_metrics is not None:
        report["optimized"] = _metrics_to_json(optimized_metrics)
    if cv_folds is not None:
        report["cv_folds"] = cv_folds
    if params is not None:
        report["params"] = params
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")


def main():
    report_only = "--report-only" in sys.argv
    phase2_only = "--phase2-only" in sys.argv
    priors_only = "--priors-only" in sys.argv
    no_gate = "--no-gate" in sys.argv
    holdout_year = None
    exclude_years = None
    objective_name = "brier"
    maxiter = 100
    popsize = 16
    multi_start = 2
    use_cv_objective = "--no-cv" not in sys.argv
    save_report_path = None
    min_floor = 0.0
    reg_lambda = 0.002  # default L2 regularization strength (prevents overfitting)
    brier_f4_weights = (0.6, 0.4)
    for i, arg in enumerate(sys.argv):
        if arg == "--save-report" and i + 1 < len(sys.argv):
            save_report_path = sys.argv[i + 1]
        elif arg == "--holdout" and i + 1 < len(sys.argv):
            holdout_year = int(sys.argv[i + 1])
        elif arg == "--exclude-years" and i + 1 < len(sys.argv):
            exclude_years = [int(y.strip()) for y in sys.argv[i + 1].split(",") if y.strip()]
        elif arg == "--min-floor" and i + 1 < len(sys.argv):
            try:
                min_floor = float(sys.argv[i + 1])
            except ValueError:
                pass
        elif arg == "--objective" and i + 1 < len(sys.argv):
            objective_name = sys.argv[i + 1]
        elif arg == "--maxiter" and i + 1 < len(sys.argv):
            try:
                maxiter = int(sys.argv[i + 1])
            except ValueError:
                pass
        elif arg == "--popsize" and i + 1 < len(sys.argv):
            try:
                popsize = int(sys.argv[i + 1])
            except ValueError:
                pass
        elif arg == "--multi-start" and i + 1 < len(sys.argv):
            try:
                multi_start = int(sys.argv[i + 1])
            except ValueError:
                pass
        elif arg == "--regularization" and i + 1 < len(sys.argv):
            try:
                reg_lambda = float(sys.argv[i + 1])
            except ValueError:
                pass
        elif arg == "--no-regularization":
            reg_lambda = 0.0
        elif arg == "--brier-f4-weights" and i + 1 < len(sys.argv):
            parts = sys.argv[i + 1].split(",")
            if len(parts) >= 2:
                try:
                    w1, w2 = float(parts[0].strip()), float(parts[1].strip())
                    if w1 > 0 and w2 > 0:
                        brier_f4_weights = (w1, w2)
                except ValueError:
                    pass

    print("Loading historical results...")
    games = load_results()
    if exclude_years:
        games = [g for g in games if g.get("year") not in exclude_years]
        print(f"  {len(games)} games (excluded years: {exclude_years})")
    else:
        print(f"  {len(games)} games")

    print("Loading team stats...")
    teams_by_year = load_all_teams()
    print(f"  {len(teams_by_year)} years: {sorted(teams_by_year.keys())}")

    print("Building game pairs (matching teams to stats)...")
    pairs = build_game_pairs(games, teams_by_year)
    print(f"  {len(pairs)} matchable games")
    partial_weights = build_partial_year_weight_overrides(pairs)
    if partial_weights:
        shown = ", ".join(f"{yr}x{mult:.2f}" for yr, mult in sorted(partial_weights.items()))
        print(f"  Partial-year weighting: {shown}")
    if reg_lambda > 0:
        print(f"  L2 regularization: λ={reg_lambda}")

    # Score with baseline (calibrated_config.json if exists, else engine defaults)
    cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
    baseline_config = replace(DEFAULT_CONFIG, num_sims=1)
    baseline_label = "CURRENT BASELINE" if os.path.isfile(cal_path) else "DEFAULT CONFIG"
    default_metrics = score_model(pairs, baseline_config)
    print_report(default_metrics, baseline_label)

    if report_only:
        if holdout_year:
            holdout_pairs = [p for p in pairs if p[4] == holdout_year]
            if holdout_pairs:
                holdout_metrics = score_model(holdout_pairs, baseline_config)
                print_report(holdout_metrics, f"HELD-OUT {holdout_year}")
        if save_report_path:
            save_report(save_report_path, default_metrics)
        return

    seeds = [42, 123, 456][:multi_start] if multi_start <= 3 else [42 + r for r in range(multi_start)]
    if priors_only:
        print(f"\n--- Priors only: objective={objective_name}, maxiter={maxiter}, popsize={popsize} ---")
        best_opt = None
        best_score = 1e9
        for run, s in enumerate(seeds):
            if multi_start > 1:
                print(f"\n  Multi-start run {run + 1}/{multi_start} (seed={s})")
            opt, cfg = calibrate_priors_only(pairs, objective_name, maxiter, popsize, s, reg_lambda=reg_lambda)
            m = score_model(pairs, cfg)
            sc = _compute_objective(m, objective_name)
            if sc < best_score:
                best_score = sc
                best_opt = (opt, cfg)
        optimized, opt_config = best_opt
    elif phase2_only:
        print(f"\n--- Phase 2 only: objective={objective_name}, maxiter={maxiter}, popsize={popsize} ---")
        best_opt = None
        best_score = 1e9
        for run, s in enumerate(seeds):
            if multi_start > 1:
                print(f"\n  Multi-start run {run + 1}/{multi_start} (seed={s})")
            opt, cfg = calibrate_phase2_only(pairs, objective_name, maxiter, popsize, s, reg_lambda=reg_lambda)
            m = score_model(pairs, cfg)
            sc = _compute_objective(m, objective_name)
            if sc < best_score:
                best_score = sc
                best_opt = (opt, cfg)
        optimized, opt_config = best_opt
    else:
        # Walk-forward calibration
        best_opt = None
        best_cv_brier = 1e9
        for run, s in enumerate(seeds):
            if multi_start > 1:
                print(f"\n  Multi-start run {run + 1}/{multi_start} (seed={s})")
            opt, cfg = calibrate_walk_forward(
                pairs, objective_name, maxiter, popsize, s, use_cv_objective=use_cv_objective, min_floor=min_floor,
                games=games if objective_name == "brier+f4" else None,
                brier_f4_weights=brier_f4_weights, reg_lambda=reg_lambda
            )
            pairs_by_year = _pairs_by_year(pairs)
            test_years = [y for y in sorted(pairs_by_year.keys()) if y >= 2017]
            cv_briers = []
            for ty in test_years:
                train_p, test_p = _fold_pairs(pairs_by_year, ty)
                if not test_p:
                    continue
                c = replace(cfg, num_sims=1)
                for k, v in opt.items():
                    setattr(c, k, v)
                cv_briers.append(score_model(test_p, c)["brier_score"])
            cv_b = sum(cv_briers) / len(cv_briers) if cv_briers else 1e9
            if cv_b < best_cv_brier:
                best_cv_brier = cv_b
                best_opt = (opt, cfg)
        optimized, opt_config = best_opt
    print(f"\nOptimized parameters:")
    for k, v in optimized.items():
        default_val = getattr(DEFAULT_CONFIG, k)
        print(f"  {k}: {default_val} -> {v}")

    gate_report = None
    if not no_gate:
        gate_report, keep_coach, keep_pedigree, opt_config, optimized = apply_prior_gate(
            pairs, opt_config, optimized=optimized, enabled=True
        )
        print_prior_gate_report(
            gate_report,
            keep_coach,
            keep_pedigree,
            recent_years=_resolve_prior_gate_recent_years(pairs),
        )

    # Score with optimized config (full data)
    opt_metrics = score_model(pairs, opt_config)
    label = "OPTIMIZED CONFIG (all data)"
    if gate_report is not None:
        label = "OPTIMIZED + GATED CONFIG (all data)"
    print_report(opt_metrics, label)

    # NOTE: --holdout eval uses the final model which WAS trained on holdout_year data.
    # The honest out-of-sample estimate is the cross-validated Brier printed above.
    if holdout_year:
        holdout_pairs = [p for p in pairs if p[4] == holdout_year]
        if holdout_pairs:
            holdout_metrics = score_model(holdout_pairs, opt_config)
            print_report(holdout_metrics, f"IN-SAMPLE CHECK: {holdout_year} (NOTE: model trained on this data)")
            print(f"  WARNING: This is NOT a true holdout — use CV Brier above for honest estimate.")

    # Improvement summary
    brier_delta = default_metrics["brier_score"] - opt_metrics["brier_score"]
    acc_delta = opt_metrics["accuracy"] - default_metrics["accuracy"]
    print(f"\n  Improvement: Brier {brier_delta:+.4f}, Accuracy {acc_delta:+.1%}")

    save_config(optimized)

    if save_report_path:
        pairs_by_year = _pairs_by_year(pairs)
        test_years = [y for y in sorted(pairs_by_year.keys()) if y >= 2017]
        cv_folds = []
        for ty in test_years:
            _, test_p = _fold_pairs(pairs_by_year, ty)
            if test_p:
                m = score_model(test_p, opt_config)
                cv_folds.append({"year": ty, "brier": m["brier_score"], "accuracy": m["accuracy"]})
        save_report(save_report_path, default_metrics, opt_metrics, cv_folds, optimized)


if __name__ == "__main__":
    main()
