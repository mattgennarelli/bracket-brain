"""
Calibrate the prediction model against historical tournament results.

Loads results_all.json + teams_merged_YYYY.json for each year, runs
predict_game for every historical matchup, and optimizes ModelConfig
parameters to minimize Brier score.

Uses walk-forward cross-validation: train on years N-k, test on year N.
Drops params with |calibrated - default| < 0.05 across folds; target ≤ 12 params.

Output: data/calibrated_config.json

Usage:
  python scripts/calibrate.py                # optimize + report (walk-forward)
  python scripts/calibrate.py --report-only   # just score current params
  python scripts/calibrate.py --holdout 2025  # hold out 2025 for final eval
"""
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from engine import predict_game, ModelConfig, _normalize_team_for_match, enrich_team


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


def score_model(pairs, config):
    """Evaluate model on all game pairs. Returns metrics dict."""
    brier_sum = 0.0
    log_loss_sum = 0.0
    correct = 0
    margin_errors = []
    total_errors = []
    round_stats = {}
    n = len(pairs)

    for a, b, a_won, actual_margin, yr, g in pairs:
        rname = g.get("round_name", "Unknown")
        try:
            result = predict_game(enrich_team(a), enrich_team(b), config=config,
                                  round_name=rname)
        except Exception:
            continue
        prob_a = result["win_prob_a"]
        prob_a = max(0.001, min(0.999, prob_a))

        brier_sum += (prob_a - a_won) ** 2
        log_loss_sum -= (a_won * math.log(prob_a) + (1 - a_won) * math.log(1 - prob_a))

        predicted_winner_is_a = prob_a >= 0.5
        if predicted_winner_is_a == bool(a_won):
            correct += 1

        pred_margin = result["predicted_margin"]
        actual_signed = actual_margin if a_won else -actual_margin
        margin_errors.append(abs(pred_margin - actual_signed))

        # Total prediction tracking
        score_a = g.get("score_a")
        score_b = g.get("score_b")
        if score_a is not None and score_b is not None:
            predicted_total = result["predicted_score_a"] + result["predicted_score_b"]
            total_errors.append(predicted_total - (score_a + score_b))

        if rname not in round_stats:
            round_stats[rname] = {"correct": 0, "total": 0, "brier": 0.0}
        round_stats[rname]["total"] += 1
        round_stats[rname]["brier"] += (prob_a - a_won) ** 2
        if predicted_winner_is_a == bool(a_won):
            round_stats[rname]["correct"] += 1

    return {
        "brier_score": brier_sum / n if n else 1.0,
        "log_loss": log_loss_sum / n if n else 10.0,
        "accuracy": correct / n if n else 0.0,
        "spread_mae": sum(margin_errors) / len(margin_errors) if margin_errors else 99,
        "n_games": n,
        "correct": correct,
        "round_stats": round_stats,
        "total_bias": sum(total_errors) / len(total_errors) if total_errors else 0,
        "total_mae": sum(abs(e) for e in total_errors) / len(total_errors) if total_errors else 0,
    }


PARAM_SPEC = [
    # seed_weight removed: efficiency model already captures seed info;
    # 0.0 ties or beats 0.18 on 2023-2025 folds (Brier diff < 0.002)
    ("seed_weight", 0.0, 8.0),
    ("base_scoring_stdev", 8.0, 18.0),
    ("sos_max_bonus", 0.0, 8.0),
    ("possession_edge_max_bonus", 0.0, 8.0),
    ("ft_clutch_max_bonus", 0.0, 6.0),
    ("experience_max_bonus", 0.0, 4.0),
    ("coach_tourney_max_bonus", 0.0, 6.0),
    ("pedigree_max_bonus", 0.0, 6.0),
    ("three_pt_volatility_factor", 0.0, 3.0),
    ("tempo_volatility_weight", 0.0, 4.0),
    ("star_player_max_bonus", 0.0, 8.0),
    ("momentum_max_bonus", 0.0, 5.0),
    ("win_pct_max_bonus", 0.0, 5.0),
    ("conf_rating_max_bonus", 0.0, 4.0),
    ("depth_max_bonus", 0.0, 8.0),
    ("em_opp_adjust_max_bonus", 0.0, 8.0),
    ("em_adj_o_weight", 0.0, 1.0),
    ("ft_foul_rate_max_bonus", 0.0, 6.0),
    # Per-round score scaling (Phase 1)
    ("score_scale", 0.88, 1.00),
    ("score_scale_r64", 0.90, 1.00),
    ("score_scale_r32", 0.90, 1.00),
    ("score_scale_s16", 0.88, 0.98),
    ("score_scale_e8", 0.86, 0.96),
    ("score_scale_ff", 0.85, 0.95),
    # Per-round stdev inflation (Phase 2)
    ("round_stdev_inflation_e8", 1.0, 1.25),
    ("round_stdev_inflation_ff", 1.0, 1.30),
    # Late-round dampening: pull win-probs toward 0.5 in Sweet 16+ (reduces overconfidence)
    ("late_round_dampening", 0.0, 0.35),
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


def calibrate_single(pairs, param_spec):
    """Optimize params on given pairs. Returns (optimized_dict, config)."""
    from scipy.optimize import differential_evolution

    eval_count = [0]
    best_brier = [1.0]

    def objective(x):
        config = ModelConfig(num_sims=1)
        for i, (name, _, _) in enumerate(param_spec):
            setattr(config, name, x[i])
        metrics = score_model(pairs, config)
        eval_count[0] += 1
        # Blend: Brier is primary metric; total_mae is tiebreaker (0.001 weight)
        score = metrics["brier_score"] + 0.001 * metrics.get("total_mae", 0)
        if metrics["brier_score"] < best_brier[0]:
            best_brier[0] = metrics["brier_score"]
            if eval_count[0] % 50 == 0:
                print(f"  [{eval_count[0]}] Best Brier: {best_brier[0]:.5f}")
        return score

    bounds = [(lo, hi) for _, lo, hi in param_spec]
    result = differential_evolution(objective, bounds, seed=42,
                                    maxiter=60, tol=1e-6, popsize=12,
                                    mutation=(0.5, 1.5), recombination=0.8)

    optimized = {}
    config = ModelConfig(num_sims=1)
    for i, (name, _, _) in enumerate(param_spec):
        val = float(result.x[i])
        optimized[name] = round(val, 4)
        setattr(config, name, val)
    return optimized, config


def calibrate_walk_forward(pairs):
    """Walk-forward: train on N-k, test on N. Reduce params; return final config."""
    pairs_by_year = _pairs_by_year(pairs)
    years = sorted(pairs_by_year.keys())
    if len(years) < 3:
        print("  Not enough years for walk-forward; falling back to single calibration.")
        return calibrate_single(pairs, PARAM_SPEC)

    # Use all years >= 2017 as test folds for more stable CV estimate
    test_years = [y for y in sorted(pairs_by_year.keys()) if y >= 2017]
    fold_results = []  # list of (test_year, train_optimized, test_brier)

    for test_year in test_years:
        train_pairs, test_pairs = _fold_pairs(pairs_by_year, test_year)
        if not train_pairs or not test_pairs:
            continue
        print(f"\n--- Fold: train on years < {test_year}, test on {test_year} ({len(test_pairs)} games) ---")
        opt, _ = calibrate_single(train_pairs, PARAM_SPEC)
        config = ModelConfig(num_sims=1)
        for k, v in opt.items():
            setattr(config, k, v)
        test_metrics = score_model(test_pairs, config)
        fold_results.append((test_year, opt, test_metrics["brier_score"]))
        print(f"  Test Brier on {test_year}: {test_metrics['brier_score']:.4f}")

    fold_briers = [b for _, _, b in fold_results]
    cv_brier = sum(fold_briers) / len(fold_briers)
    print(f"\n  Cross-validated Brier (avg of {len(fold_briers)} folds): {cv_brier:.4f}  ← honest out-of-sample estimate")

    # Reduce params: drop any with |calibrated - default| < 0.05 across all folds
    defaults = {name: getattr(ModelConfig(), name) for name, _, _ in PARAM_SPEC}
    param_deltas = {name: [] for name, _, _ in PARAM_SPEC}
    for _, opt, _ in fold_results:
        for name in opt:
            param_deltas[name].append(abs(opt[name] - defaults.get(name, 0)))

    # Keep params that move meaningfully (max_delta >= 0.05)
    reduced_spec = [(n, lo, hi) for n, lo, hi in PARAM_SPEC if max(param_deltas.get(n, [0])) >= 0.05]

    if len(reduced_spec) > 12:
        # Top 12 by max delta
        ranked = [(n, max(param_deltas.get(n, [0]))) for n, _, _ in PARAM_SPEC]
        ranked.sort(key=lambda x: -x[1])
        keep = {n for n, _ in ranked[:12]}
        reduced_spec = [(n, lo, hi) for n, lo, hi in PARAM_SPEC if n in keep]

    print(f"\n  Reduced to {len(reduced_spec)} params (target ≤12): {[p[0] for p in reduced_spec]}")

    # Final calibration on all data except holdout (if any)
    all_train = [p for p in pairs if p[4] < max(years)]
    print(f"\n--- Final calibration on {len(all_train)} games (all years except {max(years)}) ---")
    optimized, config = calibrate_single(all_train, reduced_spec)

    # M1: Drop params that hit bounds — use 0 at lower bound (remove signal), default at upper
    bounds_map = {n: (lo, hi) for n, lo, hi in PARAM_SPEC}
    for name in list(optimized.keys()):
        lo, hi = bounds_map.get(name, (0, 1))
        val = optimized[name]
        if val <= lo + 0.01:
            optimized[name] = 0.0
            setattr(config, name, 0.0)
            print(f"  Param {name} hit lower bound ({val}) -> removed (set to 0)")
        elif val >= hi - 0.01:
            default_val = defaults.get(name, getattr(ModelConfig(), name))
            optimized[name] = round(default_val, 4)
            setattr(config, name, default_val)
            print(f"  Param {name} hit upper bound ({val}) -> using default {default_val}")

    # Fill in dropped params with defaults
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


def main():
    report_only = "--report-only" in sys.argv
    holdout_year = None
    for i, arg in enumerate(sys.argv):
        if arg == "--holdout" and i + 1 < len(sys.argv):
            holdout_year = int(sys.argv[i + 1])
            break

    print("Loading historical results...")
    games = load_results()
    print(f"  {len(games)} games")

    print("Loading team stats...")
    teams_by_year = load_all_teams()
    print(f"  {len(teams_by_year)} years: {sorted(teams_by_year.keys())}")

    print("Building game pairs (matching teams to stats)...")
    pairs = build_game_pairs(games, teams_by_year)
    print(f"  {len(pairs)} matchable games")

    # Score with default config
    default_metrics = score_model(pairs, ModelConfig(num_sims=1))
    print_report(default_metrics, "DEFAULT CONFIG")

    if report_only:
        if holdout_year:
            holdout_pairs = [p for p in pairs if p[4] == holdout_year]
            if holdout_pairs:
                holdout_metrics = score_model(holdout_pairs, ModelConfig(num_sims=1))
                print_report(holdout_metrics, f"HELD-OUT {holdout_year}")
        return

    # Walk-forward calibration
    optimized, opt_config = calibrate_walk_forward(pairs)
    print(f"\nOptimized parameters:")
    for k, v in optimized.items():
        default_val = getattr(ModelConfig(), k)
        print(f"  {k}: {default_val} -> {v}")

    # Score with optimized config (full data)
    opt_metrics = score_model(pairs, opt_config)
    print_report(opt_metrics, "OPTIMIZED CONFIG (all data)")

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


if __name__ == "__main__":
    main()
