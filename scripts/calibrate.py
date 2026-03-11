"""
Calibrate the prediction model against historical tournament results.

Loads results_all.json + teams_merged_YYYY.json for each year, runs
predict_game for every historical matchup, and optimizes ModelConfig
parameters to minimize Brier score.

Output: data/calibrated_config.json

Usage:
  python scripts/calibrate.py                # optimize + report
  python scripts/calibrate.py --report-only  # just score current params
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
    round_stats = {}
    n = len(pairs)

    for a, b, a_won, actual_margin, yr, g in pairs:
        try:
            result = predict_game(enrich_team(a), enrich_team(b), config=config)
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

        rname = g.get("round_name", "Unknown")
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
    }


def calibrate(pairs):
    """Optimize ModelConfig parameters to minimize Brier score using global search."""
    from scipy.optimize import differential_evolution

    param_spec = [
        ("seed_weight", 0.0, 0.50),
        ("base_scoring_stdev", 8.0, 18.0),
        # Secondary signals — capped tightly to prevent efficiency-proxy overfitting
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
        # win_pct only fires for elite (>85%) teams — small bonus, not a proxy for adj_o
        ("win_pct_max_bonus", 0.0, 5.0),
        ("conf_rating_max_bonus", 0.0, 4.0),
        # EvanMiya-sourced signals — depth/stars capped to prevent dominating
        ("depth_max_bonus", 0.0, 8.0),
        ("em_opp_adjust_max_bonus", 0.0, 8.0),
        ("em_adj_o_weight", 0.0, 1.0),        # how much EvanMiya efficiency shifts base score
        # Foul drawing / committing rate edge
        ("ft_foul_rate_max_bonus", 0.0, 6.0),
    ]

    eval_count = [0]
    best_brier = [1.0]

    def objective(x):
        config = ModelConfig(num_sims=1)
        for i, (name, _, _) in enumerate(param_spec):
            setattr(config, name, x[i])
        metrics = score_model(pairs, config)
        eval_count[0] += 1
        if metrics["brier_score"] < best_brier[0]:
            best_brier[0] = metrics["brier_score"]
            if eval_count[0] % 50 == 0:
                print(f"  [{eval_count[0]}] Best Brier: {best_brier[0]:.5f}")
        return metrics["brier_score"]

    bounds = [(lo, hi) for _, lo, hi in param_spec]

    print(f"\nOptimizing {len(param_spec)} parameters over {len(pairs)} games...")
    defaults = {name: getattr(ModelConfig(), name) for name, _, _ in param_spec}
    print(f"Defaults: {defaults}")

    result = differential_evolution(objective, bounds, seed=42,
                                    maxiter=60, tol=1e-6, popsize=12,
                                    mutation=(0.5, 1.5), recombination=0.8)

    optimized = {}
    config = ModelConfig(num_sims=1)
    for i, (name, _, _) in enumerate(param_spec):
        val = float(result.x[i])
        optimized[name] = round(val, 4)
        setattr(config, name, val)

    print(f"  Converged after {eval_count[0]} evaluations (Brier={result.fun:.5f})")
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
        return

    # Optimize
    optimized, opt_config = calibrate(pairs)
    print(f"\nOptimized parameters:")
    for k, v in optimized.items():
        default_val = getattr(ModelConfig(), k)
        print(f"  {k}: {default_val} -> {v}")

    # Score with optimized config
    opt_metrics = score_model(pairs, opt_config)
    print_report(opt_metrics, "OPTIMIZED CONFIG")

    # Improvement summary
    brier_delta = default_metrics["brier_score"] - opt_metrics["brier_score"]
    acc_delta = opt_metrics["accuracy"] - default_metrics["accuracy"]
    print(f"\n  Improvement: Brier {brier_delta:+.4f}, Accuracy {acc_delta:+.1%}")

    save_config(optimized)


if __name__ == "__main__":
    main()
