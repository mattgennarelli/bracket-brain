"""
benchmark.py — Compare prediction models on held-out tournament data.

Baselines:
  seed-only     Always pick lower seed number; win_prob proportional to seed gap
  efficiency    Predict based on adj_o - adj_d differential only (no extras)
  full          Full calibrated model (calibrated_config.json)

Uses walk-forward CV: for each test year, train baseline efficiency model on
years before it and evaluate on that year alone. Reports Brier, accuracy, and
per-round breakdown for each model.

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --year 2025    # single year only
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
from scripts.calibrate import (
    load_results, load_all_teams, build_game_pairs,
    score_model, _fold_pairs, _pairs_by_year, calibrate_single, PARAM_SPEC,
)


# ---------------------------------------------------------------------------
# Baseline predictors
# ---------------------------------------------------------------------------

# Seed-gap -> historical upset rate table (from NCCA data 1985-2025)
SEED_WIN_PROB = {
    (1, 16): 0.991,
    (2, 15): 0.938,
    (3, 14): 0.849,
    (4, 13): 0.793,
    (5, 12): 0.647,
    (6, 11): 0.630,
    (7, 10): 0.609,
    (8, 9):  0.509,
}


def _seed_prob(seed_a, seed_b):
    """Return historical win prob for seed_a vs seed_b."""
    lo, hi = min(seed_a, seed_b), max(seed_a, seed_b)
    key = (lo, hi)
    p = SEED_WIN_PROB.get(key)
    if p is None:
        # Later rounds: estimate from seed gap
        gap = abs(seed_a - seed_b)
        p = 0.5 + min(gap * 0.04, 0.45)
    return p if seed_a < seed_b else 1 - p


def score_seed_only(pairs):
    """Predict using historical seed matchup win rates only."""
    brier_sum = 0.0
    log_loss_sum = 0.0
    correct = 0
    round_stats = {}
    n = 0

    for a, b, a_won, _, _, g in pairs:
        seed_a = a.get("seed", 8)
        seed_b = b.get("seed", 9)
        prob_a = _seed_prob(seed_a, seed_b)
        prob_a = max(0.001, min(0.999, prob_a))

        brier_sum += (prob_a - a_won) ** 2
        log_loss_sum -= (a_won * math.log(prob_a) + (1 - a_won) * math.log(1 - prob_a))
        if (prob_a >= 0.5) == bool(a_won):
            correct += 1
        n += 1

        rname = g.get("round_name", "Unknown")
        if rname not in round_stats:
            round_stats[rname] = {"correct": 0, "total": 0, "brier": 0.0}
        round_stats[rname]["total"] += 1
        round_stats[rname]["brier"] += (prob_a - a_won) ** 2
        if (prob_a >= 0.5) == bool(a_won):
            round_stats[rname]["correct"] += 1

    return {
        "brier_score": brier_sum / n if n else 1.0,
        "log_loss": log_loss_sum / n if n else 10.0,
        "accuracy": correct / n if n else 0.0,
        "n_games": n,
        "correct": correct,
        "round_stats": round_stats,
    }


def score_efficiency_only(pairs):
    """Predict using adj_o - adj_d differential only (no extras). Logistic mapping."""
    brier_sum = 0.0
    log_loss_sum = 0.0
    correct = 0
    round_stats = {}
    n = 0

    for a, b, a_won, _, _, g in pairs:
        net_a = a.get("adj_o", 100) - a.get("adj_d", 100)
        net_b = b.get("adj_o", 100) - b.get("adj_d", 100)
        diff = net_a - net_b
        # Logistic: scale factor ~0.15 gives reasonable spread
        prob_a = 1.0 / (1.0 + math.exp(-diff * 0.15))
        prob_a = max(0.001, min(0.999, prob_a))

        brier_sum += (prob_a - a_won) ** 2
        log_loss_sum -= (a_won * math.log(prob_a) + (1 - a_won) * math.log(1 - prob_a))
        if (prob_a >= 0.5) == bool(a_won):
            correct += 1
        n += 1

        rname = g.get("round_name", "Unknown")
        if rname not in round_stats:
            round_stats[rname] = {"correct": 0, "total": 0, "brier": 0.0}
        round_stats[rname]["total"] += 1
        round_stats[rname]["brier"] += (prob_a - a_won) ** 2
        if (prob_a >= 0.5) == bool(a_won):
            round_stats[rname]["correct"] += 1

    return {
        "brier_score": brier_sum / n if n else 1.0,
        "log_loss": log_loss_sum / n if n else 10.0,
        "accuracy": correct / n if n else 0.0,
        "n_games": n,
        "correct": correct,
        "round_stats": round_stats,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

ROUND_ORDER = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]


def print_comparison(year, results_by_model):
    """Print side-by-side comparison for one year."""
    print(f"\n{'─' * 72}")
    print(f"  Year: {year}  ({results_by_model[list(results_by_model.keys())[0]]['n_games']} games)")
    print(f"{'─' * 72}")
    header = f"  {'Round':16s}"
    for name in results_by_model:
        header += f"  {name:>22s}"
    print(header)
    print(f"  {'':16s}" + "".join(f"  {'Acc':>10s} {'Brier':>10s}" for _ in results_by_model))
    print(f"  {'─' * 16}" + ("  " + "─" * 22) * len(results_by_model))

    for rname in ROUND_ORDER + ["TOTAL"]:
        row = f"  {rname:16s}"
        for metrics in results_by_model.values():
            if rname == "TOTAL":
                acc = metrics["accuracy"]
                brier = metrics["brier_score"]
                n = metrics["n_games"]
                row += f"  {acc:>9.1%}  {brier:>9.4f}"
            else:
                rs = metrics.get("round_stats", {}).get(rname)
                if rs and rs["total"] > 0:
                    acc = rs["correct"] / rs["total"]
                    brier = rs["brier"] / rs["total"]
                    row += f"  {acc:>9.1%}  {brier:>9.4f}"
                else:
                    row += f"  {'—':>9s}  {'—':>9s}"
        print(row)


def print_cv_summary(cv_results):
    """Print cross-validated summary table across years."""
    print(f"\n{'=' * 72}")
    print(f"  CROSS-VALIDATED SUMMARY")
    print(f"{'=' * 72}")
    models = list(cv_results[list(cv_results.keys())[0]].keys())
    print(f"  {'Model':25s}  {'CV Brier':>10s}  {'CV Accuracy':>12s}  {'vs Seed':>10s}  {'vs Eff':>10s}")
    print(f"  {'─' * 25}  {'─' * 10}  {'─' * 12}  {'─' * 10}  {'─' * 10}")

    model_avgs = {}
    for model in models:
        briers = [cv_results[yr][model]["brier_score"] for yr in cv_results]
        accs = [cv_results[yr][model]["accuracy"] for yr in cv_results]
        model_avgs[model] = {
            "brier": sum(briers) / len(briers),
            "accuracy": sum(accs) / len(accs),
        }

    seed_brier = model_avgs.get("seed-only", {}).get("brier", 1.0)
    eff_brier = model_avgs.get("efficiency", {}).get("brier", 1.0)

    for model, avgs in model_avgs.items():
        vs_seed = avgs["brier"] - seed_brier
        vs_eff = avgs["brier"] - eff_brier
        vs_seed_str = f"{vs_seed:+.4f}" if model != "seed-only" else "—"
        vs_eff_str = f"{vs_eff:+.4f}" if model != "efficiency" else "—"
        print(f"  {model:25s}  {avgs['brier']:>10.4f}  {avgs['accuracy']:>11.1%}  {vs_seed_str:>10s}  {vs_eff_str:>10s}")

    print()
    print(f"  Per-year Brier:")
    print(f"  {'Year':6s}" + "".join(f"  {m:>12s}" for m in models))
    for yr in sorted(cv_results.keys()):
        row = f"  {yr:6d}"
        for m in models:
            b = cv_results[yr][m]["brier_score"]
            row += f"  {b:>12.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    single_year = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--year" and i + 2 <= len(sys.argv) - 1:
            single_year = int(sys.argv[i + 2])

    print("Loading data...")
    games = load_results()
    teams_by_year = load_all_teams()
    pairs = build_game_pairs(games, teams_by_year)
    pairs_by_year = _pairs_by_year(pairs)

    with open(os.path.join(DATA_DIR, "calibrated_config.json")) as f:
        full_config = ModelConfig(**json.load(f))

    # Test years: last 3 (or single year)
    all_years = sorted(pairs_by_year.keys())
    test_years = [single_year] if single_year else all_years[-3:]

    cv_results = {}

    for test_year in test_years:
        train_pairs, test_pairs = _fold_pairs(pairs_by_year, test_year)
        if not test_pairs:
            print(f"  No games for {test_year}, skipping.")
            continue

        seed_metrics = score_seed_only(test_pairs)
        eff_metrics = score_efficiency_only(test_pairs)
        full_metrics = score_model(test_pairs, full_config)

        cv_results[test_year] = {
            "seed-only": seed_metrics,
            "efficiency": eff_metrics,
            "full model": full_metrics,
        }

        print_comparison(test_year, cv_results[test_year])

    if len(cv_results) > 1:
        print_cv_summary(cv_results)
    elif cv_results:
        yr = list(cv_results.keys())[0]
        m = cv_results[yr]
        print(f"\n  seed-only Brier: {m['seed-only']['brier_score']:.4f}")
        print(f"  efficiency Brier: {m['efficiency']['brier_score']:.4f}")
        print(f"  full model Brier: {m['full model']['brier_score']:.4f}")


if __name__ == "__main__":
    main()
