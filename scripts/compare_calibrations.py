"""
Run multiple calibration objectives and compare in-sample vs backtest performance.

Loads pairs (same as calibrate), runs Phase 2 optimization for each objective,
scores on full data and on backtest years (2017, 2019, 2021, 2023, 2024),
and prints a comparison table. Does NOT overwrite calibrated_config.json.

Usage:
  python scripts/compare_calibrations.py
  python scripts/compare_calibrations.py --quick
  python scripts/compare_calibrations.py --objectives brier brier-acc composite
"""
import json
import os
import sys
from dataclasses import replace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from engine import ModelConfig, DEFAULT_CONFIG

# Import calibrate helpers (after path setup)
import calibrate
from backtest import run_year

BACKTEST_YEARS = [2017, 2019, 2021, 2023, 2024]
DEFAULT_OBJECTIVES = ["brier", "brier-acc", "brier-close", "brier-upset", "composite"]


def load_baseline_config():
    """Load current calibrated_config.json or fall back to engine defaults."""
    cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
    config = replace(DEFAULT_CONFIG, num_sims=1)
    if os.path.isfile(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        for k, v in cal.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return config


def run_backtest_aggregate(config, years):
    """Run backtest for each year, return aggregate accuracy and Brier."""
    total_correct = 0
    total_games = 0
    brier_sum = 0.0
    n_years = 0
    for y in years:
        m = run_year(y, config=config, upset_aggression=0.0, verbose=False)
        if m:
            total_correct += m.get("correct", 0)
            total_games += m.get("n_games", 0)
            brier_sum += m.get("brier_score", 0)
            n_years += 1
    acc = total_correct / total_games if total_games else 0.0
    avg_brier = brier_sum / n_years if n_years else 0.0
    return acc, avg_brier


def parse_args():
    """Parse --quick and --objectives."""
    quick = "--quick" in sys.argv
    objectives = list(DEFAULT_OBJECTIVES)
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--objectives" and i + 1 < len(args):
            objectives = []
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                objectives.append(args[i])
                i += 1
            continue
        i += 1
    return quick, objectives


def main():
    quick, objectives = parse_args()
    maxiter = 30 if quick else 60
    popsize = 8 if quick else 12

    print("Loading historical results...")
    games = calibrate.load_results()
    print(f"  {len(games)} games")

    print("Loading team stats...")
    teams_by_year = calibrate.load_all_teams()
    print(f"  {len(teams_by_year)} years")

    print("Building game pairs...")
    pairs = calibrate.build_game_pairs(games, teams_by_year)
    print(f"  {len(pairs)} matchable games")

    baseline_config = load_baseline_config()
    baseline_metrics = calibrate.score_model(pairs, baseline_config)
    baseline_acc, baseline_brier = run_backtest_aggregate(baseline_config, BACKTEST_YEARS)

    results = [
        ("baseline", baseline_metrics["accuracy"], baseline_metrics["brier_score"],
         baseline_acc, baseline_brier, None),
    ]

    for obj in objectives:
        print(f"\n--- Optimizing: {obj} (maxiter={maxiter}, popsize={popsize}) ---")
        opt, cfg = calibrate.calibrate_phase2_only(pairs, obj, maxiter, popsize, seed=42)
        in_metrics = calibrate.score_model(pairs, cfg)
        bt_acc, bt_brier = run_backtest_aggregate(cfg, BACKTEST_YEARS)
        results.append((obj, in_metrics["accuracy"], in_metrics["brier_score"],
                       bt_acc, bt_brier, cfg))

    # Print comparison table
    print("\n" + "=" * 75)
    print("  CALIBRATION COMPARISON")
    print("=" * 75)
    print(f"{'OBJECTIVE':<14} | {'IN-SAMPLE ACC':>12} | {'IN-SAMPLE BRIER':>14} | "
          f"{'BACKTEST ACC':>12} | {'BACKTEST BRIER':>14}")
    print("-" * 75)
    for name, in_acc, in_brier, bt_acc, bt_brier, _ in results:
        print(f"{name:<14} | {in_acc:>11.1%} | {in_brier:>14.4f} | "
              f"{bt_acc:>11.1%} | {bt_brier:>14.4f}")
    print("=" * 75)
    print(f"\nBacktest years: {BACKTEST_YEARS}")
    print("(calibrated_config.json was NOT modified)")


if __name__ == "__main__":
    main()
