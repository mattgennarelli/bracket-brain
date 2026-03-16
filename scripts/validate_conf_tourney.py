"""
M3.7: Validate conf tourney momentum — compare Brier and accuracy with/without conf_tourney_max_bonus.

Usage:
  python scripts/validate_conf_tourney.py
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from engine import ModelConfig
from scripts.calibrate import load_results, load_all_teams, build_game_pairs, score_model


def load_calibrated_config():
    path = os.path.join(DATA_DIR, "calibrated_config.json")
    if not os.path.isfile(path):
        return ModelConfig()
    with open(path) as f:
        data = json.load(f)
    config = ModelConfig()
    for k, v in data.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config


def main():
    print("M3.7 Conf Tourney Calibration Validation")
    print("=" * 50)

    games = load_results()
    teams_by_year = load_all_teams()
    pairs = build_game_pairs(games, teams_by_year)
    print(f"Loaded {len(pairs)} game pairs from {len(games)} games")

    config_full = load_calibrated_config()
    conf_val = getattr(config_full, "conf_tourney_max_bonus", 0.5)
    print(f"\nCalibrated conf_tourney_max_bonus: {conf_val}")

    # With conf tourney
    metrics_full = score_model(pairs, config_full)
    print(f"\nWith conf_tourney (full):")
    print(f"  Brier: {metrics_full['brier_score']:.4f}")
    print(f"  Accuracy: {metrics_full['accuracy']:.1%}")

    # Without conf tourney (ablation)
    config_ablated = load_calibrated_config()
    config_ablated.conf_tourney_max_bonus = 0.0
    metrics_ablated = score_model(pairs, config_ablated)
    print(f"\nWithout conf_tourney (ablation, conf_tourney_max_bonus=0):")
    print(f"  Brier: {metrics_ablated['brier_score']:.4f}")
    print(f"  Accuracy: {metrics_ablated['accuracy']:.1%}")

    brier_diff = metrics_full["brier_score"] - metrics_ablated["brier_score"]
    acc_diff = (metrics_full["accuracy"] or 0) - (metrics_ablated["accuracy"] or 0)
    print(f"\nDifference (full - ablated):")
    print(f"  Brier: {brier_diff:+.4f} (negative = conf tourney helps)")
    print(f"  Accuracy: {acc_diff:+.1%}")

    if brier_diff < 0:
        print("\nConclusion: conf_tourney momentum IMPROVES Brier (keep signal)")
    elif brier_diff > 0.001:
        print("\nConclusion: conf_tourney momentum WORSENS Brier (consider reducing or removing)")
    else:
        print("\nConclusion: conf_tourney momentum has negligible impact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
