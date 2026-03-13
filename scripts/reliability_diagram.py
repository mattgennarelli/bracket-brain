"""
Generate a calibration reliability diagram from backtest results.

Runs predict_game on every historical matchup, bins predictions into
probability buckets, and reports actual win rate vs predicted probability
for each bucket.

Usage:
  python scripts/reliability_diagram.py 2023 2024 2025
  python scripts/reliability_diagram.py 2017 2019 2021 2023 2024 2025

Outputs:
  data/reliability_diagram.json — binned calibration data
  ASCII table to stdout showing calibration error per bin
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from engine import predict_game, ModelConfig, enrich_team
from backtest import load_results, load_teams_for_year, _lookup


def run_reliability(years):
    """Compute calibration data across all given years.

    Bins predictions into 50-55%, 55-60%, ..., 95-100% buckets.
    Each game contributes one data point: the higher win probability
    assigned to either team, and whether that team actually won.
    """
    # 10 bins from 50% to 100%
    bins = [(0.50 + i * 0.05, 0.50 + (i + 1) * 0.05) for i in range(10)]
    bin_data = [{"lo": lo, "hi": hi, "n": 0, "wins": 0} for lo, hi in bins]

    config = ModelConfig()

    for year in years:
        actual_games = load_results(year)
        if not actual_games:
            print(f"  [{year}] No results file, skipping.")
            continue
        teams = load_teams_for_year(year)
        n_year = 0

        for g in actual_games:
            a = _lookup(g["team_a"], g["seed_a"], teams)
            b = _lookup(g["team_b"], g["seed_b"], teams)
            try:
                result = predict_game(enrich_team(a), enrich_team(b), config=config,
                                      round_name=g.get("round_name"))
            except Exception:
                continue
            prob_a = result["win_prob_a"]
            a_won = 1 if g["winner"] == g["team_a"] else 0

            # Bin by the "pick" side's probability (always >= 0.5)
            if prob_a >= 0.5:
                pred_prob = prob_a
                actual_win = a_won
            else:
                pred_prob = 1 - prob_a
                actual_win = 1 - a_won

            placed = False
            for bd in bin_data:
                if bd["lo"] <= pred_prob < bd["hi"]:
                    bd["n"] += 1
                    bd["wins"] += actual_win
                    placed = True
                    break
            if not placed and pred_prob >= 0.95:
                bin_data[-1]["n"] += 1
                bin_data[-1]["wins"] += actual_win
            n_year += 1

        print(f"  [{year}] {n_year} games processed.")

    # Compute calibration error for each bin
    results = []
    for bd in bin_data:
        if bd["n"] == 0:
            continue
        midpoint = (bd["lo"] + bd["hi"]) / 2
        actual_rate = bd["wins"] / bd["n"]
        cal_error = actual_rate - midpoint
        results.append({
            "bin": f"{bd['lo']*100:.0f}-{bd['hi']*100:.0f}%",
            "n_games": bd["n"],
            "predicted_midpoint": round(midpoint, 3),
            "actual_win_rate": round(actual_rate, 3),
            "calibration_error": round(cal_error, 4),
        })

    return results


def print_table(results):
    """Print ASCII reliability table."""
    print(f"\n{'Predicted':>12}  {'N_games':>7}  {'Actual_win%':>11}  {'Cal_error':>10}  Status")
    print(f"  {'-' * 62}")
    for r in results:
        cal_err = r["calibration_error"]
        if abs(cal_err) < 0.03:
            status = "good"
        elif cal_err < -0.03:
            status = "overconfident"
        else:
            status = "underconfident"
        flag = " !" if abs(cal_err) > 0.05 else ""
        print(f"  {r['bin']:>10}  {r['n_games']:>7}  {r['actual_win_rate']*100:>10.1f}%  "
              f"{cal_err*100:>+9.1f}%  {status}{flag}")


def main():
    years = []
    for a in sys.argv[1:]:
        try:
            years.append(int(a))
        except ValueError:
            pass
    if not years:
        years = [2023, 2024, 2025]

    print(f"Computing reliability diagram for years {years}...")
    results = run_reliability(years)

    if not results:
        print("No games found for the given years.")
        return

    print_table(results)

    out_path = os.path.join(DATA_DIR, "reliability_diagram.json")
    with open(out_path, "w") as f:
        json.dump({"years": years, "bins": results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary assessment
    over_bins = [r for r in results if r["calibration_error"] < -0.03]
    under_bins = [r for r in results if r["calibration_error"] > 0.03]
    large_errors = [r for r in results if abs(r["calibration_error"]) > 0.05]

    print(f"\nSummary:")
    if large_errors:
        print(f"  WARNING: {len(large_errors)} bin(s) with >5% calibration error: "
              f"{[r['bin'] for r in large_errors]}")
        if over_bins:
            print(f"  Overconfident (actual < predicted): {[r['bin'] for r in over_bins]}")
            print(f"    → Consider: inflate stdev, reduce factor bonuses, raise bet thresholds")
        if under_bins:
            print(f"  Underconfident (actual > predicted): {[r['bin'] for r in under_bins]}")
            print(f"    → Model is conservative; can bet more aggressively in these ranges")
    else:
        print(f"  All bins within 5% calibration error — model is well calibrated!")

    total_n = sum(r["n_games"] for r in results)
    avg_err = sum(abs(r["calibration_error"]) for r in results) / len(results) if results else 0
    print(f"  Total games: {total_n}  Average absolute calibration error: {avg_err*100:.1f}%")


if __name__ == "__main__":
    main()
