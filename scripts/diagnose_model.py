"""
Diagnose model failure modes: per-game Brier, grouped breakdowns, top worst predictions.

Usage:
  python scripts/diagnose_model.py 2017 2019 2021 2023 2024
  python scripts/diagnose_model.py  # uses all available years
"""
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from engine import predict_game, enrich_team, DEFAULT_CONFIG
from scripts.calibrate import (
    load_results,
    load_all_teams,
    build_game_pairs,
    _pairs_by_year,
)


def diagnose(pairs, config=None):
    """Run predict_game on all pairs, collect per-game records and aggregates."""
    config = config or DEFAULT_CONFIG
    records = []
    brier_sum = 0.0
    n = 0

    for a, b, a_won, actual_margin, yr, g in pairs:
        rname = g.get("round_name", "Unknown")
        try:
            result = predict_game(enrich_team(a), enrich_team(b), config=config, round_name=rname)
        except Exception:
            continue
        prob_a = max(0.001, min(0.999, result["win_prob_a"]))
        pred_margin = result["predicted_margin"]
        brier_contrib = (prob_a - a_won) ** 2
        upset_bonus = result.get("upset_tolerance_bonus", 0.0)

        seed_a = a.get("seed", 8)
        seed_b = b.get("seed", 8)
        is_upset = g.get("upset", (a_won == 1 and seed_a > seed_b) or (a_won == 0 and seed_b > seed_a))
        seed_pair = (min(seed_a, seed_b), max(seed_a, seed_b))

        records.append({
            "year": yr,
            "round": rname,
            "team_a": g["team_a"],
            "team_b": g["team_b"],
            "seed_a": seed_a,
            "seed_b": seed_b,
            "seed_pair": seed_pair,
            "a_won": a_won,
            "prob_a": prob_a,
            "pred_margin": pred_margin,
            "actual_margin": actual_margin,
            "brier_contrib": brier_contrib,
            "upset": is_upset,
            "upset_bonus": upset_bonus,
            "correct": (prob_a >= 0.5) == bool(a_won),
        })
        brier_sum += brier_contrib
        n += 1

    return records, brier_sum, n


def bucket_closeness(pred_margin):
    if abs(pred_margin) < 2:
        return "|margin|<2"
    if abs(pred_margin) < 5:
        return "2-5"
    if abs(pred_margin) < 10:
        return "5-10"
    return "10+"


def main():
    years = []
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            try:
                years.append(int(a))
            except ValueError:
                pass

    print("Loading historical results...")
    games = load_results()
    if years:
        games = [g for g in games if g.get("year") in years]
        print(f"  Filtered to years {years}: {len(games)} games")
    else:
        print(f"  All years: {len(games)} games")

    print("Loading team stats...")
    teams_by_year = load_all_teams()
    print(f"  {len(teams_by_year)} years of team data")

    print("Building game pairs...")
    pairs = build_game_pairs(games, teams_by_year)
    print(f"  {len(pairs)} matchable games")

    records, brier_sum, n = diagnose(pairs)
    if not records:
        print("No games to diagnose.")
        return

    total_brier = brier_sum / n
    print(f"\n{'='*60}")
    print("  DIAGNOSTIC REPORT")
    print(f"{'='*60}")
    print(f"  Total games: {n}")
    print(f"  Overall Brier: {total_brier:.4f}")
    print(f"  Accuracy: {sum(1 for r in records if r['correct'])/n:.1%}")

    # By closeness
    by_close = {}
    for r in records:
        b = bucket_closeness(r["pred_margin"])
        by_close.setdefault(b, {"brier": 0.0, "n": 0})
        by_close[b]["brier"] += r["brier_contrib"]
        by_close[b]["n"] += 1
    print(f"\n  By predicted margin closeness:")
    for b in ["|margin|<2", "2-5", "5-10", "10+"]:
        if b in by_close:
            d = by_close[b]
            pct_brier = 100 * d["brier"] / brier_sum if brier_sum else 0
            avg_brier = d["brier"] / d["n"] if d["n"] else 0
            print(f"    {b:12s}: n={d['n']:4d}  Brier={avg_brier:.4f}  ({pct_brier:.1f}% of total)")

    # By upset status
    upset_records = [r for r in records if r["upset"]]
    chalk_records = [r for r in records if not r["upset"]]
    brier_upset = sum(r["brier_contrib"] for r in upset_records)
    brier_chalk = sum(r["brier_contrib"] for r in chalk_records)
    n_upset = len(upset_records)
    n_chalk = len(chalk_records)
    print(f"\n  By upset status:")
    print(f"    Upsets (lower seed won): n={n_upset}  Brier={brier_upset/n_upset:.4f}  ({100*brier_upset/brier_sum:.1f}% of total)" if n_upset else "    Upsets: 0 games")
    print(f"    Chalk (higher seed won): n={n_chalk}  Brier={brier_chalk/n_chalk:.4f}  ({100*brier_chalk/brier_sum:.1f}% of total)" if n_chalk else "    Chalk: 0 games")

    # By round
    by_round = {}
    for r in records:
        rn = r["round"]
        by_round.setdefault(rn, {"brier": 0.0, "n": 0, "correct": 0})
        by_round[rn]["brier"] += r["brier_contrib"]
        by_round[rn]["n"] += 1
        if r["correct"]:
            by_round[rn]["correct"] += 1
    print(f"\n  By round:")
    for rn in ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]:
        if rn in by_round:
            d = by_round[rn]
            acc = d["correct"] / d["n"] if d["n"] else 0
            avg_brier = d["brier"] / d["n"] if d["n"] else 0
            print(f"    {rn:16s}: n={d['n']:3d}  acc={acc:.1%}  Brier={avg_brier:.4f}")

    # By seed pair (top 10 by game count)
    by_seed = {}
    for r in records:
        sp = r["seed_pair"]
        by_seed.setdefault(sp, {"brier": 0.0, "n": 0})
        by_seed[sp]["brier"] += r["brier_contrib"]
        by_seed[sp]["n"] += 1
    top_seeds = sorted(by_seed.items(), key=lambda x: -x[1]["n"])[:10]
    print(f"\n  By seed pair (top 10 by count):")
    for (lo, hi), d in top_seeds:
        avg_brier = d["brier"] / d["n"] if d["n"] else 0
        print(f"    ({lo})v({hi}): n={d['n']:3d}  Brier={avg_brier:.4f}")

    # Top 20 worst games
    worst = sorted(records, key=lambda r: -r["brier_contrib"])[:20]
    print(f"\n  Top 20 worst predictions (by Brier contribution):")
    for i, r in enumerate(worst, 1):
        pred = "A" if r["prob_a"] >= 0.5 else "B"
        actual = "A" if r["a_won"] else "B"
        bonus = f"  bonus={r['upset_bonus']:+.2f}" if r["upset_bonus"] else ""
        print(f"    {i:2d}. {r['year']} {r['round']:12s} ({r['seed_a']}){r['team_a'][:20]:20s} vs ({r['seed_b']}){r['team_b'][:20]:20s}")
        print(f"        prob_A={r['prob_a']:.3f}  pred={pred} actual={actual}  Brier={r['brier_contrib']:.4f}{bonus}")


if __name__ == "__main__":
    main()
