"""
Backtest harness for Bracket Brain.

Generates bracket picks for historical years and scores them against actual results.
Reports per-game accuracy, Brier score, log loss, and per-round breakdown.

Usage:
  python backtest.py 2024                     # single year
  python backtest.py 2017 2019 2021 2023 2024 # multiple years
  python backtest.py --chaos 0.1 2024        # use 10% upset aggression
  python backtest.py --sweep-chaos 2017 2019 2021 2023 2024 2025  # sweep chaos levels
  python backtest.py                          # defaults
"""

import json
import math
import os
import sys
from typing import List, Optional, Tuple

from engine import (ModelConfig, load_bracket, generate_bracket_picks,
                    predict_game, enrich_team, _normalize_team_for_match,
                    REGIONS, FIRST_ROUND_MATCHUPS, DEFAULT_CONFIG)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

CHAOS_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def parse_args():
    """Parse --chaos, --sweep-chaos and year args. Returns (chaos, sweep_chaos, years)."""
    args = sys.argv[1:]
    chaos = None
    sweep_chaos = False
    years = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--chaos":
            if i + 1 < len(args):
                try:
                    chaos = float(args[i + 1])
                    i += 2
                    continue
                except ValueError:
                    pass
            i += 1
            continue
        if a == "--sweep-chaos":
            sweep_chaos = True
            i += 1
            continue
        if not a.startswith("-"):
            try:
                years.append(int(a))
            except ValueError:
                pass
        i += 1
    if not years:
        years = [2017, 2019, 2021, 2023, 2024]
    return chaos, sweep_chaos, years


def years_from_args() -> List[int]:
    years: List[int] = []
    for a in sys.argv[1:]:
        if a.startswith("-"):
            continue
        try:
            years.append(int(a))
        except ValueError:
            continue
    if not years:
        years = [2017, 2019, 2021, 2023, 2024]
    return years


def load_results(year):
    """Load actual results for a year."""
    path = os.path.join(DATA_DIR, f"results_{year}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("games", [])


def load_teams_for_year(year):
    """Load teams_merged for stat lookups."""
    for fname in (f"teams_merged_{year}.json", f"torvik_{year}.json"):
        path = os.path.join(DATA_DIR, fname)
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                out = {}
                for r in data:
                    t = r.get("team", "")
                    if t and not t.startswith("Unknown"):
                        key = _normalize_team_for_match(t)
                        if key:
                            out[key] = dict(r)
                return out
    return {}


def _lookup(name, seed, teams):
    """Look up team stats."""
    key = _normalize_team_for_match(name)
    if key in teams:
        t = dict(teams[key])
        t["seed"] = seed
        t["team"] = name
        return t
    for k, v in teams.items():
        if key in k or k in key:
            t = dict(v)
            t["seed"] = seed
            t["team"] = name
            return t
    return {"team": name, "seed": seed, "adj_o": 85, "adj_d": 112,
            "adj_tempo": 64, "barthag": 0.05}


def score_bracket_picks(picks, actual_games):
    """Score the bracket's picks against actual results. Only scores first-round games
    (which are known matchups) for fair evaluation."""
    r64_actuals = {}
    for g in actual_games:
        if g["round"] == 64:
            key_a = _normalize_team_for_match(g["team_a"])
            key_b = _normalize_team_for_match(g["team_b"])
            r64_actuals[(key_a, key_b)] = g
            r64_actuals[(key_b, key_a)] = g

    scored = []
    for pick in picks:
        if pick["round"] != 64:
            continue
        key_a = _normalize_team_for_match(pick["team_a"])
        key_b = _normalize_team_for_match(pick["team_b"])
        actual = r64_actuals.get((key_a, key_b)) or r64_actuals.get((key_b, key_a))
        if not actual:
            continue
        actual_winner = actual["winner"]
        pick_correct = (_normalize_team_for_match(pick["pick"]) ==
                       _normalize_team_for_match(actual_winner))
        scored.append({
            "pick": pick,
            "actual_winner": actual_winner,
            "correct": pick_correct,
            "actual_margin": actual["margin"],
        })
    return scored


def score_all_games(actual_games, teams, config):
    """Score predict_game on every actual game (all rounds). Returns metrics."""
    brier_sum = 0.0
    log_loss_sum = 0.0
    correct = 0
    n = 0
    round_stats = {}

    for g in actual_games:
        a = _lookup(g["team_a"], g["seed_a"], teams)
        b = _lookup(g["team_b"], g["seed_b"], teams)
        try:
            result = predict_game(enrich_team(a), enrich_team(b), config=config)
        except Exception:
            continue
        prob_a = max(0.001, min(0.999, result["win_prob_a"]))
        a_won = 1 if g["winner"] == g["team_a"] else 0

        brier_sum += (prob_a - a_won) ** 2
        log_loss_sum -= (a_won * math.log(prob_a) + (1 - a_won) * math.log(1 - prob_a))
        if (prob_a >= 0.5) == bool(a_won):
            correct += 1
        n += 1

        rname = g.get("round_name", f"Round of {g['round']}")
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


def run_year(year: int, config=DEFAULT_CONFIG, upset_aggression: float = 0.0,
             verbose: bool = True) -> Optional[dict]:
    bracket_path = os.path.join(DATA_DIR, f"bracket_{year}.json")
    if not os.path.isfile(bracket_path):
        if verbose:
            print(f"[{year}] Skipping — {bracket_path} not found.")
        return None

    actual_games = load_results(year)
    if not actual_games:
        if verbose:
            print(f"[{year}] Skipping — no results file. Run: python scripts/extract_results.py {year}")
        return None

    teams = load_teams_for_year(year)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  [{year}] BACKTEST (chaos={upset_aggression:.0%})")
        print(f"{'='*60}")

    bracket, _, quadrant_order = load_bracket(bracket_path, data_dir=DATA_DIR, year=year)

    # Generate bracket picks
    bracket_result = generate_bracket_picks(bracket, config, upset_aggression=upset_aggression,
                                            quadrant_order=quadrant_order,
                                            data_dir=DATA_DIR, year=year)
    picks = bracket_result["picks"]
    if verbose:
        print(f"  Champion pick: {bracket_result['champion']}")
        print(f"  Final Four: {', '.join(bracket_result['final_four'])}")

    # Score R64 bracket picks against actuals
    r64_scored = score_bracket_picks(picks, actual_games)
    r64_correct = sum(1 for s in r64_scored if s["correct"]) if r64_scored else 0
    r64_total = len(r64_scored)
    if verbose and r64_scored:
        print(f"\n  First-round picks: {r64_correct}/{r64_total} ({r64_correct/r64_total:.0%})")
        wrong = [s for s in r64_scored if not s["correct"]]
        if wrong:
            print(f"  Missed ({len(wrong)}):")
            for s in wrong:
                p = s["pick"]
                print(f"    ({p['seed_a']}) {p['team_a']} vs ({p['seed_b']}) {p['team_b']}: "
                      f"picked {p['pick']}, actual {s['actual_winner']} (spread={p['projected_spread']})")

    # Score all games with predict_game
    metrics = score_all_games(actual_games, teams, config)
    metrics["r64_correct"] = r64_correct
    metrics["r64_total"] = r64_total
    if verbose:
        print(f"\n  All-game scoring ({metrics['n_games']} games):")
        print(f"    Accuracy:    {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['n_games']})")
        print(f"    Brier score: {metrics['brier_score']:.4f}")
        print(f"    Log loss:    {metrics['log_loss']:.4f}")
        print(f"\n    Per-round:")
        for rname in ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]:
            rs = metrics["round_stats"].get(rname)
            if not rs:
                continue
            acc = rs["correct"] / rs["total"] if rs["total"] else 0
            print(f"      {rname:16s}: {acc:.0%} ({rs['correct']}/{rs['total']})")

        # Check champion
        actual_champ_game = [g for g in actual_games if g["round"] == 2]
        if actual_champ_game:
            actual_champ = actual_champ_game[0]["winner"]
            hit = "CORRECT" if _normalize_team_for_match(bracket_result["champion"] or "") == _normalize_team_for_match(actual_champ) else "WRONG"
            print(f"\n  Champion: picked {bracket_result['champion']}, actual {actual_champ} [{hit}]")

    # Save results
    out = {
        "year": year,
        "champion_pick": bracket_result["champion"],
        "final_four": bracket_result["final_four"],
        "picks": picks,
        "metrics": {k: v for k, v in metrics.items() if k != "round_stats"},
        "round_metrics": {k: {"accuracy": v["correct"]/v["total"] if v["total"] else 0,
                              "correct": v["correct"], "total": v["total"]}
                         for k, v in metrics["round_stats"].items()},
    }
    out_path = os.path.join(DATA_DIR, f"backtest_{year}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    if verbose:
        print(f"\n  Wrote {out_path}")
    return metrics


def run_sweep_chaos(years: List[int], config=DEFAULT_CONFIG) -> List[Tuple[float, int, int, float, float]]:
    """Run backtest for each chaos level. Returns [(chaos, r64_correct, r64_total, all_acc, brier), ...]."""
    results = []
    for chaos in CHAOS_LEVELS:
        r64_correct = 0
        r64_total = 0
        all_correct = 0
        all_games = 0
        brier_sum = 0.0
        n_years = 0
        for y in years:
            m = run_year(y, config=config, upset_aggression=chaos, verbose=False)
            if m:
                r64_correct += m.get("r64_correct", 0)
                r64_total += m.get("r64_total", 0)
                all_correct += m.get("correct", 0)
                all_games += m.get("n_games", 0)
                brier_sum += m.get("brier_score", 0)
                n_years += 1
        all_acc = all_correct / all_games if all_games else 0
        avg_brier = brier_sum / n_years if n_years else 0
        results.append((chaos, r64_correct, r64_total, all_acc, avg_brier))
    return results


def main():
    chaos, sweep_chaos, years = parse_args()

    if sweep_chaos:
        print(f"\nSweeping chaos levels over years {years}...")
        results = run_sweep_chaos(years)
        print(f"\n{'Chaos':>8}  {'R64_correct':>10}  {'R64_total':>9}  {'All_game_acc':>12}  {'Brier':>8}")
        print("-" * 55)
        for chaos_val, r64_c, r64_t, acc, brier in results:
            print(f"  {chaos_val*100:>5.0f}%  {r64_c:>10}  {r64_t:>9}  {acc:>11.1%}  {brier:>8.4f}")
        best = max(results, key=lambda r: r[1])
        print(f"\nBest R64: {best[1]}/{best[2]} at {best[0]*100:.0f}% chaos")
        return

    upset_aggression = chaos if chaos is not None else 0.0
    all_metrics = []
    for y in years:
        m = run_year(y, upset_aggression=upset_aggression)
        if m:
            all_metrics.append((y, m))

    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print(f"  AGGREGATE ({len(all_metrics)} years)")
        print(f"{'='*60}")
        total_correct = sum(m["correct"] for _, m in all_metrics)
        total_games = sum(m["n_games"] for _, m in all_metrics)
        avg_brier = sum(m["brier_score"] for _, m in all_metrics) / len(all_metrics)
        avg_logloss = sum(m["log_loss"] for _, m in all_metrics) / len(all_metrics)
        print(f"  Total accuracy: {total_correct}/{total_games} ({total_correct/total_games:.1%})")
        print(f"  Avg Brier:      {avg_brier:.4f}")
        print(f"  Avg Log Loss:   {avg_logloss:.4f}")
        print()
        for y, m in all_metrics:
            print(f"  {y}: {m['accuracy']:.1%} ({m['correct']}/{m['n_games']})  Brier={m['brier_score']:.4f}")


if __name__ == "__main__":
    main()
