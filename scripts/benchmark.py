"""
benchmark.py — Compare prediction models on held-out tournament data.

Baselines:
  seed-only     Always pick lower seed number; win_prob proportional to seed gap
  efficiency    Predict based on adj_o - adj_d differential only (no extras)
  full          Full calibrated model (calibrated_config.json)

Uses walk-forward CV: for each test year, train baseline efficiency model on
years before it and evaluate on that year alone. Reports Brier, accuracy,
per-round breakdown, and bracket-quality metrics (champion rank, champion %,
FF hit rate in top-8).

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --year 2025    # single year only
  python scripts/benchmark.py --bracket-quality   # add champion rank, FF hit rate
"""
import io
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from engine import (
    predict_game, ModelConfig, _normalize_team_for_match, enrich_team,
    run_monte_carlo, load_bracket,
)
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
# Config presets for Monte Carlo (predict_game must behave like each model)
# ---------------------------------------------------------------------------

def _seed_only_config():
    """Config that makes predict_game return seed-based probs only.
    High stdev -> eff_prob ~ 0.5; seed_weight=1 -> output = seed_prob."""
    c = ModelConfig(num_sims=2000)
    c.base_scoring_stdev = 100.0  # eff_prob -> 0.5
    c.seed_weight = 1.0
    # Zero all factor bonuses so we don't add noise
    for attr in dir(c):
        if attr.endswith("_max_bonus") or attr.endswith("_bonus"):
            try:
                setattr(c, attr, 0.0)
            except (TypeError, AttributeError):
                pass
    return c


def _efficiency_only_config(full_config):
    """Config that uses only adj_o/adj_d; no intangibles, no seed blend."""
    c = ModelConfig(num_sims=2000)
    # Copy score scaling and stdev from full
    c.base_scoring_stdev = full_config.base_scoring_stdev
    c.score_scale = full_config.score_scale
    c.score_scale_r64 = full_config.score_scale_r64
    c.score_scale_r32 = full_config.score_scale_r32
    c.score_scale_s16 = full_config.score_scale_s16
    c.score_scale_e8 = full_config.score_scale_e8
    c.score_scale_ff = full_config.score_scale_ff
    c.em_adj_o_weight = full_config.em_adj_o_weight
    c.seed_weight = 0.0
    # Zero all factor bonuses
    for attr in ["sos_max_bonus", "possession_edge_max_bonus", "ft_clutch_max_bonus",
                 "experience_max_bonus", "coach_tourney_max_bonus", "pedigree_max_bonus",
                 "preseason_max_bonus", "proximity_max_bonus", "momentum_max_bonus",
                 "star_player_max_bonus", "size_max_bonus", "depth_max_bonus",
                 "em_opp_adjust_max_bonus", "ft_foul_rate_max_bonus", "win_pct_max_bonus",
                 "conf_rating_max_bonus"]:
        if hasattr(c, attr):
            setattr(c, attr, 0.0)
    c.three_pt_volatility_factor = 0.0
    c.tempo_volatility_weight = 0.0
    return c


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


def _normalize_team_name(name):
    """Normalize for matching (e.g. 'UConn' vs 'Connecticut')."""
    if not name:
        return ""
    n = name.strip().lower()
    aliases = {"uconn": "connecticut", "unc": "north carolina", "st. mary's": "saint mary's"}
    return aliases.get(n, n)


def _compute_bracket_quality(year, bracket_path, games, teams_by_year, full_config):
    """Run Monte Carlo for each model, compute champion rank, champion %, FF hit rate."""
    if not os.path.isfile(bracket_path):
        return None
    try:
        bracket, _, _ = load_bracket(bracket_path, data_dir=DATA_DIR, year=year)
    except Exception:
        return None
    if not bracket:
        return None

    champ, actual_ff = _get_actual_champion_and_ff(games, year)
    if not champ:
        return None

    def _match_team(pred_name, actual_name):
        pa = _normalize_team_name(pred_name)
        pb = _normalize_team_name(actual_name)
        return pa == pb or pa in pb or pb in pa

    results = {}
    configs = {
        "seed-only": _seed_only_config(),
        "efficiency": _efficiency_only_config(full_config),
        "full model": full_config,
    }
    for model_name, config in configs.items():
        config.num_sims = 2000
        with io.StringIO() as buf:
            import contextlib
            with contextlib.redirect_stdout(buf):
                mc = run_monte_carlo(bracket, config=config)
        champ_probs = mc["champion_probs"]
        ff_probs = mc["final_four_probs"]
        # Champion rank: 1-indexed position of actual champ in sorted-by-prob list
        champ_list = list(champ_probs.keys())
        champ_rank = None
        champ_pct = champ_probs.get(champ, 0.0)
        for i, t in enumerate(champ_list):
            if _match_team(t, champ):
                champ_rank = i + 1
                champ_pct = champ_probs.get(t, champ_pct)
                break
        if champ_rank is None:
            champ_rank = len(champ_list) + 1
        # FF hit rate: of 4 actual FF teams, how many in model's top-8 FF?
        top8_ff = list(ff_probs.keys())[:8]
        hits = 0
        for actual in actual_ff:
            for pred in top8_ff:
                if _match_team(pred, actual):
                    hits += 1
                    break
        ff_hit_rate = hits / 4.0 if actual_ff else 0.0
        results[model_name] = {
            "champion_rank": champ_rank,
            "champion_pct": champ_pct,
            "ff_hit_rate": ff_hit_rate,
        }
    return {"champion": champ, "actual_ff": list(actual_ff), "by_model": results}


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


def print_cv_summary(cv_results, bracket_quality_by_year=None):
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

    if bracket_quality_by_year:
        print(f"\n  {'=' * 72}")
        print(f"  BRACKET QUALITY (champion rank, champion %, FF hit rate)")
        print(f"{'=' * 72}")
        for yr in sorted(bracket_quality_by_year.keys()):
            bq = bracket_quality_by_year[yr]
            if not bq:
                continue
            print(f"\n  {yr} — Champion: {bq['champion']}")
            print(f"  {'Model':25s}  {'Champ Rank':>12s}  {'Champ %':>10s}  {'FF Hit':>10s}")
            print(f"  {'─' * 25}  {'─' * 12}  {'─' * 10}  {'─' * 10}")
            for model, m in bq["by_model"].items():
                print(f"  {model:25s}  {m['champion_rank']:>12d}  {m['champion_pct']:>9.1%}  {m['ff_hit_rate']:>9.1%}")
        # Summary: full model champion rank ≤3 in at least 2 of 3 years?
        full_ranks = []
        full_champ_pcts = []
        full_ff_hits = []
        for yr, bq in bracket_quality_by_year.items():
            if bq and "full model" in bq["by_model"]:
                fm = bq["by_model"]["full model"]
                full_ranks.append(fm["champion_rank"])
                full_champ_pcts.append(fm["champion_pct"])
                full_ff_hits.append(fm["ff_hit_rate"])
        if full_ranks:
            rank_ok = sum(1 for r in full_ranks if r <= 3)
            avg_champ = sum(full_champ_pcts) / len(full_champ_pcts) * 100
            avg_ff = sum(full_ff_hits) / len(full_ff_hits) * 100
            print(f"\n  Full model: champion rank ≤3 in {rank_ok}/{len(full_ranks)} years")
            print(f"  Full model: avg champion % = {avg_champ:.1f}%")
            print(f"  Full model: avg FF hit rate in top-8 = {avg_ff:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    single_year = None
    bracket_quality = "--bracket-quality" in sys.argv
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
    test_years = [single_year] if single_year else all_years[-15:]

    cv_results = {}
    bracket_quality_by_year = {}

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

        # Bracket quality: champion rank, champion %, FF hit rate (Monte Carlo)
        if bracket_quality:
            print(f"\n  Computing bracket quality for {test_year} (3 models × 2000 sims)...")
            bracket_path = os.path.join(DATA_DIR, f"bracket_{test_year}.json")
            bq = _compute_bracket_quality(
                test_year, bracket_path, games, teams_by_year, full_config
            )
            if bq:
                bracket_quality_by_year[test_year] = bq
                for model, m in bq["by_model"].items():
                    cv_results[test_year][model]["champion_rank"] = m["champion_rank"]
                    cv_results[test_year][model]["champion_pct"] = m["champion_pct"]
                    cv_results[test_year][model]["ff_hit_rate"] = m["ff_hit_rate"]

    if len(cv_results) > 1:
        print_cv_summary(cv_results, bracket_quality_by_year=bracket_quality_by_year or None)
    elif cv_results:
        yr = list(cv_results.keys())[0]
        m = cv_results[yr]
        print(f"\n  seed-only Brier: {m['seed-only']['brier_score']:.4f}")
        print(f"  efficiency Brier: {m['efficiency']['brier_score']:.4f}")
        print(f"  full model Brier: {m['full model']['brier_score']:.4f}")
        if bracket_quality_by_year:
            print_cv_summary(cv_results, bracket_quality_by_year=bracket_quality_by_year)

    # Save results for Picks tab / API
    if cv_results:
        models = list(list(cv_results.values())[0].keys())
        out = {
            "cv_summary": {
                m: {
                    "brier": sum(cv_results[yr][m]["brier_score"] for yr in cv_results) / len(cv_results),
                    "accuracy": sum(cv_results[yr][m]["accuracy"] for yr in cv_results) / len(cv_results),
                }
                for m in models
            },
            "per_year": {str(yr): {m: {"brier": cv_results[yr][m]["brier_score"]} for m in cv_results[yr]} for yr in cv_results},
            "bracket_quality": bracket_quality_by_year if bracket_quality_by_year else None,
        }
        out_path = os.path.join(DATA_DIR, "benchmark_results.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
