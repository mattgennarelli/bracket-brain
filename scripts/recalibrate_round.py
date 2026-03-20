"""
Run a post-results tournament recalibration cycle and promote the new config only
if it clears historical/recent/current-year guardrails.

Usage:
  python3 scripts/recalibrate_round.py --cutoff-date 2026-03-19
  python3 scripts/recalibrate_round.py --cutoff-date 2026-03-19 --skip-refresh
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
DOCS_DIR = os.path.join(ROOT, "docs")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)

from engine import DEFAULT_CONFIG
import calibrate

HISTORICAL_CV_MAX_WORSEN = 0.003
RECENT_BRIER_MAX_WORSEN = 0.005


def _run_command(args):
    print("+", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def _load_json(path, default=None):
    if not os.path.isfile(path):
        return {} if default is None else default
    with open(path) as f:
        return json.load(f)


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_model_config(config_path):
    config = replace(DEFAULT_CONFIG, num_sims=1)
    data = _load_json(config_path, default={})
    if isinstance(data, dict):
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return config


def _load_pairs():
    games = calibrate.load_results()
    teams_by_year = calibrate.load_all_teams()
    return calibrate.build_game_pairs(games, teams_by_year)


def _historical_cv_years(pairs, year):
    pairs_by_year = calibrate._pairs_by_year(pairs)
    partial_years = set(calibrate.partial_years_from_pairs(pairs))
    return [yr for yr in sorted(pairs_by_year) if yr >= 2017 and yr < year and yr not in partial_years]


def _cv_brier(pairs, config, years):
    pairs_by_year = calibrate._pairs_by_year(pairs)
    briers = []
    for test_year in years:
        _, test_pairs = calibrate._fold_pairs(pairs_by_year, test_year)
        if test_pairs:
            briers.append(calibrate.score_model(test_pairs, config)["brier_score"])
    return sum(briers) / len(briers) if briers else None


def _brier_for_years(pairs, config, years):
    year_set = set(years)
    slice_pairs = [p for p in pairs if p[4] in year_set]
    if not slice_pairs:
        return None
    return calibrate.score_model(slice_pairs, config)["brier_score"]


def compute_evaluation_snapshot(pairs, config_path, year):
    config = _load_model_config(config_path)
    counts = {}
    for _, _, _, _, yr, _ in pairs:
        counts[yr] = counts.get(yr, 0) + 1
    recent_years = [yr for yr in (2023, 2024, 2025) if yr in counts]
    current_year_pairs = [p for p in pairs if p[4] == year]
    return {
        "historical_cv_brier": _cv_brier(pairs, config, _historical_cv_years(pairs, year)),
        "recent_brier": _brier_for_years(pairs, config, recent_years),
        "current_year_brier": _brier_for_years(pairs, config, [year]),
        "current_year_games": len(current_year_pairs),
        "coach_enabled": getattr(config, "coach_tourney_max_bonus", 0.0) > 0,
        "pedigree_enabled": getattr(config, "pedigree_max_bonus", 0.0) > 0,
    }


def should_promote(previous_metrics, candidate_metrics):
    checks = {
        "historical_cv_ok": (
            previous_metrics["historical_cv_brier"] is None
            or candidate_metrics["historical_cv_brier"] is None
            or candidate_metrics["historical_cv_brier"] <= previous_metrics["historical_cv_brier"] + HISTORICAL_CV_MAX_WORSEN
        ),
        "recent_brier_ok": (
            previous_metrics["recent_brier"] is None
            or candidate_metrics["recent_brier"] is None
            or candidate_metrics["recent_brier"] <= previous_metrics["recent_brier"] + RECENT_BRIER_MAX_WORSEN
        ),
        "current_year_ok": (
            previous_metrics["current_year_brier"] is None
            or candidate_metrics["current_year_brier"] is None
            or candidate_metrics["current_year_brier"] <= previous_metrics["current_year_brier"]
        ),
    }
    return all(checks.values()), checks


def run_recalibration_round(year, cutoff_date, refresh_data=True, objective="brier", maxiter=100, popsize=16, multi_start=2):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    py = sys.executable
    config_path = os.path.join(DATA_DIR, "calibrated_config.json")
    pre_snapshot_path = os.path.join(DATA_DIR, f"calibrated_config_pre_{cutoff_date}.json")
    candidate_snapshot_path = os.path.join(DATA_DIR, f"calibrated_config_{cutoff_date}.json")
    report_path = os.path.join(DOCS_DIR, f"cal_report_{cutoff_date}_partial.json")
    summary_path = os.path.join(DOCS_DIR, f"recalibration_{cutoff_date}.json")

    had_existing_config = os.path.isfile(config_path)
    if had_existing_config:
        shutil.copyfile(config_path, pre_snapshot_path)

    if refresh_data:
        _run_command([py, os.path.join("scripts", "fetch_torvik.py"), str(year)])
        _run_command([py, os.path.join("scripts", "fetch_evanmiya.py"), str(year)])
        _run_command([py, os.path.join("scripts", "fetch_data.py"), "--no-fetch", str(year)])

    _run_command([
        py,
        os.path.join("scripts", "extract_results.py"),
        "--allow-partial",
        "--cutoff-date",
        cutoff_date,
        str(year),
    ])

    pairs = _load_pairs()
    previous_metrics = compute_evaluation_snapshot(
        pairs,
        pre_snapshot_path if had_existing_config else config_path,
        year,
    )

    _run_command([
        py,
        os.path.join("scripts", "calibrate.py"),
        "--objective",
        objective,
        "--maxiter",
        str(maxiter),
        "--popsize",
        str(popsize),
        "--multi-start",
        str(multi_start),
        "--save-report",
        report_path,
    ])

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Calibration did not produce {config_path}")
    shutil.copyfile(config_path, candidate_snapshot_path)

    candidate_metrics = compute_evaluation_snapshot(pairs, config_path, year)
    promoted, checks = should_promote(previous_metrics, candidate_metrics)

    if not promoted and had_existing_config:
        shutil.copyfile(pre_snapshot_path, config_path)

    summary = {
        "year": year,
        "cutoff_date": cutoff_date,
        "refresh_data": refresh_data,
        "report_path": report_path,
        "pre_snapshot_path": pre_snapshot_path if had_existing_config else None,
        "candidate_snapshot_path": candidate_snapshot_path,
        "promoted": promoted,
        "checks": checks,
        "previous_metrics": previous_metrics,
        "candidate_metrics": candidate_metrics,
    }
    _save_json(summary_path, summary)

    status = "PROMOTED" if promoted else "REVERTED"
    print(f"\nRecalibration {status}")
    print(f"  Historical CV Brier: {previous_metrics['historical_cv_brier']} -> {candidate_metrics['historical_cv_brier']}")
    print(f"  Recent Brier:        {previous_metrics['recent_brier']} -> {candidate_metrics['recent_brier']}")
    print(f"  {year} Brier:           {previous_metrics['current_year_brier']} -> {candidate_metrics['current_year_brier']}")
    print(f"  Summary: {summary_path}")
    return summary


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run a dated post-round recalibration flow.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--cutoff-date", required=True, help="Include tournament results through YYYY-MM-DD.")
    parser.add_argument("--skip-refresh", action="store_true", help="Skip Torvik/EvanMiya refresh and teams rebuild.")
    parser.add_argument("--objective", default="brier")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--multi-start", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    run_recalibration_round(
        year=args.year,
        cutoff_date=args.cutoff_date,
        refresh_data=not args.skip_refresh,
        objective=args.objective,
        maxiter=args.maxiter,
        popsize=args.popsize,
        multi_start=args.multi_start,
    )


if __name__ == "__main__":
    main()
