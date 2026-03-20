from dataclasses import replace

from engine import DEFAULT_CONFIG
from scripts import calibrate


def test_evaluate_prior_gate_keeps_only_helpful_prior(monkeypatch):
    pairs = [
        ({}, {}, 1, 5, 2022, {}),
        ({}, {}, 1, 5, 2023, {}),
        ({}, {}, 1, 5, 2024, {}),
        ({}, {}, 1, 5, 2025, {}),
    ]

    def fake_score_model(scored_pairs, config, recency_weight=None, round_weight=None):
        years = {p[4] for p in scored_pairs}
        is_recent = years.issubset({2023, 2024, 2025}) and years
        brier = 0.1600 if is_recent else 0.1650
        if config.coach_tourney_max_bonus > 0:
            brier -= 0.0020
        if config.pedigree_max_bonus > 0:
            brier += 0.0015
        return {"brier_score": brier, "accuracy": 0.75}

    monkeypatch.setattr(calibrate, "score_model", fake_score_model)

    config = replace(DEFAULT_CONFIG, num_sims=1, coach_tourney_max_bonus=1.5, pedigree_max_bonus=2.0)
    report, keep_coach, keep_pedigree, gated = calibrate.evaluate_prior_gate(pairs, config)

    assert report["base"]["all"]["brier_score"] < report["coach_zero"]["all"]["brier_score"]
    assert report["base"]["recent"]["brier_score"] < report["coach_zero"]["recent"]["brier_score"]
    assert keep_coach is True
    assert keep_pedigree is False
    assert gated.coach_tourney_max_bonus == 1.5
    assert gated.pedigree_max_bonus == 0.0


def test_build_partial_year_weight_overrides_caps_effective_weight():
    cases = {
        4: 16.0,
        8: 16.0,
        16: 32.0,
        24: 32.0,
    }
    for n_games, expected_total in cases.items():
        pairs = [({}, {}, 1, 5, 2026, {}) for _ in range(n_games)]
        overrides = calibrate.build_partial_year_weight_overrides(pairs)
        assert 2026 in overrides
        assert round(overrides[2026] * n_games, 6) == expected_total


def test_main_runs_prior_gate_for_full_calibration(monkeypatch, tmp_path):
    pairs = [({}, {}, 1, 5, 2025, {"round_name": "Round of 64"})]
    metrics = {
        "brier_score": 0.16,
        "accuracy": 0.75,
        "n_games": 1,
        "correct": 1,
        "round_stats": {},
        "log_loss": 0.4,
        "spread_mae": 1.0,
        "total_bias": 0.0,
        "total_mae": 0.0,
    }

    monkeypatch.setattr(calibrate, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(calibrate, "load_results", lambda: [{"year": 2025}])
    monkeypatch.setattr(calibrate, "load_all_teams", lambda: {})
    monkeypatch.setattr(calibrate, "build_game_pairs", lambda games, teams: pairs)
    monkeypatch.setattr(calibrate, "score_model", lambda *args, **kwargs: dict(metrics))
    monkeypatch.setattr(
        calibrate,
        "calibrate_walk_forward",
        lambda *args, **kwargs: ({}, replace(DEFAULT_CONFIG, num_sims=1)),
    )
    monkeypatch.setattr(calibrate, "print_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(calibrate, "print_prior_gate_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(calibrate, "save_config", lambda optimized: None)

    called = {}

    def fake_apply_prior_gate(scored_pairs, config, optimized=None, recent_years=None, enabled=True):
        called["ran"] = True
        gated = replace(config, pedigree_max_bonus=0.0)
        updated = dict(optimized or {})
        updated["coach_tourney_max_bonus"] = round(float(gated.coach_tourney_max_bonus), 4)
        updated["pedigree_max_bonus"] = 0.0
        return {"base": {}}, True, False, gated, updated

    monkeypatch.setattr(calibrate, "apply_prior_gate", fake_apply_prior_gate)
    monkeypatch.setattr(calibrate.sys, "argv", ["calibrate.py"])

    calibrate.main()

    assert called["ran"] is True
