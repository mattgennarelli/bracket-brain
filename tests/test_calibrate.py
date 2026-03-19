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
