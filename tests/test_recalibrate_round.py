import json

from scripts import recalibrate_round


def test_run_recalibration_round_writes_summary_and_candidate_snapshot(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    docs_dir = tmp_path / "docs"
    data_dir.mkdir()
    docs_dir.mkdir()

    config_path = data_dir / "calibrated_config.json"
    config_path.write_text(json.dumps({"coach_tourney_max_bonus": 1.5, "pedigree_max_bonus": 1.0}))

    monkeypatch.setattr(recalibrate_round, "ROOT", str(tmp_path))
    monkeypatch.setattr(recalibrate_round, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(recalibrate_round, "DOCS_DIR", str(docs_dir))
    monkeypatch.setattr(
        recalibrate_round,
        "_load_pairs",
        lambda: [
            ({}, {}, 1, 5, 2023, {}),
            ({}, {}, 1, 5, 2024, {}),
            ({}, {}, 1, 5, 2025, {}),
            ({}, {}, 1, 5, 2026, {}),
        ],
    )

    snapshots = iter([
        {
            "historical_cv_brier": 0.160,
            "recent_brier": 0.170,
            "current_year_brier": 0.200,
            "current_year_games": 1,
            "coach_enabled": True,
            "pedigree_enabled": True,
        },
        {
            "historical_cv_brier": 0.162,
            "recent_brier": 0.171,
            "current_year_brier": 0.190,
            "current_year_games": 1,
            "coach_enabled": True,
            "pedigree_enabled": False,
        },
    ])
    monkeypatch.setattr(
        recalibrate_round,
        "compute_evaluation_snapshot",
        lambda pairs, path, year: next(snapshots),
    )

    def fake_run(args):
        cmd = args[1]
        if cmd.endswith("extract_results.py"):
            (data_dir / "results_2026.json").write_text(json.dumps({"year": 2026, "games": [{"team_a": "A"}]}))
        elif cmd.endswith("calibrate.py"):
            config_path.write_text(json.dumps({"coach_tourney_max_bonus": 1.5, "pedigree_max_bonus": 0.0}))
            (docs_dir / "cal_report_2026-03-19_partial.json").write_text(json.dumps({"ok": True}))

    monkeypatch.setattr(recalibrate_round, "_run_command", fake_run)

    summary = recalibrate_round.run_recalibration_round(
        year=2026,
        cutoff_date="2026-03-19",
        refresh_data=False,
    )

    assert summary["promoted"] is True
    assert (data_dir / "calibrated_config_2026-03-19.json").exists()
    assert (docs_dir / "recalibration_2026-03-19.json").exists()
