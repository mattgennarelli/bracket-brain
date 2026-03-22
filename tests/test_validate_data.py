import json

from scripts import validate_data


def _write_fixture(tmp_path, teams):
    bracket = {
        "regions": {
            "East": [
                {"team": "Duke", "seed": 1},
                {"team": "Houston", "seed": 2},
            ]
        },
        "first_four": [
            {"team_a": "Long Island", "seed_a": 16, "team_b": "NC State", "seed_b": 16}
        ],
    }
    (tmp_path / "bracket_2026.json").write_text(json.dumps(bracket))
    (tmp_path / "teams_merged_2026.json").write_text(json.dumps(teams))


def test_validate_year_accepts_rich_team_file(monkeypatch, tmp_path):
    teams = [
        {
            "team": "Duke",
            "adj_o": 120.0,
            "barthag": 0.98,
            "wins": 32,
            "games": 34,
            "ppg": 80.0,
            "opp_ppg": 62.0,
            "three_pt_pct": 36.0,
            "experience": 0.6,
            "top_player": "Star A",
            "top_player_bpr": 10.0,
            "em_bpr": 30.0,
        },
        {
            "team": "Houston",
            "adj_o": 118.0,
            "barthag": 0.96,
            "wins": 30,
            "games": 34,
            "ppg": 77.0,
            "opp_ppg": 59.0,
            "three_pt_pct": 35.0,
            "experience": 0.7,
            "top_player": "Star B",
            "top_player_bpr": 9.0,
            "em_bpr": 28.0,
        },
        {
            "team": "Long Island",
            "adj_o": 105.0,
            "barthag": 0.70,
            "wins": 20,
            "games": 32,
            "ppg": 71.0,
            "opp_ppg": 68.0,
            "three_pt_pct": 33.0,
            "experience": 0.2,
            "top_player": "Star C",
            "top_player_bpr": 4.0,
            "em_bpr": 12.0,
        },
        {
            "team": "NC State",
            "adj_o": 107.0,
            "barthag": 0.72,
            "wins": 21,
            "games": 33,
            "ppg": 72.0,
            "opp_ppg": 69.0,
            "three_pt_pct": 34.0,
            "experience": 0.3,
            "top_player": "Star D",
            "top_player_bpr": 4.5,
            "em_bpr": 13.0,
        },
    ]
    _write_fixture(tmp_path, teams)
    monkeypatch.setattr(validate_data, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(validate_data, "_RICHNESS_MIN_TEAM_COUNT", 4)
    monkeypatch.setattr(
        validate_data,
        "_RICHNESS_MIN_FIELD_COUNTS",
        {field: 4 for field in validate_data._RICHNESS_MIN_FIELD_COUNTS},
    )
    monkeypatch.setattr(
        validate_data,
        "_RICHNESS_MAX_MISSING_BRACKET_TEAMS",
        {field: 0 for field in validate_data._RICHNESS_MAX_MISSING_BRACKET_TEAMS},
    )

    ok, issues, matched, total = validate_data.validate_year(2026)

    assert ok is True
    assert issues == []
    assert matched == 4
    assert total == 4


def test_validate_year_rejects_stripped_team_file(monkeypatch, tmp_path):
    teams = [
        {"team": "Duke", "adj_o": 120.0, "barthag": 0.98},
        {"team": "Houston", "adj_o": 118.0, "barthag": 0.96},
        {"team": "Long Island", "adj_o": 105.0, "barthag": 0.70},
        {"team": "NC State", "adj_o": 107.0, "barthag": 0.72},
    ]
    _write_fixture(tmp_path, teams)
    monkeypatch.setattr(validate_data, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(validate_data, "_RICHNESS_MIN_TEAM_COUNT", 4)
    monkeypatch.setattr(
        validate_data,
        "_RICHNESS_MIN_FIELD_COUNTS",
        {field: 4 for field in validate_data._RICHNESS_MIN_FIELD_COUNTS},
    )
    monkeypatch.setattr(
        validate_data,
        "_RICHNESS_MAX_MISSING_BRACKET_TEAMS",
        {field: 0 for field in validate_data._RICHNESS_MAX_MISSING_BRACKET_TEAMS},
    )

    ok, issues, matched, total = validate_data.validate_year(2026)

    assert ok is False
    assert matched == 4
    assert total == 4
    assert any("FIELD COVERAGE: wins" in issue for issue in issues)
    assert any("TOURNAMENT FIELD: wins missing" in issue for issue in issues)
