from scripts import extract_results


def test_extract_partial_results_from_records_accepts_small_slate(monkeypatch):
    context = {
        "team_map": {
            "alpha": "Alpha",
            "beta": "Beta",
            "michigan": "Michigan",
            "howard": "Howard",
            "duke": "Duke",
            "vermont": "Vermont",
            "florida": "Florida",
            "yale": "Yale",
        },
        "team_meta": {
            "Alpha": {"seed": 6, "region": "East"},
            "Beta": {"seed": 11, "region": "East"},
            "Michigan": {"seed": 2, "region": "South"},
            "Howard": {"seed": 15, "region": "South"},
            "Duke": {"seed": 1, "region": "Midwest"},
            "Vermont": {"seed": 16, "region": "Midwest"},
            "Florida": {"seed": 3, "region": "West"},
            "Yale": {"seed": 14, "region": "West"},
        },
        "matchup_info": {
            ("alpha", "beta"): {"round": 64, "region": "East"},
            ("howard", "michigan"): {"round": 64, "region": "South"},
            ("duke", "vermont"): {"round": 64, "region": "Midwest"},
            ("florida", "yale"): {"round": 64, "region": "West"},
        },
    }
    monkeypatch.setattr(extract_results, "_build_tournament_context", lambda year: context)

    records = [
        {"home_team": "Alpha", "away_team": "Beta", "home_score": 80, "away_score": 71, "date": "2026-03-19"},
        {"home_team": "Michigan", "away_team": "Howard", "home_score": 93, "away_score": 62, "date": "2026-03-19"},
        {"home_team": "Duke", "away_team": "Vermont", "home_score": 86, "away_score": 58, "date": "2026-03-19"},
        {"home_team": "Florida", "away_team": "Yale", "home_score": 77, "away_score": 64, "date": "2026-03-19"},
    ]

    result = extract_results.extract_partial_results_from_records(2026, records)

    assert result["year"] == 2026
    assert len(result["games"]) == 4
    assert all(game["round"] == 64 for game in result["games"])


def test_extract_partial_results_excludes_pre_tournament_matchups(monkeypatch):
    context = {
        "team_map": {
            "alpha": "Alpha",
            "beta": "Beta",
        },
        "team_meta": {
            "Alpha": {"seed": 1, "region": "East"},
            "Beta": {"seed": 2, "region": "West"},
        },
        "matchup_info": {
            ("alpha", "beta"): {"round": 2, "region": "Championship"},
        },
    }
    monkeypatch.setattr(extract_results, "_build_tournament_context", lambda year: context)

    result = extract_results.extract_partial_results_from_records(
        2026,
        [
            {
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 80,
                "away_score": 70,
                "date": "2026-03-13",
            }
        ],
    )

    assert result["games"] == []
