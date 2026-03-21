"""
API endpoint tests using FastAPI's TestClient (no live server needed).
"""
import collections
import sys
import os
import json
from datetime import datetime
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
import api
from api import app

client = TestClient(app)

# Some tests require teams_merged data which is not committed to the repo
HAS_TEAM_DATA = os.path.isfile(os.path.join(ROOT, "data", "teams_merged_2026.json"))


@pytest.fixture(autouse=True)
def clear_bracket_caches():
    api._load_bracket_file.cache_clear()
    api._tournament_team_map.cache_clear()
    api._exact_tournament_matchups.cache_clear()
    yield
    api._load_bracket_file.cache_clear()
    api._tournament_team_map.cache_clear()
    api._exact_tournament_matchups.cache_clear()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert isinstance(d["available_years"], list)
    if d["available_years"]:
        assert d["current_year"] == max(d["available_years"])
    else:
        assert d["current_year"] is None


def test_ready():
    """/ready returns 200 when teams_merged_2026.json exists, 503 otherwise."""
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert r.json()["ready"] is True


@pytest.mark.skipif(not HAS_TEAM_DATA, reason="teams_merged_2026.json not available in CI")
def test_teams_2026():
    r = client.get("/teams/2026")
    assert r.status_code == 200
    d = r.json()
    assert d["year"] == 2026
    assert d["count"] > 300
    assert all("team" in t and "barthag" in t and "adj_o" in t for t in d["teams"][:5])


def test_teams_missing_year():
    r = client.get("/teams/1999")
    assert r.status_code == 404


@pytest.mark.skipif(not HAS_TEAM_DATA, reason="teams_merged_2026.json not available in CI")
def test_predict_known_matchup():
    r = client.post("/predict", json={
        "team_a": "Duke", "team_b": "Houston", "year": 2026,
        "seed_a": 1, "seed_b": 2,
    })
    assert r.status_code == 200
    d = r.json()
    assert d["team_a"] == "Duke"
    assert d["team_b"] == "Houston"
    assert d["win_prob_a"] > 0.5        # Duke (#1) favored over Houston (#2)
    assert abs(d["win_prob_a"] + d["win_prob_b"] - 1.0) < 1e-4
    assert d["favorite"] == "Duke"
    assert isinstance(d["predicted_margin"], float)


@pytest.mark.skipif(not HAS_TEAM_DATA, reason="teams_merged_2026.json not available in CI")
def test_predict_home_a_game_site():
    r = client.post("/predict", json={
        "team_a": "Duke", "team_b": "Houston", "year": 2026,
        "seed_a": 1, "seed_b": 2, "game_site": "home_a",
    })
    assert r.status_code == 200
    d = r.json()
    assert d["team_a"] == "Duke"
    assert d["team_b"] == "Houston"


def test_predict_invalid_game_site():
    r = client.post("/predict", json={
        "team_a": "Duke", "team_b": "Houston", "year": 2026,
        "game_site": "not-a-site",
    })
    assert r.status_code == 422


def test_predict_unknown_team():
    r = client.post("/predict", json={
        "team_a": "Nonexistent University", "team_b": "Duke", "year": 2026,
    })
    assert r.status_code == 404


def test_predict_unknown_year():
    r = client.post("/predict", json={
        "team_a": "Duke", "team_b": "Houston", "year": 1990,
    })
    assert r.status_code == 404


def test_lookup_team_accepts_mascot_display_name(monkeypatch):
    monkeypatch.setattr(api, "load_teams_merged", lambda data_dir, year: {
        "kentucky": {"team": "Kentucky"},
        "santa clara": {"team": "Santa Clara"},
    })

    team = api._lookup_team("Kentucky Wildcats", 2026)

    assert team["team"] == "Kentucky"


@pytest.mark.skipif(not HAS_TEAM_DATA, reason="teams_merged_2026.json not available in CI")
def test_analyze_accepts_team_display_name_with_mascot():
    r = client.get("/analyze", params={
        "team_a": "Kentucky Wildcats",
        "team_b": "Santa Clara Broncos",
        "year": 2026,
        "round_name": "Round of 64",
        "region": "Midwest",
    })

    assert r.status_code == 200
    d = r.json()
    assert d["team_a"] == "Kentucky"
    assert d["team_b"] == "Santa Clara"


def test_bracket_picks_2026():
    r = client.get("/bracket/2026")
    assert r.status_code == 200
    d = r.json()
    assert d["year"] == 2026
    picks = d["picks"]["picks"]
    assert len(picks) == 63
    for p in picks:
        assert "pick" in p
        assert "win_prob" in p
        assert 0 < p["win_prob"] <= 1.0


def test_bracket_missing_year():
    r = client.get("/bracket/1999")
    assert r.status_code == 404


def test_bracket_upset_aggression_range():
    r = client.get("/bracket/2026?upset_aggression=1.5")  # out of range
    assert r.status_code == 422  # FastAPI validation error


def test_bracket_cache_busts_when_prediction_inputs_change(monkeypatch):
    monkeypatch.setattr(api, "_cache", collections.OrderedDict())

    current_inputs = {"mtime": "100"}
    calls = {"count": 0}

    monkeypatch.setattr(api, "_prediction_inputs_mtime", lambda year: current_inputs["mtime"])
    monkeypatch.setattr(api, "_load_bracket_for_year", lambda year: ({}, [], ["East", "West", "South", "Midwest"]))
    monkeypatch.setattr(api, "_load_config", lambda num_sims=10000: object())

    def fake_generate_bracket_picks(*args, **kwargs):
        calls["count"] += 1
        return {
            "picks": [],
            "champion": f"Champion {calls['count']}",
            "final_four": [],
            "biggest_upsets": [],
            "most_uncertain_games": [],
        }

    monkeypatch.setattr(api, "generate_bracket_picks", fake_generate_bracket_picks)

    first = client.get("/bracket/2026")
    assert first.status_code == 200
    assert first.json()["picks"]["champion"] == "Champion 1"
    assert calls["count"] == 1

    second = client.get("/bracket/2026")
    assert second.status_code == 200
    assert second.json()["picks"]["champion"] == "Champion 1"
    assert calls["count"] == 1

    current_inputs["mtime"] = "200"
    third = client.get("/bracket/2026")
    assert third.status_code == 200
    assert third.json()["picks"]["champion"] == "Champion 2"
    assert calls["count"] == 2


def test_monte_carlo_2026():
    r = client.get("/bracket/2026/monte-carlo?sims=200")
    assert r.status_code == 200
    d = r.json()
    assert d["year"] == 2026
    assert d["num_simulations"] == 200
    assert "champion_probs" in d
    # Probabilities should sum to ~1.0
    total = sum(d["champion_probs"].values())
    assert 0.95 <= total <= 1.05  # some rounding OK


def test_monte_carlo_sims_limit():
    r = client.get("/bracket/2026/monte-carlo?sims=200000")  # over max
    assert r.status_code == 422


def test_bets_card_filters_to_tournament_games(tmp_path, monkeypatch):
    today = datetime.now().strftime("%Y-%m-%d")
    card_path = tmp_path / f"card_{today}.json"
    card_path.write_text(json.dumps({
        "games": [
            {"home_team": "Duke", "away_team": "Mount St. Mary's", "picks": []},
            {"home_team": "North Texas", "away_team": "UAB", "picks": []},
        ]
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(
        api,
        "is_ncaa_tournament_game",
        lambda home, away, year=2026: {home, away} == {"Duke", "Mount St. Mary's"},
    )

    r = client.get("/bets/card")
    assert r.status_code == 200
    d = r.json()
    assert d["available"] is True
    assert len(d["games"]) == 1
    assert d["games"][0]["home_team"] == "Duke"
    assert d["games"][0]["ncaa_tournament"] is True


def test_filter_tournament_card_games_requires_exact_round_window(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "South": [
                {"team": "Florida", "seed": 1},
                {"team": "Team 16", "seed": 16},
                {"team": "Team 8", "seed": 8},
                {"team": "Team 9", "seed": 9},
                {"team": "Team 5", "seed": 5},
                {"team": "Vanderbilt", "seed": 12},
                {"team": "Team 4", "seed": 4},
                {"team": "Team 13", "seed": 13},
            ],
            "East": [],
            "West": [],
            "Midwest": [],
        },
        "quadrant_order": ["South", "East", "West", "Midwest"],
        "final_four_matchups": [[0, 3], [1, 2]],
        "first_four": [],
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))

    games = api._filter_tournament_card_games([
        {
            "home_team": "Florida Gators",
            "away_team": "Vanderbilt Commodores",
            "commence_time": "2026-03-15T17:00:00Z",
            "picks": [],
        },
        {
            "home_team": "Florida Gators",
            "away_team": "Vanderbilt Commodores",
            "commence_time": "2026-03-26T23:00:00Z",
            "picks": [],
        },
    ], year=2026)

    assert len(games) == 1
    assert games[0]["round_of"] == 16
    assert games[0]["round_name"] == "Sweet 16"


def test_bets_today_uses_game_date_from_commence_time(tmp_path, monkeypatch):
    ledger_path = tmp_path / "bets_ledger.json"
    ledger_path.write_text(json.dumps({
        "picks": [
            {
                "date": "2026-03-19",
                "home_team": "Duke",
                "away_team": "Arizona",
                "commence_time": "2026-03-20T17:00:00Z",
                "bet_type": "ml",
                "bet_side": "Duke",
            },
            {
                "date": "2026-03-19",
                "home_team": "Houston",
                "away_team": "Florida",
                "commence_time": "2026-03-21T05:00:00Z",
                "bet_type": "ml",
                "bet_side": "Houston",
            },
        ]
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "LEDGER_PATH", str(ledger_path))
    monkeypatch.setattr(api, "_today_et_str", lambda: "2026-03-20")
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(api, "is_ncaa_tournament_game", lambda home, away, year=2026: True)

    r = client.get("/bets/today")
    assert r.status_code == 200
    d = r.json()
    assert d["date"] == "2026-03-20"
    assert len(d["picks"]) == 1
    assert d["picks"][0]["home_team"] == "Duke"
    assert d["picks"][0]["date"] == "2026-03-20"


def test_bets_history_dedupes_same_pick_across_date_changes(tmp_path, monkeypatch):
    ledger_path = tmp_path / "bets_ledger.json"
    ledger_path.write_text(json.dumps({
        "picks": [
            {
                "date": "2026-03-19",
                "home_team": "Duke",
                "away_team": "Arizona",
                "commence_time": "2026-03-20T17:00:00Z",
                "bet_type": "ml",
                "bet_side": "Duke",
                "generated_at": "2026-03-19T14:00:00Z",
            },
            {
                "date": "2026-03-20",
                "home_team": "Duke",
                "away_team": "Arizona",
                "commence_time": "2026-03-20T17:00:00Z",
                "bet_type": "ml",
                "bet_side": "Duke",
                "generated_at": "2026-03-19T22:00:00Z",
            },
        ]
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "LEDGER_PATH", str(ledger_path))
    monkeypatch.setattr(api, "CARD_LEDGER_PATH", str(tmp_path / "card_ledger.json"))
    monkeypatch.setattr(api, "is_ncaa_tournament_game", lambda home, away, year=2026: True)

    r = client.get("/bets/history")
    assert r.status_code == 200
    d = r.json()
    assert len(d["picks"]) == 1
    assert d["picks"][0]["date"] == "2026-03-20"
    assert d["picks"][0]["generated_at"] == "2026-03-19T22:00:00Z"


def test_load_retro_card_games_dedupes_rescheduled_matchup(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "South": [
                {"team": "Nebraska", "seed": 3},
                {"team": "Team 14", "seed": 14},
                {"team": "Team 6", "seed": 6},
                {"team": "Vanderbilt", "seed": 11},
            ],
            "East": [],
            "West": [],
            "Midwest": [],
        },
        "quadrant_order": ["South", "East", "West", "Midwest"],
        "final_four_matchups": [[0, 3], [1, 2]],
        "first_four": [],
    }))
    (tmp_path / "card_2026-03-20.json").write_text(json.dumps({
        "games": [{
            "home_team": "Nebraska Cornhuskers",
            "away_team": "Vanderbilt Commodores",
            "commence_time": "2026-03-21T16:00:00Z",
            "picks": [],
        }],
    }))
    (tmp_path / "card_2026-03-21.json").write_text(json.dumps({
        "games": [{
            "home_team": "Nebraska Cornhuskers",
            "away_team": "Vanderbilt Commodores",
            "commence_time": "2026-03-22T00:45:00Z",
            "picks": [],
        }],
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(api, "refresh_saved_card_games", lambda games, year=None: games)

    games = api._load_retro_card_games(2026)

    assert len(games) == 1
    assert games[0]["commence_time"] == "2026-03-22T00:45:00Z"
    assert games[0]["round_of"] == 32


def test_get_bets_card_falls_back_to_today_card_ledger(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "East": [
                {"team": "Duke", "seed": 1},
                {"team": "American", "seed": 16},
            ],
            "West": [],
            "South": [],
            "Midwest": [],
        },
        "quadrant_order": ["East", "West", "Midwest", "South"],
        "final_four_matchups": [[0, 3], [1, 2]],
        "first_four": [],
    }))
    card_ledger_path = tmp_path / "card_ledger.json"
    card_ledger_path.write_text(json.dumps({
        "picks": [
            {
                "home_team": "Duke",
                "away_team": "American",
                "commence_time": "2026-03-20T18:50:00Z",
                "date": "2026-03-20",
                "generated_at": "2026-03-20T14:00:00Z",
                "bet_type": "ml",
                "bet_side": "Duke",
                "bet_odds": -600,
                "model_prob": 0.91,
                "implied_prob": 0.84,
                "model_margin": 18.5,
                "model_total": 143.0,
                "vegas_spread": -17.5,
                "vegas_total": 145.5,
            },
            {
                "home_team": "Duke",
                "away_team": "American",
                "commence_time": "2026-03-20T18:50:00Z",
                "date": "2026-03-20",
                "generated_at": "2026-03-20T14:00:00Z",
                "bet_type": "spread",
                "bet_team": "Duke",
                "bet_spread": -17.5,
                "bet_odds": -110,
                "model_margin": 18.5,
                "model_total": 143.0,
                "vegas_spread": -17.5,
                "vegas_total": 145.5,
            },
            {
                "home_team": "Duke",
                "away_team": "American",
                "commence_time": "2026-03-20T18:50:00Z",
                "date": "2026-03-20",
                "generated_at": "2026-03-20T14:00:00Z",
                "bet_type": "total",
                "bet_side": "UNDER",
                "bet_odds": -110,
                "model_margin": 18.5,
                "model_total": 143.0,
                "vegas_spread": -17.5,
                "vegas_total": 145.5,
            },
        ],
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "CARD_LEDGER_PATH", str(card_ledger_path))
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(api, "_today_et_str", lambda: "2026-03-20")
    monkeypatch.setattr(api, "refresh_saved_card_games", lambda games, year=None: games)

    r = client.get("/bets/card")
    assert r.status_code == 200
    d = r.json()
    assert d["date"] == "2026-03-20"
    assert d["available"] is True
    assert len(d["games"]) == 1
    assert d["games"][0]["round_name"] == "Round of 64"
    assert len(d["games"][0]["picks"]) == 3


def test_annotate_tournament_record_accepts_mascot_variant_matchup(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "East": [
                {"team": "Duke", "seed": 1},
                {"team": "Team 16", "seed": 16},
                {"team": "Team 8", "seed": 8},
                {"team": "TCU", "seed": 9},
            ],
            "West": [],
            "South": [],
            "Midwest": [],
        },
        "quadrant_order": ["East", "West", "South", "Midwest"],
        "final_four_matchups": [[0, 3], [1, 2]],
        "first_four": [],
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))

    annotated = api._annotate_tournament_record({
        "home_team": "Duke Blue Devils",
        "away_team": "TCU Horned Frogs",
        "commence_time": "2026-03-21T21:15:00Z",
    }, year=2026)

    assert annotated is not None
    assert annotated["round_of"] == 32
    assert annotated["round_name"] == "Round of 32"


def test_load_current_card_games_from_ledger_prefers_latest_market_snapshot(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "East": [
                {"team": "Michigan St.", "seed": 3},
                {"team": "Team 14", "seed": 14},
                {"team": "Team 6", "seed": 6},
                {"team": "Louisville", "seed": 11},
            ],
            "West": [],
            "South": [],
            "Midwest": [],
        },
        "quadrant_order": ["East", "West", "South", "Midwest"],
        "final_four_matchups": [[0, 3], [1, 2]],
        "first_four": [],
    }))
    card_ledger_path = tmp_path / "card_ledger.json"
    card_ledger_path.write_text(json.dumps({
        "picks": [
            {
                "home_team": "Michigan St Spartans",
                "away_team": "Louisville Cardinals",
                "commence_time": "2026-03-21T16:00:00Z",
                "date": "2026-03-21",
                "bet_type": "ml",
                "bet_side": "Michigan St Spartans",
            },
            {
                "home_team": "Michigan St Spartans",
                "away_team": "Louisville Cardinals",
                "commence_time": "2026-03-21T16:00:00Z",
                "date": "2026-03-21",
                "bet_type": "spread",
                "bet_team": "Michigan St Spartans",
            },
            {
                "home_team": "Michigan St Spartans",
                "away_team": "Louisville Cardinals",
                "commence_time": "2026-03-21T16:00:00Z",
                "date": "2026-03-21",
                "bet_type": "total",
                "bet_side": "OVER",
            },
            {
                "home_team": "Michigan St Spartans",
                "away_team": "Louisville Cardinals",
                "commence_time": "2026-03-21T18:45:00Z",
                "date": "2026-03-21",
                "bet_type": "ml",
                "bet_side": "Louisville Cardinals",
            },
            {
                "home_team": "Michigan St Spartans",
                "away_team": "Louisville Cardinals",
                "commence_time": "2026-03-21T18:45:00Z",
                "date": "2026-03-21",
                "bet_type": "spread",
                "bet_team": "Louisville Cardinals",
            },
            {
                "home_team": "Michigan St Spartans",
                "away_team": "Louisville Cardinals",
                "commence_time": "2026-03-21T18:45:00Z",
                "date": "2026-03-21",
                "bet_type": "total",
                "bet_side": "UNDER",
            },
        ]
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "CARD_LEDGER_PATH", str(card_ledger_path))
    monkeypatch.setattr(api, "_today_et_str", lambda: "2026-03-21")

    games = api._load_current_card_games_from_ledger(2026, refresh=False)

    assert len(games) == 1
    assert games[0]["commence_time"] == "2026-03-21T18:45:00Z"
    assert [p["bet_type"] for p in games[0]["picks"]] == ["ml", "spread", "total"]


def test_get_bets_card_prefers_current_ledger_over_saved_snapshot(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "East": [
                {"team": "Duke", "seed": 1},
                {"team": "Team 16", "seed": 16},
                {"team": "Team 8", "seed": 8},
                {"team": "TCU", "seed": 9},
            ],
            "West": [],
            "South": [],
            "Midwest": [],
        },
        "quadrant_order": ["East", "West", "South", "Midwest"],
        "final_four_matchups": [[0, 3], [1, 2]],
        "first_four": [],
    }))
    (tmp_path / "card_2026-03-21.json").write_text(json.dumps({
        "games": [
            {
                "home_team": "Duke Blue Devils",
                "away_team": "TCU Horned Frogs",
                "commence_time": "2026-03-21T16:00:00Z",
                "picks": [],
            }
        ]
    }))
    card_ledger_path = tmp_path / "card_ledger.json"
    card_ledger_path.write_text(json.dumps({
        "picks": [
            {
                "home_team": "Duke Blue Devils",
                "away_team": "TCU Horned Frogs",
                "commence_time": "2026-03-21T21:15:00Z",
                "date": "2026-03-21",
                "bet_type": "ml",
                "bet_side": "Duke Blue Devils",
            },
            {
                "home_team": "Duke Blue Devils",
                "away_team": "TCU Horned Frogs",
                "commence_time": "2026-03-21T21:15:00Z",
                "date": "2026-03-21",
                "bet_type": "spread",
                "bet_team": "Duke Blue Devils",
            },
            {
                "home_team": "Duke Blue Devils",
                "away_team": "TCU Horned Frogs",
                "commence_time": "2026-03-21T21:15:00Z",
                "date": "2026-03-21",
                "bet_type": "total",
                "bet_side": "UNDER",
            },
        ]
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "CARD_LEDGER_PATH", str(card_ledger_path))
    monkeypatch.setattr(api, "_today_et_str", lambda: "2026-03-21")
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(api, "refresh_saved_card_games", lambda games, year=None: games)

    r = client.get("/bets/card")
    assert r.status_code == 200
    d = r.json()
    assert d["date"] == "2026-03-21"
    assert len(d["games"]) == 1
    assert d["games"][0]["commence_time"] == "2026-03-21T21:15:00Z"


def test_bracket_scores_maps_to_bracket_team_names(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "East": [
                {"team": "Duke", "seed": 1},
                {"team": "American", "seed": 16},
            ],
            "West": [],
            "South": [],
            "Midwest": [],
        },
        "quadrant_order": ["East", "West", "Midwest", "South"],
        "final_four_matchups": [[0, 3], [1, 2]],
        "first_four": [{"team_a": "Mount St. Mary's", "team_b": "American", "seed": 16, "region": "East"}],
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(api, "fetch_espn_scoreboard", lambda dates: [{
        "home_team": "Duke",
        "away_team": "Mount St. Mary's",
        "scheduled_at": "2026-03-19T23:00:00Z",
        "home_score": 81,
        "away_score": 77,
        "completed": True,
        "status_detail": "Final",
        "display_clock": "",
        "period": 2,
    }])

    r = client.get("/bracket/2026/scores")
    assert r.status_code == 200
    d = r.json()
    assert d["scores"]["Duke|Mount St. Mary's"]["score_a"] == 81
    assert d["scores"]["Duke|Mount St. Mary's"]["score_b"] == 77
    assert d["scores"]["Duke|Mount St. Mary's"]["round_of"] == 64
    assert d["scores"]["Mount St. Mary's|Duke"]["score_a"] == 77
    assert d["scores"]["Mount St. Mary's|Duke"]["score_b"] == 81


def test_bracket_scores_strip_espn_mascots_and_aliases(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "East": [{"team": "Duke", "seed": 1}],
            "West": [{"team": "Arizona", "seed": 1}],
            "Midwest": [{"team": "Houston", "seed": 1}],
            "South": [{"team": "Florida", "seed": 1}],
        },
        "quadrant_order": ["East", "West", "Midwest", "South"],
        "final_four_matchups": [[0, 3], [1, 2]],
        "first_four": [],
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(api, "fetch_espn_scoreboard", lambda dates: [{
        "home_team": "Duke Blue Devils",
        "away_team": "Florida Gators",
        "home_aliases": ["Duke", "Duke Blue Devils"],
        "away_aliases": ["Florida", "Florida Gators"],
        "scheduled_at": "2026-04-04T23:00:00Z",
        "home_score": 71,
        "away_score": 68,
        "completed": True,
        "status_detail": "Final",
        "display_clock": "",
        "period": 2,
    }])

    r = client.get("/bracket/2026/scores")
    assert r.status_code == 200
    d = r.json()
    assert d["scores"]["Duke|Florida"]["score_a"] == 71
    assert d["scores"]["Florida|Duke"]["score_b"] == 71
    assert d["scores"]["Duke|Florida"]["round_of"] == 4


def test_bracket_scores_skip_pre_tournament_game_for_same_pair(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "West": [{"team": "Florida"}],
            "South": [{"team": "Vanderbilt"}],
        },
        "first_four": [],
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(api, "fetch_espn_scoreboard", lambda dates: [{
        "home_team": "Florida",
        "away_team": "Vanderbilt",
        "scheduled_at": "2026-03-15T20:00:00Z",
        "home_score": 74,
        "away_score": 71,
        "completed": True,
        "status_detail": "Final",
        "display_clock": "",
        "period": 2,
    }])

    r = client.get("/bracket/2026/scores")
    assert r.status_code == 200
    d = r.json()
    assert d["scores"] == {}
