"""
API endpoint tests using FastAPI's TestClient (no live server needed).
"""
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


def test_bracket_scores_maps_to_bracket_team_names(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "East": [{"team": "Duke"}],
            "West": [{"team": "Arizona"}],
        },
        "first_four": [{"team_a": "Mount St. Mary's", "team_b": "American"}],
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(api, "fetch_espn_scoreboard", lambda dates: [{
        "home_team": "Duke",
        "away_team": "Arizona",
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
    assert d["scores"]["Duke|Arizona"]["score_a"] == 81
    assert d["scores"]["Duke|Arizona"]["score_b"] == 77
    assert d["scores"]["Arizona|Duke"]["score_a"] == 77
    assert d["scores"]["Arizona|Duke"]["score_b"] == 81


def test_bracket_scores_strip_espn_mascots_and_aliases(tmp_path, monkeypatch):
    bracket_path = tmp_path / "bracket_2026.json"
    bracket_path.write_text(json.dumps({
        "regions": {
            "East": [{"team": "Duke"}],
            "South": [{"team": "Florida"}],
        },
        "first_four": [],
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", {})
    monkeypatch.setattr(api, "fetch_espn_scoreboard", lambda dates: [{
        "home_team": "Duke Blue Devils",
        "away_team": "Florida Gators",
        "home_aliases": ["Duke", "Duke Blue Devils"],
        "away_aliases": ["Florida", "Florida Gators"],
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
