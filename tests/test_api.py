"""
API endpoint tests using FastAPI's TestClient (no live server needed).
"""
import sys
import os
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
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
