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


def test_bracket_picks_2026(monkeypatch):
    monkeypatch.setattr(api, "_static_artifact_cache", {})
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


def test_bracket_actual_404_when_no_precomputed_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_static_artifact_cache", {})
    r = client.get("/bracket/2026/actual")
    assert r.status_code == 404


def test_bracket_actual_serves_precomputed_artifact(tmp_path, monkeypatch):
    (tmp_path / "bracket_actual_2026.json").write_text(json.dumps({
        "year": 2026, "upset_aggression": 0.0,
        "picks": {"picks": [], "champion": "RealChamp", "final_four": [],
                  "biggest_upsets": [], "most_uncertain_games": []},
        "quadrant_order": ["East", "West", "South", "Midwest"], "ff_pairs": [],
    }))
    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_static_artifact_cache", {})
    r = client.get("/bracket/2026/actual")
    assert r.status_code == 200
    assert r.json()["picks"]["champion"] == "RealChamp"
    assert r.headers.get("cache-control") == "public, max-age=3600"


def test_simulate_uses_only_user_locks_for_advancement_not_real_results(tmp_path, monkeypatch):
    """POST /bracket/{year}/simulate must advance strictly on the user's own
    manually-locked picks -- real completed-game results must never be
    merged into locked_picks here either, only supplied as grading_locks."""
    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_load_bracket_for_year", lambda year: ({}, [], ["East", "West", "South", "Midwest"]))
    monkeypatch.setattr(api, "_load_config", lambda: object())
    monkeypatch.setattr(api, "_completed_tournament_locked_picks", lambda year, bracket, quadrant_order, ff_matchups: {"East-64-0": "RealWinner"})

    seen = {}

    def fake_generate_bracket_picks(*args, **kwargs):
        seen["locked_picks"] = kwargs.get("locked_picks")
        seen["grading_locks"] = kwargs.get("grading_locks")
        return {
            "picks": [], "champion": "UserPick", "final_four": [],
            "biggest_upsets": [], "most_uncertain_games": [],
        }

    monkeypatch.setattr(api, "generate_bracket_picks", fake_generate_bracket_picks)

    r = client.post("/bracket/2026/simulate", json={
        "upset_aggression": 0.3,
        "locked_picks": {"West-64-0": "UserLockedTeam"},
    })
    assert r.status_code == 200
    assert seen["locked_picks"] == {"West-64-0": "UserLockedTeam"}
    assert seen["grading_locks"] == {"East-64-0": "RealWinner"}


def test_bracket_cache_busts_when_prediction_inputs_change(tmp_path, monkeypatch):
    # DATA_DIR -> tmp_path (no real bracket_picks_2026.json there) so this exercises
    # the live-compute path, not the precomputed-artifact fast path.
    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", collections.OrderedDict())
    monkeypatch.setattr(api, "_static_artifact_cache", {})

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


def test_bracket_never_locks_advancement_to_real_results(tmp_path, monkeypatch):
    """GET /bracket/{year} (the simulator) must never pass real results as
    locked_picks -- that would let actual outcomes override the model's own
    advancement, exactly the bug this test guards against. Real results
    still flow in, but only as grading_locks (badges/accuracy), never as
    locked_picks (which team advances)."""
    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", collections.OrderedDict())
    monkeypatch.setattr(api, "_static_artifact_cache", {})
    monkeypatch.setattr(api, "_prediction_inputs_mtime", lambda year: "100")
    monkeypatch.setattr(api, "_load_bracket_for_year", lambda year: ({}, [], ["East", "West", "South", "Midwest"]))
    monkeypatch.setattr(api, "_load_config", lambda num_sims=10000: object())
    monkeypatch.setattr(api, "_completed_tournament_locked_picks", lambda year, bracket, quadrant_order, ff_matchups: {"East-64-0": "Alpha"})

    seen = {}

    def fake_generate_bracket_picks(*args, **kwargs):
        seen["locked_picks"] = kwargs.get("locked_picks")
        seen["grading_locks"] = kwargs.get("grading_locks")
        return {
            "picks": [],
            "champion": "Alpha",
            "final_four": [],
            "biggest_upsets": [],
            "most_uncertain_games": [],
        }

    monkeypatch.setattr(api, "generate_bracket_picks", fake_generate_bracket_picks)

    r = client.get("/bracket/2026?live=true")
    assert r.status_code == 200
    assert seen["locked_picks"] is None
    assert seen["grading_locks"] == {"East-64-0": "Alpha"}


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


def test_monte_carlo_default_sims_always_serves_precomputed_file(tmp_path, monkeypatch):
    """Default (sims=10000) must always serve the precomputed artifact as a pure
    static read -- no per-request hashing, no live recompute, even if a fresh
    hash would differ from what's stored in the file. Freshness is a
    regenerate-and-redeploy concern (`python run.py --all`), not a per-request
    check: a cold free-tier instance must never run a live 10k-sim simulation
    just to serve a page load."""
    mc_path = tmp_path / "monte_carlo_2026.json"
    mc_path.write_text(json.dumps({
        "year": 2026,
        "num_simulations": 10000,
        "prediction_inputs_hash": "stalehash",
        "champion_probs": {"Precomputed Team": 1.0},
        "final_four_probs": {"Precomputed Team": 1.0},
        "elite_eight_probs": {"Precomputed Team": 1.0},
        "sweet_sixteen_probs": {"Precomputed Team": 1.0},
        "round_of_32_probs": {"Precomputed Team": 1.0},
        "final_four_by_region": {},
    }))

    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", collections.OrderedDict())
    monkeypatch.setattr(api, "_static_artifact_cache", {})
    monkeypatch.setattr(api, "_prediction_inputs_hash", lambda year: "freshhash")

    def fail_if_called(*a, **k):
        raise AssertionError("live Monte Carlo compute must not run for default sims")
    monkeypatch.setattr(api, "run_monte_carlo", fail_if_called)

    r = client.get("/bracket/2026/monte-carlo?sims=10000")
    assert r.status_code == 200
    d = r.json()
    assert d["champion_probs"] == {"Precomputed Team": 1.0}
    assert d["prediction_inputs_hash"] == "stalehash"
    assert r.headers.get("cache-control") == "public, max-age=3600"


def test_monte_carlo_never_passes_locked_picks(tmp_path, monkeypatch):
    """The Monte Carlo / Championship & F4 Odds path must always be the
    model's own clean simulation -- never conditioned on real results (that
    belongs only to generate_bracket_picks / the bracket-picks display).
    fake_run_monte_carlo has no locked_picks parameter, so if the route ever
    tries to pass one again, this call raises TypeError and the request
    fails instead of silently succeeding. No precomputed file exists in
    tmp_path, so sims=10000 still falls through to a live compute here."""
    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_cache", collections.OrderedDict())
    monkeypatch.setattr(api, "_static_artifact_cache", {})
    monkeypatch.setattr(api, "_prediction_inputs_hash", lambda year: "freshhash")
    monkeypatch.setattr(api, "_prediction_inputs_mtime", lambda year: "123")
    monkeypatch.setattr(api, "_load_bracket_for_year", lambda year: ({}, [], ["East", "West", "South", "Midwest"]))
    monkeypatch.setattr(api, "_load_config", lambda num_sims=10000: object())
    monkeypatch.setattr(api, "_add_final_four_by_region", lambda result, year: result)

    def fake_run_monte_carlo(bracket, config=None, year=None, quadrant_order=None, ff_matchups=None):
        return {
            "champion_probs": {"New Team": 1.0},
            "final_four_probs": {"New Team": 1.0},
            "elite_eight_probs": {"New Team": 1.0},
            "sweet_sixteen_probs": {"New Team": 1.0},
            "round_of_32_probs": {"New Team": 1.0},
        }

    monkeypatch.setattr(api, "run_monte_carlo", fake_run_monte_carlo)

    r = client.get("/bracket/2026/monte-carlo?sims=10000")
    assert r.status_code == 200
    assert r.json()["champion_probs"] == {"New Team": 1.0}


def test_monte_carlo_sims_limit():
    r = client.get("/bracket/2026/monte-carlo?sims=200000")  # over max
    assert r.status_code == 422


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
