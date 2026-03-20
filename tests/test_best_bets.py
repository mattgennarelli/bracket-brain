import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts import best_bets


def test_lookup_team_keeps_tennessee_state_distinct_from_tennessee():
    teams = {
        "tennessee": {"team": "Tennessee"},
        "tennessee st": {"team": "Tennessee St."},
    }

    team = best_bets.lookup_team("Tennessee St Tigers", teams)

    assert team is not None
    assert team["team"] == "Tennessee St."


def test_get_full_card_prefers_positive_edge_moneyline_side(monkeypatch):
    future_time = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat().replace("+00:00", "Z")

    monkeypatch.setattr(best_bets, "fetch_today_odds", lambda api_key: [{}])
    monkeypatch.setattr(best_bets, "parse_game", lambda raw: {
        "home_team": "Iowa State Cyclones",
        "away_team": "Tennessee St Tigers",
        "commence_time": future_time,
        "ml_home": -10000,
        "ml_away": 1800,
        "spread_home": -24.5,
        "spread_line": -110,
        "total_line": 148.5,
    })
    monkeypatch.setattr(best_bets, "load_team_stats", lambda year: {"dummy": {}})
    monkeypatch.setattr(best_bets, "lookup_team", lambda name, teams: {
        "team": name,
        "adj_o": 120.0,
        "adj_d": 95.0,
    })
    monkeypatch.setattr(best_bets, "run_model", lambda home, away, config, round_name=None, region=None, year=None: {
        "win_prob_a": 0.729,
        "predicted_margin": 7.0,
        "predicted_score_a": 74.0,
        "predicted_score_b": 66.2,
    })

    games = best_bets.get_full_card_json("fake-key", year=2026)

    assert len(games) == 1
    picks = games[0]["picks"]
    ml_pick = next(p for p in picks if p["bet_type"] == "ml")
    assert ml_pick["bet_side"] == "Tennessee St Tigers"
    assert ml_pick["edge"] > 0
    assert any(p["bet_type"] == "spread" for p in picks)
    assert any(p["bet_type"] == "total" for p in picks)


def test_get_full_card_always_returns_moneyline_lean(monkeypatch):
    future_time = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat().replace("+00:00", "Z")

    monkeypatch.setattr(best_bets, "fetch_today_odds", lambda api_key: [{}])
    monkeypatch.setattr(best_bets, "parse_game", lambda raw: {
        "home_team": "Alabama Crimson Tide",
        "away_team": "Hofstra Pride",
        "commence_time": future_time,
        "ml_home": -800,
        "ml_away": 550,
        "spread_home": -11.5,
        "spread_line": -110,
        "total_line": 158.5,
    })
    monkeypatch.setattr(best_bets, "load_team_stats", lambda year: {"dummy": {}})
    monkeypatch.setattr(best_bets, "lookup_team", lambda name, teams: {
        "team": name,
        "adj_o": 120.0,
        "adj_d": 95.0,
    })
    monkeypatch.setattr(best_bets, "run_model", lambda home, away, config, round_name=None, region=None, year=None: {
        "win_prob_a": 0.7507,
        "predicted_margin": 8.2,
        "predicted_score_a": 84.0,
        "predicted_score_b": 81.3,
    })

    games = best_bets.get_full_card_json("fake-key", year=2026)

    picks = games[0]["picks"]
    ml_pick = next(p for p in picks if p["bet_type"] == "ml")
    assert ml_pick["bet_side"] == "Hofstra Pride"
    assert ml_pick["edge"] > 0


def test_refresh_saved_card_games_recomputes_snapshot(monkeypatch):
    monkeypatch.setattr(best_bets, "load_team_stats", lambda year: {"dummy": {}})
    monkeypatch.setattr(best_bets, "lookup_team", lambda name, teams: {
        "team": name,
        "adj_o": 120.0,
        "adj_d": 95.0,
    })
    monkeypatch.setattr(best_bets, "run_model", lambda home, away, config, round_name=None, region=None, year=None: {
        "win_prob_a": 0.9936,
        "predicted_margin": 29.9,
        "predicted_score_a": 92.4,
        "predicted_score_b": 62.5,
    })

    refreshed = best_bets.refresh_saved_card_games([{
        "home_team": "Iowa State Cyclones",
        "away_team": "Tennessee St Tigers",
        "commence_time": "2026-03-20T18:50:00Z",
        "data_available": True,
        "model_prob_home": 0.7287,
        "model_margin": 7.0,
        "model_total": 140.2,
        "picks": [
            {
                "bet_type": "ml",
                "bet_side": "Iowa State Cyclones",
                "bet_odds": -10000,
                "implied_prob": 0.9541,
                "vegas_spread": -24.5,
                "vegas_total": 148.5,
            },
            {
                "bet_type": "spread",
                "bet_team": "Tennessee St Tigers",
                "bet_spread": 24.5,
                "bet_odds": -110,
                "vegas_spread": -24.5,
                "vegas_total": 148.5,
            },
            {
                "bet_type": "total",
                "bet_side": "UNDER",
                "bet_odds": -110,
                "vegas_total": 148.5,
                "vegas_spread": -24.5,
            },
        ],
    }], year=2026)

    game = refreshed[0]
    assert game["model_prob_home"] == 0.9936
    assert game["model_margin"] == 29.9
    assert any(p["bet_type"] == "ml" and p["bet_side"] == "Iowa State Cyclones" for p in game["picks"])


def test_refresh_saved_card_games_keeps_ml_when_model_flips_side(monkeypatch):
    monkeypatch.setattr(best_bets, "load_team_stats", lambda year: {"dummy": {}})
    monkeypatch.setattr(best_bets, "lookup_team", lambda name, teams: {
        "team": name,
        "adj_o": 120.0,
        "adj_d": 95.0,
    })
    monkeypatch.setattr(best_bets, "run_model", lambda home, away, config, round_name=None, region=None, year=None: {
        "win_prob_a": 0.4907,
        "predicted_margin": -0.3,
        "predicted_score_a": 70.1,
        "predicted_score_b": 66.6,
    })

    refreshed = best_bets.refresh_saved_card_games([{
        "home_team": "Saint Mary's Gaels",
        "away_team": "Texas A&M Aggies",
        "commence_time": "2026-03-20T18:50:00Z",
        "data_available": True,
        "picks": [
            {
                "bet_type": "ml",
                "bet_side": "Saint Mary's Gaels",
                "bet_odds": -158,
                "implied_prob": 0.59,
                "vegas_spread": -2.5,
                "vegas_total": 145.5,
            },
            {
                "bet_type": "spread",
                "bet_team": "Saint Mary's Gaels",
                "bet_spread": -2.5,
                "bet_odds": -110,
                "vegas_spread": -2.5,
                "vegas_total": 145.5,
            },
            {
                "bet_type": "total",
                "bet_side": "OVER",
                "bet_odds": -110,
                "vegas_total": 145.5,
                "vegas_spread": -2.5,
            },
        ],
    }], year=2026)

    ml_pick = next(p for p in refreshed[0]["picks"] if p["bet_type"] == "ml")
    assert ml_pick["bet_side"] == "Texas A&M Aggies"
    assert ml_pick["bet_odds"] is not None


def test_run_model_uses_analysis_pipeline_for_tournament_games(monkeypatch):
    captured = {}

    def fake_analysis(home, away, data_dir=None, year=None, region=None, round_name=None, config=None):
        captured["home"] = home["team"]
        captured["away"] = away["team"]
        captured["year"] = year
        captured["region"] = region
        captured["round_name"] = round_name
        return {
            "win_prob_a": 0.541,
            "predicted_margin": 1.2,
            "predicted_score_a": 78.4,
            "predicted_score_b": 76.1,
        }

    monkeypatch.setattr(best_bets, "get_matchup_analysis_display", fake_analysis)

    result = best_bets.run_model(
        {"team": "Kentucky", "adj_o": 120.0, "adj_d": 96.0, "seed": 7},
        {"team": "Santa Clara", "adj_o": 118.0, "adj_d": 97.0, "seed": 10},
        round_name="Round of 64",
        region="Midwest",
        year=2026,
    )

    assert result["win_prob_a"] == 0.541
    assert result["predicted_margin"] == 1.2
    assert result["predicted_score_a"] == 78.4
    assert captured == {
        "home": "Kentucky",
        "away": "Santa Clara",
        "year": 2026,
        "region": "Midwest",
        "round_name": "Round of 64",
    }
