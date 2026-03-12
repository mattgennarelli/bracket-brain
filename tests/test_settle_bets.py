"""
Unit tests for settle_bets.py — ML, spread, and total settle logic.

Uses mocked Odds API response; no live API calls.
"""
import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import from scripts
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from settle_bets import (
    settle_pick,
    match_score,
    parse_scores,
    compute_stats,
    _scores_key,
)


# ---------------------------------------------------------------------------
# Mock Odds API score response (completed game format)
# ---------------------------------------------------------------------------

def _mock_score_record(home_team, away_team, home_score, away_score):
    """Build a score record in Odds API format."""
    return {
        "id": "mock_id",
        "completed": True,
        "home_team": home_team,
        "away_team": away_team,
        "scores": [
            {"name": home_team, "score": str(home_score)},
            {"name": away_team, "score": str(away_score)},
        ],
    }


# ---------------------------------------------------------------------------
# settle_pick tests
# ---------------------------------------------------------------------------

def test_settle_ml_home_wins_pick_home():
    pick = {"bet_type": "ml", "bet_side": "Duke", "home_team": "Duke", "away_team": "UNC"}
    assert settle_pick(pick, 85, 70) == "W"


def test_settle_ml_home_wins_pick_away():
    pick = {"bet_type": "ml", "bet_side": "UNC", "home_team": "Duke", "away_team": "UNC"}
    assert settle_pick(pick, 85, 70) == "L"


def test_settle_ml_away_wins_pick_away():
    pick = {"bet_type": "ml", "bet_side": "UNC", "home_team": "Duke", "away_team": "UNC"}
    assert settle_pick(pick, 70, 85) == "W"


def test_settle_ml_away_wins_pick_home():
    pick = {"bet_type": "ml", "bet_side": "Duke", "home_team": "Duke", "away_team": "UNC"}
    assert settle_pick(pick, 70, 85) == "L"


def test_settle_ml_tie():
    pick = {"bet_type": "ml", "bet_side": "Duke", "home_team": "Duke", "away_team": "UNC"}
    assert settle_pick(pick, 70, 70) == "P"


def test_settle_spread_home_team_covers():
    # Home -7.5: home needs to win by 8+
    pick = {"bet_type": "spread", "bet_team": "Duke", "home_team": "Duke", "away_team": "UNC", "vegas_spread": -7.5}
    assert settle_pick(pick, 85, 76) == "W"  # margin 9, cover by 1.5


def test_settle_spread_home_team_does_not_cover():
    pick = {"bet_type": "spread", "bet_team": "Duke", "home_team": "Duke", "away_team": "UNC", "vegas_spread": -7.5}
    assert settle_pick(pick, 80, 75) == "L"  # margin 5, need 8


def test_settle_spread_away_team_covers():
    pick = {"bet_type": "spread", "bet_team": "UNC", "home_team": "Duke", "away_team": "UNC", "vegas_spread": -7.5}
    # Duke -7.5 means UNC +7.5. Duke wins 80-75 (margin 5). UNC +7.5: 75+7.5=82.5 vs 80, so UNC covers
    assert settle_pick(pick, 80, 75) == "W"


def test_settle_spread_push():
    # spread -7: home needs to win by 8. 85-78 = 7, cover = 7 + (-7) = 0
    pick = {"bet_type": "spread", "bet_team": "Duke", "home_team": "Duke", "away_team": "UNC", "vegas_spread": -7.0}
    assert settle_pick(pick, 85, 78) == "P"


def test_settle_total_over_hits():
    pick = {"bet_type": "total", "bet_side": "OVER", "vegas_total": 150, "home_team": "A", "away_team": "B"}
    assert settle_pick(pick, 80, 75) == "W"  # 155 > 150


def test_settle_total_over_misses():
    pick = {"bet_type": "total", "bet_side": "OVER", "vegas_total": 150, "home_team": "A", "away_team": "B"}
    assert settle_pick(pick, 70, 75) == "L"  # 145 < 150


def test_settle_total_under_hits():
    pick = {"bet_type": "total", "bet_side": "UNDER", "vegas_total": 150, "home_team": "A", "away_team": "B"}
    assert settle_pick(pick, 70, 75) == "W"  # 145 < 150


def test_settle_total_under_misses():
    pick = {"bet_type": "total", "bet_side": "UNDER", "vegas_total": 150, "home_team": "A", "away_team": "B"}
    assert settle_pick(pick, 80, 75) == "L"  # 155 > 150


def test_settle_total_push():
    pick = {"bet_type": "total", "bet_side": "OVER", "vegas_total": 150, "home_team": "A", "away_team": "B"}
    assert settle_pick(pick, 75, 75) == "P"  # 150 == 150


def test_settle_spread_missing_vegas_spread():
    pick = {"bet_type": "spread", "bet_team": "Duke", "home_team": "Duke", "away_team": "UNC"}
    assert settle_pick(pick, 85, 70) is None


def test_settle_total_missing_vegas_total():
    pick = {"bet_type": "total", "bet_side": "OVER", "home_team": "A", "away_team": "B"}
    assert settle_pick(pick, 80, 75) is None


# ---------------------------------------------------------------------------
# match_score tests
# ---------------------------------------------------------------------------

def test_match_score_direct():
    scores_by_key = {"duke|north carolina": _mock_score_record("Duke", "North Carolina", 85, 70)}
    pick = {"home_team": "Duke", "away_team": "North Carolina"}
    rec = match_score(pick, scores_by_key)
    assert rec is not None
    assert rec["home_team"] == "Duke"
    home, away = parse_scores(rec)
    assert home == 85 and away == 70


def test_match_score_flipped_order():
    scores_by_key = {"north carolina|duke": _mock_score_record("North Carolina", "Duke", 70, 85)}
    pick = {"home_team": "Duke", "away_team": "North Carolina"}
    rec = match_score(pick, scores_by_key)
    assert rec is not None
    assert rec.get("_scores_flipped") is True
    home, away = parse_scores(rec)
    assert home == 85 and away == 70  # flipped so home=Duke gets 85


def test_match_score_not_found():
    scores_by_key = {}
    pick = {"home_team": "Duke", "away_team": "North Carolina"}
    assert match_score(pick, scores_by_key) is None


# ---------------------------------------------------------------------------
# parse_scores tests
# ---------------------------------------------------------------------------

def test_parse_scores_normal():
    rec = _mock_score_record("Duke", "UNC", 85, 70)
    assert parse_scores(rec) == (85.0, 70.0)


def test_parse_scores_flipped():
    rec = {"home_team": "Duke", "away_team": "UNC", "_scores_flipped": True,
           "scores": [{"name": "UNC", "score": "70"}, {"name": "Duke", "score": "85"}]}
    home, away = parse_scores(rec)
    assert home == 85 and away == 70


# ---------------------------------------------------------------------------
# compute_stats tests
# ---------------------------------------------------------------------------

def test_compute_stats():
    picks = [
        {"bet_type": "ml", "result": "W"},
        {"bet_type": "ml", "result": "L"},
        {"bet_type": "spread", "result": "W"},
        {"bet_type": "total", "result": "P"},
    ]
    stats = compute_stats(picks)
    assert stats["settled"] == 4
    assert stats["wins"] == 2
    assert stats["losses"] == 1
    assert stats["pushes"] == 1
    assert abs(stats["hit_rate"] - 2 / 3) < 0.001  # W/(W+L), stored as rounded float
    assert stats["by_type"]["ml"]["hit_rate"] == 0.5
    assert stats["by_type"]["spread"]["hit_rate"] == 1.0
