"""
Unit tests for engine.py — scaling invariants, name normalization, and core predictions.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import (
    calc_momentum_bonus, enrich_team, _normalize_team_for_match,
    predict_game, ModelConfig, resolve_ff_pairs, get_matchup_analysis_display,
    enrich_bracket_with_teams, _calc_upset_tolerance_bonus,
)


# ---------------------------------------------------------------------------
# enrich_team scaling invariants
# ---------------------------------------------------------------------------

def _base_team(**kwargs):
    defaults = {
        "team": "Test Team", "seed": 5,
        "adj_o": 110.0, "adj_d": 100.0, "adj_tempo": 68.0, "barthag": 0.75,
    }
    defaults.update(kwargs)
    return defaults


def test_two_pt_pct_percentage_form_normalized():
    """two_pt_pct from Torvik adv data comes as e.g. 47.8 — must be scaled to 0-1."""
    t = enrich_team(_base_team(two_pt_pct=47.8))
    assert t["two_pt_pct"] <= 1.0, f"Expected ≤1.0, got {t['two_pt_pct']}"
    assert abs(t["two_pt_pct"] - 0.478) < 1e-6


def test_two_pt_pct_already_fractional_unchanged():
    t = enrich_team(_base_team(two_pt_pct=0.478))
    assert abs(t["two_pt_pct"] - 0.478) < 1e-6


def test_three_pct_percentage_form_normalized():
    t = enrich_team(_base_team(three_pt_pct=35.5))
    assert t["three_pct"] <= 1.0, f"Expected ≤1.0, got {t['three_pct']}"
    assert abs(t["three_pct"] - 0.355) < 1e-6


def test_three_pct_already_fractional_unchanged():
    t = enrich_team(_base_team(three_pt_pct=0.355))
    assert abs(t["three_pct"] - 0.355) < 1e-6


def test_blk_rate_mapped_to_block_rate():
    t = enrich_team(_base_team(blk_rate=12.5))
    assert "block_rate" in t
    assert t["block_rate"] == 12.5


def test_block_rate_not_overwritten_if_present():
    t = enrich_team(_base_team(blk_rate=12.5, block_rate=10.0))
    assert t["block_rate"] == 10.0


def test_experience_default_set():
    t = enrich_team(_base_team())
    assert "experience" in t
    assert 0.0 <= t["experience"] <= 1.0


def test_pedigree_normalizes_connecticut():
    t = enrich_team(_base_team(team="Connecticut"))
    assert t["pedigree_score"] == 0.90


def test_pedigree_normalizes_michigan_state():
    t = enrich_team(_base_team(team="Michigan St."))
    assert t["pedigree_score"] == 0.85


def test_coach_score_prefers_static_lookup_over_team_data():
    t = enrich_team(_base_team(team="Duke", coach="Jon Scheyer", coach_tourney_score=0.88))
    assert t["coach_tourney_score"] == 0.60


def test_coach_score_uses_low_fallback_when_static_lookup_missing():
    t = enrich_team(_base_team(team="Test Team", coach="Unlisted Coach", coach_tourney_score=0.88))
    assert t["coach_tourney_score"] == 0.3


def test_momentum_none_falls_back_to_recent_form():
    team = _base_team(momentum=None, adj_o=110.0, adj_d=100.0, adj_o_recent=112.0, adj_d_recent=98.0)
    bonus = calc_momentum_bonus(team)
    assert bonus > 0


def test_three_rate_default_set():
    t = enrich_team(_base_team())
    assert "three_rate" in t


def test_enrich_is_nondestructive():
    """enrich_team must not mutate the input dict."""
    original = _base_team(two_pt_pct=47.8)
    _ = enrich_team(original)
    assert original["two_pt_pct"] == 47.8


# ---------------------------------------------------------------------------
# _normalize_team_for_match
# ---------------------------------------------------------------------------

def test_normalize_unicode_dash():
    """Gardner\u2013Webb (en-dash) must normalize same as Gardner-Webb (ASCII hyphen)."""
    assert _normalize_team_for_match("Gardner\u2013Webb") == _normalize_team_for_match("Gardner-Webb")


def test_normalize_em_dash():
    """Em-dash must normalize same as hyphen."""
    assert _normalize_team_for_match("Some\u2014Team") == _normalize_team_for_match("Some-Team")


def test_normalize_nc_state_alias():
    assert _normalize_team_for_match("NC State") == "north carolina state"


def test_normalize_north_carolina_state_no_alias_conflict():
    """'North Carolina State' must map to same key as 'NC State'."""
    assert _normalize_team_for_match("North Carolina State") == _normalize_team_for_match("NC State")


def test_normalize_grambling_state():
    assert _normalize_team_for_match("Grambling State") == "grambling st"


def test_normalize_uconn():
    assert _normalize_team_for_match("UConn") == "connecticut"


def test_normalize_iowa_state_to_iowa_st():
    assert _normalize_team_for_match("Iowa State") == "iowa st"


def test_normalize_strips_periods():
    assert _normalize_team_for_match("St. John's") == "st johns"


# ---------------------------------------------------------------------------
# predict_game sanity checks
# ---------------------------------------------------------------------------

def _make_team(name, seed, adj_o, adj_d, barthag=0.75):
    return enrich_team({
        "team": name, "seed": seed,
        "adj_o": adj_o, "adj_d": adj_d, "adj_tempo": 68.0, "barthag": barthag,
    })


def test_predict_game_strong_vs_weak_favors_strong():
    strong = _make_team("Strong", 1, adj_o=125, adj_d=90, barthag=0.97)
    weak = _make_team("Weak", 16, adj_o=95, adj_d=115, barthag=0.10)
    result = predict_game(strong, weak, config=ModelConfig(num_sims=1))
    assert result["win_prob_a"] > 0.85, f"Expected strong favorite >85%, got {result['win_prob_a']:.3f}"


def test_predict_game_even_matchup_near_50():
    a = _make_team("TeamA", 5, adj_o=112, adj_d=102, barthag=0.75)
    b = _make_team("TeamB", 5, adj_o=112, adj_d=102, barthag=0.75)
    result = predict_game(a, b, config=ModelConfig(num_sims=1))
    assert 0.45 <= result["win_prob_a"] <= 0.55, f"Expected ~50%, got {result['win_prob_a']:.3f}"


def test_predict_game_prob_sums_to_one():
    a = _make_team("A", 3, adj_o=118, adj_d=98, barthag=0.88)
    b = _make_team("B", 6, adj_o=110, adj_d=104, barthag=0.72)
    result = predict_game(a, b, config=ModelConfig(num_sims=1))
    assert abs(result["win_prob_a"] + result["win_prob_b"] - 1.0) < 1e-6


def test_predict_game_returns_margin():
    a = _make_team("A", 1, adj_o=125, adj_d=90, barthag=0.96)
    b = _make_team("B", 8, adj_o=108, adj_d=106, barthag=0.65)
    result = predict_game(a, b, config=ModelConfig(num_sims=1))
    assert "predicted_margin" in result
    assert result["predicted_margin"] > 0  # strong team should be favored


def test_resolve_ff_pairs_uses_layout_matchups():
    qo = ["East", "West", "Midwest", "South"]
    assert resolve_ff_pairs(qo, [[0, 3], [1, 2]]) == [("East", "South"), ("West", "Midwest")]


def test_resolve_ff_pairs_falls_back_to_legacy_layout():
    qo = ["East", "West", "Midwest", "South"]
    assert resolve_ff_pairs(qo, None) == [("East", "South"), ("West", "Midwest")]


def test_final_four_analysis_infers_venue_without_region(monkeypatch):
    monkeypatch.setattr(
        "engine._load_venues",
        lambda year: {"F4": [39.76, -86.16], "city_labels": {"F4": "Indianapolis, IN"}},
    )
    a = _base_team(team="Duke", seed=1, location=[36.0, -78.9])
    b = _base_team(team="Florida", seed=1, location=[29.6, -82.3])
    result = get_matchup_analysis_display(
        a,
        b,
        year=2026,
        round_name="Final Four",
        config=ModelConfig(num_sims=1),
    )
    assert result["venue_city"] == "Indianapolis, IN"


def test_enrich_bracket_with_teams_overlays_canonical_team_fields():
    bracket = {"East": {1: {"team": "Duke", "seed": 1, "momentum": -0.875}}}
    teams_merged = {
        "duke": {
            "team": "Duke",
            "coach": "Jon Scheyer",
            "momentum": None,
            "adj_o": 120.0,
            "adj_d": 90.0,
            "adj_tempo": 67.0,
            "barthag": 0.98,
        }
    }
    enrich_bracket_with_teams(bracket, teams_merged)
    duke = bracket["East"][1]
    assert duke["coach"] == "Jon Scheyer"
    assert duke["momentum"] is None


def test_upset_tolerance_ignores_gated_coach_and_pedigree():
    favored = {"team": "Fav", "seed": 5, "coach_tourney_score": 0.1, "pedigree_score": 0.1}
    underdog = {"team": "Dog", "seed": 12, "coach_tourney_score": 1.0, "pedigree_score": 1.0}

    enabled = _calc_upset_tolerance_bonus(
        favored,
        underdog,
        margin=2.0,
        config=ModelConfig(
            num_sims=1,
            upset_tolerance_max_bonus=4.0,
            upset_spread_threshold=6.0,
            coach_tourney_max_bonus=1.5,
            pedigree_max_bonus=1.0,
        ),
    )
    disabled = _calc_upset_tolerance_bonus(
        favored,
        underdog,
        margin=2.0,
        config=ModelConfig(
            num_sims=1,
            upset_tolerance_max_bonus=4.0,
            upset_spread_threshold=6.0,
            coach_tourney_max_bonus=0.0,
            pedigree_max_bonus=0.0,
        ),
    )

    assert enabled < 0
    assert disabled == 0.0
