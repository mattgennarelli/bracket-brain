from scripts import save_bets, save_card


def test_save_bets_replaces_superseded_same_market_pick():
    existing = [
        {
            "date": "2026-03-23",
            "home_team": "Michigan St Spartans",
            "away_team": "Louisville Cardinals",
            "commence_time": "2026-03-23T18:45:00Z",
            "bet_type": "ml",
            "bet_side": "Louisville Cardinals",
            "generated_at": "2026-03-23T13:00:00Z",
            "result": None,
            "actual_score_home": None,
            "actual_score_away": None,
        }
    ]
    new = [
        {
            "date": "2026-03-23",
            "home_team": "Michigan St Spartans",
            "away_team": "Louisville Cardinals",
            "commence_time": "2026-03-23T18:45:00Z",
            "bet_type": "ml",
            "bet_side": "Michigan St Spartans",
            "generated_at": "2026-03-23T16:00:00Z",
            "result": None,
            "actual_score_home": None,
            "actual_score_away": None,
        }
    ]

    merged = save_bets._merge_market_picks(existing, new)

    assert len(merged) == 1
    assert merged[0]["bet_side"] == "Michigan St Spartans"


def test_save_card_preserves_started_market_when_new_side_arrives(monkeypatch):
    existing = [
        {
            "date": "2026-03-21",
            "home_team": "Michigan St Spartans",
            "away_team": "Louisville Cardinals",
            "commence_time": "2026-03-21T18:45:00Z",
            "bet_type": "spread",
            "bet_team": "Michigan St Spartans",
            "generated_at": "2026-03-21T16:00:00Z",
            "result": None,
            "actual_score_home": None,
            "actual_score_away": None,
        }
    ]
    new = [
        {
            "date": "2026-03-21",
            "home_team": "Michigan St Spartans",
            "away_team": "Louisville Cardinals",
            "commence_time": "2026-03-21T18:45:00Z",
            "bet_type": "spread",
            "bet_team": "Louisville Cardinals",
            "generated_at": "2026-03-21T19:15:00Z",
            "result": None,
            "actual_score_home": None,
            "actual_score_away": None,
        }
    ]

    monkeypatch.setattr(save_card, "_pick_has_started", lambda pick: True)

    merged = save_card._merge_market_picks(existing, new)

    assert len(merged) == 1
    assert merged[0]["bet_team"] == "Michigan St Spartans"
