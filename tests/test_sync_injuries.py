import json

from scripts import sync_injuries


def test_find_matching_team_keys_returns_aliases():
    injuries = {
        "Wisconsin": [],
        "Wisconsin Badgers": [],
        "Marquette": [],
    }
    matches = sync_injuries.find_matching_team_keys("Wisconsin", injuries)
    assert matches == ["Wisconsin", "Wisconsin Badgers"]


def test_find_matching_team_keys_does_not_match_prefix_only_names():
    injuries = {
        "Texas": [],
        "Texas Tech": [],
        "Texas Tech Red Raiders": [],
        "North Carolina": [],
        "N.C. State": [],
    }
    assert sync_injuries.find_matching_team_keys("Texas Tech", injuries) == ["Texas Tech", "Texas Tech Red Raiders"]
    assert sync_injuries.find_matching_team_keys("North Carolina", injuries) == ["North Carolina"]


def test_apply_overrides_removes_healthy_player_from_all_aliases(tmp_path, monkeypatch):
    injuries_path = tmp_path / "injuries_2026.json"
    injuries_path.write_text(
        json.dumps(
            {
                "Wisconsin": {"injuries": [{"player": "Jack Janicki", "status": "questionable"}], "roster": []},
                "Wisconsin Badgers": {
                    "injuries": [
                        {"player": "Jack Janicki", "status": "questionable"},
                        {"player": "Nolan Winter", "status": "questionable"},
                    ],
                    "roster": [],
                },
            }
        )
    )
    monkeypatch.setattr(sync_injuries, "INJURIES_PATH", str(injuries_path))

    changes = sync_injuries.apply_overrides(
        [{"team": "Wisconsin", "player": "Nolan Winter", "status": "healthy", "return_round": ""}]
    )
    saved = json.loads(injuries_path.read_text())

    assert changes == 1
    assert all(i["player"] != "Nolan Winter" for i in saved["Wisconsin Badgers"]["injuries"])
