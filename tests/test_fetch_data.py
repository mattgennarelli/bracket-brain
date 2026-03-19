import json

from scripts import fetch_data


def test_build_merged_teams_preserves_existing_rosters_when_injuries_lack_roster(tmp_path, monkeypatch):
    torvik_path = tmp_path / "torvik_2026.json"
    torvik_path.write_text(json.dumps([{"team": "Duke", "adj_o": 120.0, "adj_d": 90.0, "adj_tempo": 70.0}]))

    injuries_path = tmp_path / "injuries_2026.json"
    injuries_path.write_text(
        json.dumps(
            {
                "Duke": {
                    "injuries": [{"player": "Caleb Foster", "status": "out"}],
                    "roster": [],
                }
            }
        )
    )

    merged_path = tmp_path / "teams_merged_2026.json"
    merged_path.write_text(
        json.dumps(
            [
                {
                    "team": "Duke",
                    "adj_o": 120.0,
                    "adj_d": 90.0,
                    "adj_tempo": 70.0,
                    "roster": [{"player": "caleb foster", "bpr": 5.0, "poss": 1000.0}],
                    "injuries": [],
                }
            ]
        )
    )

    monkeypatch.setattr(fetch_data, "DATA_DIR", str(tmp_path))

    rc = fetch_data.build_merged_teams(2026, skip_torvik_fetch=True)

    saved = json.loads(merged_path.read_text())
    assert rc == 0
    assert saved[0]["team"] == "Duke"
    assert saved[0]["roster"] == [{"player": "caleb foster", "bpr": 5.0, "poss": 1000.0}]
    assert saved[0]["injuries"][0]["player"] == "Caleb Foster"

