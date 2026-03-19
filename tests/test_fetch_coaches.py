from scripts.fetch_coaches import (
    _clean_coach_name,
    _extract_wikipedia_rows,
    _map_team_name,
    _rows_to_records,
    find_duplicate_teams,
)


def test_clean_coach_name_filters_vacant_and_interim():
    assert _clean_coach_name("Vacant") == ""
    assert _clean_coach_name("Mike Scott (interim)") == "Mike Scott"


def test_extract_wikipedia_rows_and_map_rows():
    html = """
    <table class="wikitable">
      <tr><th>Team</th><th>Conference</th><th>Current coach</th><th>Since</th></tr>
      <tr><td>Duke Blue Devils</td><td>ACC</td><td>Jon Scheyer</td><td>2022-23</td></tr>
      <tr><td>Arizona State Sun Devils</td><td>Big 12</td><td>Vacant</td><td>2026-27</td></tr>
    </table>
    """
    parsed = _extract_wikipedia_rows(html)
    rows, matched = _rows_to_records(parsed, teams_merged=["Duke", "Arizona State"])
    assert rows == [{"team": "Duke", "coach": "Jon Scheyer", "source": "wikipedia_current_coaches"}]
    assert matched == 1


def test_map_team_name_strips_mascot_suffix():
    assert _map_team_name("BYU Cougars", {"BYU"}) == "BYU"


def test_map_team_name_handles_aliases():
    assert _map_team_name("Miami RedHawks", {"Miami FL", "Miami OH"}) == "Miami OH"
    assert _map_team_name("California Baptist Lancers", {"Cal Baptist"}) == "Cal Baptist"
    assert _map_team_name("Kansas City Roos", {"Kansas", "Missouri-Kansas City"}) == "Missouri-Kansas City"


def test_find_duplicate_teams():
    rows = [{"team": "Kansas", "coach": "A"}, {"team": "Kansas", "coach": "B"}, {"team": "Duke", "coach": "C"}]
    assert find_duplicate_teams(rows) == {"Kansas": 2}
