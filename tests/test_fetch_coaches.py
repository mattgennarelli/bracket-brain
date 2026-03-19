from scripts.fetch_coaches import (
    _clean_coach_name,
    _extract_sports_reference_rows,
    _extract_wikipedia_rows,
    _map_team_name,
    _rows_to_records,
    _tourney_result_flags,
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
    assert len(rows) == 1
    assert rows[0]["team"] == "Duke"
    assert rows[0]["coach"] == "Jon Scheyer"
    assert rows[0]["source"] == "wikipedia_current_coaches"
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


def test_tourney_result_flags():
    assert _tourney_result_flags("") == {"ncaa": 0, "s16": 0, "ff": 0, "titles": 0}
    assert _tourney_result_flags("Lost First Round") == {"ncaa": 1, "s16": 0, "ff": 0, "titles": 0}
    assert _tourney_result_flags("Lost Regional Final") == {"ncaa": 1, "s16": 1, "ff": 0, "titles": 0}
    assert _tourney_result_flags("National Champions") == {"ncaa": 1, "s16": 1, "ff": 1, "titles": 1}


def test_extract_sports_reference_rows_from_comment_hidden_table():
    html = """
    <!--
    <table id="coaches">
      <thead>
        <tr>
          <th>Coach</th><th>School</th><th>Conference</th><th></th>
          <th>W</th><th>L</th><th>W-L%</th><th>AP Pre</th><th>AP Post</th><th>NCAA Tournament</th>
          <th></th><th>Since</th><th>W</th><th>L</th><th>W-L%</th><th>NCAA</th><th>S16</th><th>FF</th><th>Chmp</th>
          <th></th><th>W</th><th>L</th><th>W-L%</th><th>NCAA</th><th>S16</th><th>FF</th><th>Chmp</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>John Calipari *</th><td>Arkansas</td><td>SEC</td><td></td>
          <td>23</td><td>8</td><td>.742</td><td>14</td><td></td><td>Lost Regional Final</td>
          <td></td><td>2024-25</td><td>45</td><td>22</td><td>.672</td><td>5</td><td>3</td><td>1</td><td>0</td>
          <td></td><td>900</td><td>285</td><td>.759</td><td>24</td><td>16</td><td>6</td><td>1</td>
        </tr>
        <tr>
          <th>New Coach</th><td>Test State</td><td>WAC</td><td></td>
          <td>10</td><td>20</td><td>.333</td><td></td><td></td><td></td>
          <td></td><td>2025-26</td><td>10</td><td>20</td><td>.333</td><td>0</td><td>0</td><td>0</td><td>0</td>
          <td></td><td>10</td><td>20</td><td>.333</td><td>0</td><td>0</td><td>0</td><td>0</td>
        </tr>
      </tbody>
    </table>
    -->
    """
    rows = _extract_sports_reference_rows(html, 2026)
    assert len(rows) == 2
    cal = rows[0]
    assert cal["coach"] == "John Calipari"
    assert cal["team"] == "Arkansas"
    assert cal["career_ncaa_pre"] == 23
    assert cal["career_s16_pre"] == 15
    assert cal["career_ff_pre"] == 6
    assert cal["career_titles_pre"] == 1
    assert cal["school_ncaa_pre"] == 4
    assert cal["school_s16_pre"] == 2
    assert cal["school_ff_pre"] == 1
    assert cal["coach_resume_points_pre"] == 85
    assert cal["coach_tourney_score"] == 1.0
    assert rows[1]["coach_tourney_score"] == 0.0
