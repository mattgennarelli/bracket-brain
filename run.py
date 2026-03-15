"""
run.py — Bracket Brain: Generate a complete 63-game bracket with picks and analysis.

Usage:
    python run.py                         # Run with defaults
    python run.py --sims 50000            # More Monte Carlo sims
    python run.py --bracket data/bracket_2026.json
    python run.py --upset 0.3             # Upset aggressiveness 0-1
"""

import json
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import re

from engine import (
    generate_bracket_picks, run_monte_carlo, load_bracket, analyze_matchup,
    REGIONS, FIRST_ROUND_MATCHUPS, DEFAULT_NUM_SIMS, SEED_WEIGHT,
    ModelConfig, DEFAULT_CONFIG,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def _year_from_bracket_path(path):
    m = re.search(r"bracket_(\d{4})", path)
    return int(m.group(1)) if m else 2026


def _find_team_seed(team_name, bracket):
    for region in bracket.values():
        for seed, team in region.items():
            if team["team"] == team_name:
                return seed
    return "?"


def _available_bracket_years():
    """Scan data/ for bracket_YYYY.json files and return sorted list of years."""
    years = []
    for fname in os.listdir(DATA_DIR):
        m = re.match(r"bracket_(\d{4})\.json$", fname)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def _format_bet_time(iso_str):
    """Format ISO commence_time to local time e.g. 12:30pm."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        local = dt.astimezone()
        h = local.hour % 12 or 12
        suffix = "pm" if local.hour >= 12 else "am"
        return f"{h}:{local.minute:02d}{suffix}"
    except Exception:
        return iso_str[:16] if iso_str else ""


def _build_bets_table(best_bets):
    """Build HTML table of best bets sorted by start time."""
    if not best_bets:
        return '''
    <div class="bets-empty">
      <div class="icon">📊</div>
      <h3>No betting picks yet</h3>
      <p>Run <code>python run.py --best-bets</code> with <code>ODDS_API_KEY</code> set to fetch today's games and model edges.</p>
      <p><a href="https://the-odds-api.com" target="_blank">Get a free API key</a> (500 req/month)</p>
    </div>'''
    rows = []
    for i, b in enumerate(best_bets, 1):
        time_str = _format_bet_time(b.get("commence_time", ""))
        matchup = f"{b.get('away_team', '')} @ {b.get('home_team', '')}"
        bt = b.get("bet_type", "")
        if bt == "ml":
            pick = b.get("bet_side", "")
            odds = b.get("bet_odds")
            odds_str = f"{'+' if odds and odds > 0 else ''}{int(odds)}" if odds else "—"
            edge = f"{b.get('edge', 0)*100:+.1f}%"
        elif bt == "spread":
            pick = f"{b.get('bet_team', '')} {b.get('bet_spread', 0):+.1f}"
            odds = b.get("bet_odds")
            odds_str = f"{'+' if odds and odds > 0 else ''}{int(odds)}" if odds else "—"
            edge = f"{b.get('cover_margin', 0):.1f} pts"
        else:
            pick = f"{b.get('bet_side', '')} {b.get('vegas_total', 0):.1f}"
            odds_str = "—"
            edge = f"{b.get('edge', 0):+.1f} pts"
        stars = b.get("stars", "")
        rows.append(
            f'<tr><td class="bet-time">{time_str}</td><td class="bet-matchup">{matchup}</td>'
            f'<td class="bet-pick">{pick}</td><td>{odds_str}</td>'
            f'<td class="bet-edge">{edge}</td><td class="bet-stars">{stars}</td></tr>'
        )
    return f'''
    <table class="bets-table">
      <thead><tr><th>Time</th><th>Matchup</th><th>Pick</th><th>Odds</th><th>Edge</th><th></th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>'''


def generate_html(bracket_result, mc_results, bracket, config, num_sims,
                  upset_aggression=0.0, quadrant_order=None, year=2026, best_bets=None):
    """Generate an interactive bracket HTML page."""
    if quadrant_order is None:
        quadrant_order = REGIONS[:4]
    available_years = _available_bracket_years()
    picks = bracket_result["picks"]
    champion = bracket_result["champion"]
    final_four = bracket_result["final_four"]
    biggest_upsets = bracket_result["biggest_upsets"]

    champ_probs = mc_results["champion_probs"]
    ff_probs = mc_results["final_four_probs"]
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    team_stats = {}
    for region_name, region_data in bracket.items():
        for seed, team in region_data.items():
            ts = {
                "team": team["team"],
                "seed": team.get("seed", seed),
                "region": region_name,
                "adj_o": team.get("adj_o", 85),
                "adj_d": team.get("adj_d", 112),
                "adj_tempo": team.get("adj_tempo", 64),
                "barthag": team.get("barthag", 0.05),
            }
            for extra in ("sos", "wab", "elite_sos", "qual_o", "qual_d",
                          "qual_barthag", "conf_adj_o", "conf_adj_d",
                          "to_rate", "orb_rate", "ft_pct", "three_rate", "three_pct"):
                if team.get(extra) is not None:
                    ts[extra] = team[extra]
            team_stats[team["team"]] = ts

    picks_json = json.dumps(picks)
    team_stats_json = json.dumps(team_stats)
    config_json = json.dumps({
        "seed_weight": config.seed_weight,
        "base_scoring_stdev": config.base_scoring_stdev,
        "national_avg_efficiency": config.national_avg_efficiency,
        "national_avg_tempo": config.national_avg_tempo,
    })

    bracket_structure = {}
    for region_name, region_data in bracket.items():
        teams = []
        for sa, sb in FIRST_ROUND_MATCHUPS:
            if sa in region_data:
                teams.append({"team": region_data[sa]["team"], "seed": sa})
            if sb in region_data:
                teams.append({"team": region_data[sb]["team"], "seed": sb})
        bracket_structure[region_name] = teams
    bracket_structure_json = json.dumps(bracket_structure)

    max_champ = max(champ_probs.values()) if champ_probs else 1
    champ_rows = ""
    for i, (team, prob) in enumerate(list(champ_probs.items())[:16]):
        seed = _find_team_seed(team, bracket)
        bar_w = prob / max_champ * 100
        champ_rows += (f'<tr><td class="rank">{i+1}</td>'
                       f'<td><span class="seed-badge">{seed}</span>{team}</td>'
                       f'<td class="bar-cell"><div class="bar" style="width:{bar_w:.0f}%"></div></td>'
                       f'<td class="pct">{prob*100:.1f}%</td></tr>\n')

    max_ff = max(ff_probs.values()) if ff_probs else 1
    ff_rows = ""
    for i, (team, prob) in enumerate(list(ff_probs.items())[:16]):
        seed = _find_team_seed(team, bracket)
        bar_w = prob / max_ff * 100
        ff_rows += (f'<tr><td class="rank">{i+1}</td>'
                    f'<td><span class="seed-badge">{seed}</span>{team}</td>'
                    f'<td class="bar-cell"><div class="bar" style="width:{bar_w:.0f}%"></div></td>'
                    f'<td class="pct">{prob*100:.1f}%</td></tr>\n')

    conf_counts = {"lock": 0, "strong": 0, "lean": 0, "tossup": 0}
    for p in picks:
        conf_counts[p["confidence"]] = conf_counts.get(p["confidence"], 0) + 1

    quadrant_order_json = json.dumps(quadrant_order)
    best_bets = best_bets or []
    best_bets_json = json.dumps(best_bets)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bracket Brain — March Madness Bracket Picker</title>
<style>
:root {{
  --bg: #f0f2f5; --surface: #ffffff; --surface2: #f7f8fa;
  --border: #e2e5ea; --text: #1a1e2c; --muted: #6b7280;
  --primary: #1a6dcc; --primary-bg: rgba(26,109,204,.08);
  --green: #1a8754; --green-bg: rgba(26,135,84,.08);
  --red: #c62828; --red-bg: rgba(198,40,40,.06);
  --gold: #d97706; --connector: #cbd5e1;
}}
*{{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background:var(--bg); color:var(--text); overflow-x:auto; line-height:1.4; }}

.header {{ text-align:center; padding:20px 20px 14px; background:var(--surface);
  border-bottom:1px solid var(--border); }}
.header h1 {{ font-size:1.6rem; font-weight:800; letter-spacing:-.02em; }}
.header h1 .a {{ color:var(--primary); }}
.header .sub {{ color:var(--muted); font-size:.82rem; margin:2px 0 8px; }}
.header .meta {{ display:inline-flex; gap:14px; font-size:.72rem; color:var(--muted);
  background:var(--surface2); padding:5px 14px; border-radius:12px; border:1px solid var(--border); }}

.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:center;
  padding:10px 20px; background:var(--surface); border-bottom:1px solid var(--border);
  position:sticky; top:0; z-index:100; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
.slider-group {{ display:flex; align-items:center; gap:6px; }}
.slider-group label {{ font-size:.78rem; color:var(--muted); font-weight:500; }}
.slider-group input[type=range] {{ width:110px; accent-color:var(--primary); }}
.slider-group .val {{ font-size:.78rem; font-weight:700; color:var(--primary); min-width:28px; }}
.slider-label {{ font-size:.68rem; color:var(--muted); font-style:italic; }}
.btn {{ padding:6px 14px; border-radius:6px; border:1px solid var(--border); background:var(--surface);
  color:var(--text); cursor:pointer; font-size:.78rem; font-weight:600; transition:all .15s; }}
.btn:hover {{ border-color:var(--primary); color:var(--primary); }}
.btn.primary {{ background:var(--primary); border-color:var(--primary); color:#fff; }}
.btn.primary:hover {{ background:#155db0; }}
.champ-display {{ margin-left:auto; font-size:.82rem; font-weight:500; }}
.champ-display .champ-name {{ color:var(--gold); font-weight:800; font-size:.95rem; }}

.bracket-wrap {{ display:flex; flex-direction:column; align-items:center; padding:16px 8px; min-width:1400px; }}
.bracket-top, .bracket-bottom {{ display:flex; gap:0; justify-content:center; width:100%; }}
.bracket-bottom {{ margin-top:12px; }}
.region-bracket {{ flex:1; max-width:660px; overflow:visible; }}
.region-label {{ text-align:center; font-size:.78rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.08em; color:var(--primary); padding:2px 0 6px; }}

.rounds {{ display:flex; gap:0; align-items:stretch; }}
.region-bracket.flipped > .rounds {{ flex-direction:row-reverse; }}
.round {{ display:flex; flex-direction:column; justify-content:space-around;
  min-width:128px; max-width:155px; flex:1; padding:0 14px 0 2px; }}
.round.round-last {{ padding-right:2px; }}
.region-bracket.flipped .round {{ padding:0 2px 0 14px; }}
.region-bracket.flipped .round.round-last {{ padding-left:2px; }}
.round-header {{ text-align:center; font-size:.62rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); padding:1px 0 4px; }}

.game-pair {{ display:flex; flex-direction:column; justify-content:center; position:relative; flex:1; gap:2px; }}
.round:not(.round-last) .game-pair::after {{
  content:''; position:absolute; right:-14px; top:25%; bottom:25%; width:14px;
  border:1px solid var(--connector); border-left:none; border-radius:0 3px 3px 0; }}
.round:not(.round-last) .game-pair::before {{
  content:''; position:absolute; right:-14px; top:50%; width:14px;
  border-top:1px solid var(--connector); transform:translateX(14px); }}
.region-bracket.flipped .round:not(.round-last) .game-pair::after {{
  right:auto; left:-14px; border:1px solid var(--connector); border-right:none;
  border-radius:3px 0 0 3px; }}
.region-bracket.flipped .round:not(.round-last) .game-pair::before {{
  right:auto; left:-14px; transform:translateX(-14px); }}

.game {{ margin:1px 0; position:relative; cursor:pointer; }}
.team-slot {{ display:flex; align-items:center; gap:4px; padding:3px 6px; position:relative;
  background:var(--surface); border:1px solid var(--border); cursor:pointer;
  transition:all .12s; font-size:.72rem; min-height:22px; overflow:hidden; }}
.team-slot:first-child {{ border-radius:3px 3px 0 0; border-bottom:none; }}
.team-slot:last-of-type {{ border-radius:0 0 3px 3px; }}
.team-slot:hover {{ border-color:var(--primary); z-index:2; background:var(--primary-bg); }}
.team-slot.picked {{ background:var(--green-bg); border-color:var(--green); }}
.team-slot.picked .tm {{ color:var(--green); font-weight:700; }}
.team-slot.locked {{ background:rgba(26,135,84,.12); }}
.team-slot.locked .tm::after {{ content:' \\1F512'; font-size:.5rem; }}
.team-slot .sd {{ color:var(--muted); font-size:.62rem; font-weight:700; min-width:14px;
  text-align:right; flex-shrink:0; }}
.team-slot .tm {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:500; min-width:0; }}
.team-slot.empty {{ opacity:.35; cursor:default; }}
.team-slot.upset-pick {{ border-color:var(--red); }}
.team-slot.upset-pick.picked {{ background:var(--red-bg); }}
.team-slot.upset-pick.picked .tm {{ color:var(--red); }}
.team-slot .upset-badge {{ flex-shrink:0; font-size:.5rem; font-weight:600; padding:1px 3px;
  border-radius:2px; margin-left:2px; letter-spacing:.02em; }}
.team-slot .upset-badge.strong {{ background:var(--red); color:#fff; }}
.team-slot .upset-badge.mild {{ background:var(--gold); color:var(--text); }}
.team-slot .upset-badge.blowout {{ background:var(--green); color:#fff; }}
.game.has-alert .info-btn {{ border-color:var(--primary); color:var(--primary); opacity:.9; }}
.region-bracket.flipped .team-slot {{ flex-direction:row-reverse; text-align:right; }}
.region-bracket.flipped .team-slot .sd {{ text-align:left; }}

.ff-center {{ display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-width:200px; gap:8px; padding:10px 8px; }}
.ff-center .game {{ width:180px; }}
.ff-center .champ-banner {{ text-align:center; margin-top:10px; padding:8px; }}
.ff-center .champ-banner .trophy {{ font-size:1.8rem; }}
.ff-center .champ-banner .champ-team {{ font-size:1rem; font-weight:800; color:var(--gold); }}
.ff-label {{ font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
  color:var(--primary); text-align:center; margin-bottom:2px; }}

.info-btn {{ position:absolute; top:1px; right:1px; width:18px; height:18px; border-radius:50%;
  background:var(--surface2); border:1px solid var(--border); color:var(--muted); font-size:.6rem;
  cursor:pointer; display:flex; align-items:center; justify-content:center; z-index:10;
  opacity:.75; transition:opacity .15s; line-height:1; pointer-events:auto; }}
.info-btn:hover {{ opacity:1; border-color:var(--primary); color:var(--primary); }}
.region-bracket.flipped .info-btn {{ right:auto; left:1px; }}

.analysis-overlay {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.3); z-index:199; }}
.analysis-overlay.show {{ display:block; }}
.analysis-panel {{ display:none; position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
  background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px;
  width:460px; max-width:95vw; max-height:85vh; overflow-y:auto; z-index:200;
  box-shadow:0 20px 60px rgba(0,0,0,.12); }}
.analysis-panel.show {{ display:block; }}
.analysis-panel h3 {{ font-size:1rem; margin-bottom:14px; font-weight:700; }}
.prob-bar-wrap {{ display:flex; height:6px; border-radius:3px; overflow:hidden; margin:8px 0 12px;
  background:var(--surface2); }}
.prob-bar-a {{ background:var(--green); border-radius:3px 0 0 3px; transition:width .3s; }}
.prob-bar-b {{ background:var(--red); border-radius:0 3px 3px 0; transition:width .3s; }}
.matchup-line {{ display:flex; justify-content:space-between; align-items:center; padding:6px 0;
  border-bottom:1px solid var(--border); font-size:.88rem; }}
.team-side {{ font-weight:600; }}
.team-side.fav {{ color:var(--green); }}
.stat-row {{ display:flex; justify-content:space-between; font-size:.8rem; padding:4px 0;
  color:var(--muted); }}
.stat-row .val {{ color:var(--text); font-weight:600; }}
.matchup-alert {{ border-radius:6px; }}
.insight {{ font-size:.82rem; color:var(--muted); margin:12px 0 8px; line-height:1.5; padding:10px 12px;
  background:var(--surface2); border-radius:8px; border:1px solid var(--border); }}
.factors {{ list-style:none; padding:0; }}
.factors li {{ font-size:.8rem; color:var(--muted); padding:3px 0; }}
.factors li::before {{ content:'\\2022'; color:var(--primary); margin-right:6px; }}
.close-btn {{ position:absolute; top:12px; right:16px; background:none; border:none;
  color:var(--muted); font-size:1.3rem; cursor:pointer; }}
.close-btn:hover {{ color:var(--text); }}
.conf-badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:.68rem;
  font-weight:700; margin-left:8px; }}
.conf-lock {{ background:rgba(26,135,84,.1); color:var(--green); }}
.conf-strong {{ background:rgba(26,109,204,.1); color:var(--primary); }}
.conf-lean {{ background:rgba(217,119,6,.1); color:var(--gold); }}
.conf-tossup {{ background:rgba(198,40,40,.08); color:var(--red); }}

.odds-section {{ max-width:720px; margin:0 auto; padding:0 20px 20px; }}
.odds-panel {{ background:var(--surface); border:1px solid var(--border); border-radius:10px;
  margin:16px 0; padding:18px; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
.odds-panel h3 {{ font-size:.9rem; font-weight:700; margin-bottom:10px; }}
.odds-table {{ width:100%; border-collapse:collapse; }}
.odds-table td {{ padding:5px 8px; font-size:.78rem; border-bottom:1px solid var(--border); vertical-align:middle; }}
.odds-table .rank {{ color:var(--muted); width:24px; font-weight:600; }}
.odds-table .pct {{ text-align:right; font-weight:700; font-variant-numeric:tabular-nums; width:50px; }}
.odds-table .bar-cell {{ width:100px; padding:5px 8px; }}
.bar {{ height:6px; border-radius:3px; background:var(--primary); opacity:.6; min-width:2px; }}
.seed-badge {{ display:inline-block; background:var(--surface2); color:var(--muted); font-size:.62rem;
  font-weight:700; padding:1px 4px; border-radius:3px; margin-right:4px; border:1px solid var(--border); }}

.footer {{ text-align:center; padding:18px; color:var(--muted); font-size:.72rem;
  border-top:1px solid var(--border); background:var(--surface); }}

.tabs {{ display:flex; gap:0; padding:0 20px; background:var(--surface); border-bottom:1px solid var(--border);
  position:sticky; top:0; z-index:99; }}
.tab {{ padding:12px 20px; font-size:.88rem; font-weight:600; color:var(--muted); cursor:pointer;
  border-bottom:2px solid transparent; margin-bottom:-1px; transition:all .15s; }}
.tab:hover {{ color:var(--text); }}
.tab.active {{ color:var(--primary); border-bottom-color:var(--primary); }}
.tab-pane {{ display:none; }}
.tab-pane.active {{ display:block; }}
.bets-panel {{ max-width:720px; margin:0 auto; padding:24px 20px; }}
.bets-panel h2 {{ font-size:1.1rem; margin-bottom:16px; }}
.bets-table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
.bets-table th {{ text-align:left; padding:8px 10px; border-bottom:2px solid var(--border); color:var(--muted); font-weight:600; }}
.bets-table td {{ padding:10px; border-bottom:1px solid var(--border); vertical-align:middle; }}
.bets-table tr:hover {{ background:var(--surface2); }}
.bet-time {{ font-variant-numeric:tabular-nums; color:var(--muted); white-space:nowrap; }}
.bet-matchup {{ font-weight:600; }}
.bet-pick {{ color:var(--green); font-weight:700; }}
.bet-edge {{ color:var(--primary); font-weight:600; }}
.bet-stars {{ color:var(--gold); letter-spacing:1px; }}
.bets-empty {{ background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:32px 24px; margin:0 auto; max-width:420px; text-align:center; }}
.bets-empty .icon {{ font-size:2rem; margin-bottom:12px; opacity:.5; }}
.bets-empty h3 {{ font-size:.95rem; margin:0 0 8px; color:var(--text); }}
.bets-empty p {{ margin:6px 0; font-size:.85rem; color:var(--muted); line-height:1.5; }}
.bets-empty code {{ background:var(--surface2); padding:3px 8px; border-radius:4px; font-size:.8rem; border:1px solid var(--border); }}
.bets-empty a {{ color:var(--primary); text-decoration:none; }}
.bets-empty a:hover {{ text-decoration:underline; }}

@media (max-width:1400px) {{
  .bracket-wrap {{ min-width:100%; }}
  .bracket-top, .bracket-bottom {{ flex-direction:column; align-items:center; }}
  .region-bracket {{ max-width:100%; }}
  .ff-center {{ flex-direction:row; flex-wrap:wrap; justify-content:center; }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>Bracket <span class="a">Brain</span></h1>
  <p class="sub">Interactive March Madness Bracket Picker</p>
  <div class="meta">
    <span>{num_sims:,} Monte Carlo sims</span>
    <span>Calibrated on 945 games</span>
    <span>{timestamp}</span>
  </div>
</div>

<div class="tabs">
  <div class="tab active" data-tab="bracket" onclick="switchTab('bracket')">Bracket</div>
  <div class="tab" data-tab="bets" onclick="switchTab('bets')">Top Bets</div>
</div>

<div class="controls" id="controls">
  <div class="slider-group">
    <label>Year:</label>
    <select id="year-select" onchange="onYearChange(this.value)" style="font-size:.78rem;padding:3px 6px;border-radius:4px;border:1px solid var(--border);font-weight:600;">
      {''.join(f'<option value="{y}"{"selected" if y == year else ""}>{y}</option>' for y in available_years)}
    </select>
  </div>
  <div class="slider-group">
    <label>Chaos:</label>
    <input type="range" id="upset-slider" min="0" max="100" value="{int(upset_aggression*100)}"
      oninput="onUpsetChange(this.value)">
    <span class="val" id="upset-val">{int(upset_aggression*100)}%</span>
    <span class="slider-label" id="upset-label">All Chalk</span>
  </div>
  <button class="btn primary" onclick="simulateAll()">Simulate</button>
  <button class="btn" onclick="resetPicks()">Reset</button>
  <div class="champ-display">
    Champion: <span class="champ-name" id="champ-name">{champion or 'TBD'}</span>
  </div>
</div>

<div class="tab-pane active" id="tab-bracket">
  <div class="bracket-wrap" id="bracket-wrap"></div>
  <div class="odds-section">
    <div class="odds-panel">
      <h3>Championship Odds ({num_sims:,} sims)</h3>
      <table class="odds-table">{champ_rows}</table>
    </div>
    <div class="odds-panel">
      <h3>Final Four Odds</h3>
      <table class="odds-table">{ff_rows}</table>
    </div>
  </div>
</div>

<div class="tab-pane" id="tab-bets">
  <div class="bets-panel">
    <h2>Today's Top Betting Picks</h2>
    {_build_bets_table(best_bets)}
  </div>
</div>

<div class="footer">
  <p>Built by Matt Gennarelli &middot; Calibrated on 945 tournament games (2010-2025) &middot; Data: Bart Torvik T-Rank</p>
</div>

<div class="analysis-overlay" id="analysis-overlay" onclick="closeAnalysis()"></div>
<div class="analysis-panel" id="analysis-panel">
  <button class="close-btn" onclick="closeAnalysis()">&times;</button>
  <div id="analysis-content"></div>
</div>

<script>
const PICKS = {picks_json};
const TEAM_STATS = {team_stats_json};
const CONFIG = {config_json};
const BRACKET = {bracket_structure_json};
const REGIONS = {json.dumps(REGIONS)};
const QUADRANT = {quadrant_order_json};

const BAD_STATS = {{ adj_o:85, adj_d:112, adj_tempo:64, seed:16 }};
const DISPLAY_NAMES = {{ "Texas A&M-Corpus Christi":"TAMU-CC", "Texas A&M Corpus Christi":"TAMU-CC", "Texas A&M Corpus Chris":"TAMU-CC", "Texas A&M\\u2013Corpus Christi":"TAMU-CC" }};
let lockedPicks = {{}};
let simPicks = {{}};
let bracketState = {{}};
let upsetAggression = {upset_aggression};
let _simTimer = null;
let gameAlerts = {{}};

const SEED_EXPECTED_DIR_MARGINS = {{
  '1-16':22.6,'1-8':6.6,'1-9':9.4,'2-15':13.5,'2-7':3.8,'2-10':4.5,
  '3-14':10.4,'3-6':3.6,'3-11':1.1,'4-13':6.4,'4-5':3.4,'5-12':2.4,
  '6-11':-0.3,'7-10':2.0,'8-9':0.4,'1-4':4.0,'1-5':4.8,'2-3':1.6,
  '1-2':2.4,'1-3':6.6,'2-6':5.1,'3-7':0
}};

function computeUpsetAlert(seedA, seedB, margin, probA) {{
  if (!seedA || !seedB || seedA === seedB) return null;
  const hi = Math.min(seedA, seedB), lo = Math.max(seedA, seedB);
  const diff = lo - hi;
  if (diff < 2) return null;
  const key = hi + '-' + lo;
  const expected = SEED_EXPECTED_DIR_MARGINS[key] ?? diff * 1.2;
  const favoriteMargin = seedA < seedB ? margin : -margin;
  const base = Math.max(Math.abs(expected), 1);
  const gap = expected - favoriteMargin;
  const gapPct = gap / base;
  const underdogProb = 1 - Math.max(probA, 1 - probA);
  if (gapPct > 0.45 && gap > 5 && underdogProb > 0.38 && diff >= 3) return {{ level:'strong', icon:'\\uD83D\\uDD25', reason:'(' + hi + ') seed favored by only ' + favoriteMargin.toFixed(1) + ' pts vs expected ' + expected.toFixed(1) + ' (' + (gapPct*100).toFixed(0) + '% tighter). Upset danger.', badgeOnUnderdog: true }};
  if (gapPct > 0.35 && gap > 3.5 && underdogProb > 0.33 && diff >= 3) return {{ level:'mild', icon:'\\u26A0\\uFE0F', reason:'Margin (' + favoriteMargin.toFixed(1) + ') ' + (gapPct*100).toFixed(0) + '% tighter than typical (' + expected.toFixed(1) + ') for ' + hi + '-' + lo + ' seeds.', badgeOnUnderdog: true }};
  const excess = favoriteMargin - expected;
  const excessPct = excess / base;
  if (excessPct > 0.6 && excess > 8 && diff >= 4) return {{ level:'blowout', icon:'\\uD83D\\uDCAA', reason:'Projected margin (' + favoriteMargin.toFixed(1) + ') ' + (excessPct*100).toFixed(0) + '% above ' + hi + '-' + lo + ' norm (' + expected.toFixed(1) + '). Dominant.', badgeOnUnderdog: false }};
  return null;
}}

function getStats(name, fallbackSeed) {{
  if (TEAM_STATS[name]) return TEAM_STATS[name];
  return Object.assign({{}}, BAD_STATS, {{ team:name, seed:fallbackSeed||16 }});
}}

function seedWinProb(sA,sB) {{ return 1/(1+Math.exp(-.145*(sB-sA))); }}
function blendProbs(p1,p2,w2) {{
  const logit=p=>Math.log(Math.max(.001,Math.min(.999,p))/(1-Math.max(.001,Math.min(.999,p))));
  return 1/(1+Math.exp(-((1-w2)*logit(p1)+w2*logit(p2))));
}}
function erfApprox(x) {{
  const a1=.254829592,a2=-.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=.3275911;
  const s=x<0?-1:1,t=1/(1+p*Math.abs(x));
  return s*(1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x));
}}
function predictGame(tA,tB) {{
  const avg=CONFIG.national_avg_efficiency, avgT=CONFIG.national_avg_tempo;
  const oA=tA.adj_o||BAD_STATS.adj_o, dA=tA.adj_d||BAD_STATS.adj_d, temA=tA.adj_tempo||BAD_STATS.adj_tempo;
  const oB=tB.adj_o||BAD_STATS.adj_o, dB=tB.adj_d||BAD_STATS.adj_d, temB=tB.adj_tempo||BAD_STATS.adj_tempo;
  const poss=(temA*temB)/avgT;
  const sA=(oA*dB)/avg*poss/100, sB=(oB*dA)/avg*poss/100;
  const margin=sA-sB;
  const stdev=CONFIG.base_scoring_stdev*Math.sqrt(poss/avgT);
  const effP=stdev===0?.5:.5*(1+erfApprox(margin/stdev/Math.SQRT2));
  const prob=blendProbs(effP, seedWinProb(tA.seed||16,tB.seed||16), CONFIG.seed_weight);
  return {{ probA:prob, margin }};
}}
function shouldPickUpset(prob,sA,sB,agg) {{
  if (agg<=0) return prob>=.5;
  const boost=Math.min(Math.abs(sA-sB)/15,1)*.3;
  return Math.random() < (prob+(prob<.5?1:-1)*agg*boost);
}}

function getNextGameId(gameId) {{
  if (gameId==='FF-2-0') return null;
  if (gameId==='FF-4-0'||gameId==='FF-4-1') return 'FF-2-0';
  const parts=gameId.split('-');
  const gi=parseInt(parts.pop()), round=parseInt(parts.pop()), region=parts.join('-');
  const next={{64:32,32:16,16:8}}[round];
  if (next) return `${{region}}-${{next}}-${{Math.floor(gi/2)}}`;
  if (round===8) {{
    if (region===QUADRANT[0]||region===QUADRANT[3]) return 'FF-4-0';
    if (region===QUADRANT[1]||region===QUADRANT[2]) return 'FF-4-1';
  }}
  return null;
}}

function recomputeBracket() {{
  const newState = {{}};
  const regionWinners = {{}};
  REGIONS.forEach(region => {{
    const teams = BRACKET[region];
    if (!teams) return;
    let prev = [];
    for (let gi=0; gi<8; gi++) {{
      const gid = `${{region}}-64-${{gi}}`;
      const tA = teams[gi*2], tB = teams[gi*2+1];
      if (!tA||!tB) {{ prev.push(null); continue; }}
      const pick = lockedPicks[gid] || simPicks[gid] || null;
      const sA = getStats(tA.team, tA.seed), sB = getStats(tB.team, tB.seed);
      const pred = predictGame(sA, sB);
      newState[gid] = {{ teamA:tA.team, teamB:tB.team, seedA:tA.seed, seedB:tB.seed,
                         pick, probA:pred.probA, margin:pred.margin, isLocked:!!lockedPicks[gid] }};
      prev.push(pick ? (pick===tA.team ? tA : tB) : null);
    }}
    [[32,4],[16,2],[8,1]].forEach(([roundOf, n]) => {{
      let next = [];
      for (let gi=0; gi<n; gi++) {{
        const gid = `${{region}}-${{roundOf}}-${{gi}}`;
        const wA = prev[gi*2], wB = prev[gi*2+1];
        const teamA = wA ? wA.team : null, teamB = wB ? wB.team : null;
        const seedA = wA ? wA.seed : null, seedB = wB ? wB.seed : null;
        let pick = lockedPicks[gid] || simPicks[gid] || null;
        if (pick && pick !== teamA && pick !== teamB) {{
          delete lockedPicks[gid]; delete simPicks[gid]; pick = null;
        }}
        let probA = null, margin = 0;
        if (teamA && teamB) {{
          const sA = getStats(teamA, seedA), sB = getStats(teamB, seedB);
          const pred = predictGame(sA, sB);
          probA = pred.probA; margin = pred.margin;
        }}
        newState[gid] = {{ teamA, teamB, seedA, seedB, pick, probA, margin, isLocked:!!lockedPicks[gid] }};
        next.push(pick ? (pick===teamA ? wA : wB) : null);
      }}
      prev = next;
    }});
    const e8 = newState[`${{region}}-8-0`];
    if (e8 && e8.pick) {{
      const seed = e8.pick===e8.teamA ? e8.seedA : e8.seedB;
      regionWinners[region] = getStats(e8.pick, seed);
    }}
  }});
  const ffPairs = [['FF-4-0', QUADRANT[0], QUADRANT[3]], ['FF-4-1', QUADRANT[1], QUADRANT[2]]];
  const ffWinners = {{}};
  ffPairs.forEach(([gid, rA, rB]) => {{
    const tA = regionWinners[rA] || null, tB = regionWinners[rB] || null;
    let pick = lockedPicks[gid] || simPicks[gid] || null;
    if (pick && (!tA || !tB || (pick !== tA.team && pick !== tB.team))) {{
      delete lockedPicks[gid]; delete simPicks[gid]; pick = null;
    }}
    let probA = null, margin = 0;
    if (tA && tB) {{ const pred = predictGame(tA, tB); probA = pred.probA; margin = pred.margin; }}
    newState[gid] = {{ teamA:tA?tA.team:null, teamB:tB?tB.team:null,
      seedA:tA?tA.seed:null, seedB:tB?tB.seed:null, pick, probA, margin, isLocked:!!lockedPicks[gid] }};
    if (pick) ffWinners[gid] = pick===tA?.team ? tA : tB;
  }});
  const cA = ffWinners['FF-4-0']||null, cB = ffWinners['FF-4-1']||null;
  const s0 = newState['FF-4-0'], s1 = newState['FF-4-1'];
  const champSeedA = cA && s0?.pick ? (s0.pick === s0.teamA ? s0.seedA : s0.seedB) : (cA?.seed ?? null);
  const champSeedB = cB && s1?.pick ? (s1.pick === s1.teamA ? s1.seedA : s1.seedB) : (cB?.seed ?? null);
  let cPick = lockedPicks['FF-2-0'] || simPicks['FF-2-0'] || null;
  if (cPick && (!cA || !cB || (cPick !== cA.team && cPick !== cB.team))) {{
    delete lockedPicks['FF-2-0']; delete simPicks['FF-2-0']; cPick = null;
  }}
  let cProbA = null, cMargin = 0;
  if (cA && cB) {{ const pred = predictGame(cA, cB); cProbA = pred.probA; cMargin = pred.margin; }}
  newState['FF-2-0'] = {{ teamA:cA?cA.team:null, teamB:cB?cB.team:null,
    seedA:champSeedA, seedB:champSeedB,
    pick:cPick, probA:cProbA, margin:cMargin, isLocked:!!lockedPicks['FF-2-0'] }};
  bracketState = newState;
  gameAlerts = {{}};
  Object.keys(newState).forEach(gid => {{
    const st = newState[gid];
    if (st.teamA && st.teamB && st.probA !== null) {{
      gameAlerts[gid] = computeUpsetAlert(st.seedA, st.seedB, st.margin, st.probA);
    }}
  }});
  renderBracket();
}}

function simulateAll() {{
  const agg = upsetAggression;
  REGIONS.forEach(region => {{
    const teams = BRACKET[region];
    if (!teams) return;
    let prev = [];
    for (let gi=0; gi<8; gi++) {{
      const gid = `${{region}}-64-${{gi}}`;
      const tA = teams[gi*2], tB = teams[gi*2+1];
      if (!tA||!tB) {{ prev.push(null); continue; }}
      if (!lockedPicks[gid] && !simPicks[gid]) {{
        const sA = getStats(tA.team, tA.seed), sB = getStats(tB.team, tB.seed);
        const pred = predictGame(sA, sB);
        simPicks[gid] = shouldPickUpset(pred.probA, tA.seed, tB.seed, agg) ? tA.team : tB.team;
      }}
      const pick = lockedPicks[gid] || simPicks[gid];
      prev.push(pick === tA.team ? tA : tB);
    }}
    [[32,4],[16,2],[8,1]].forEach(([roundOf, n]) => {{
      let next = [];
      for (let gi=0; gi<n; gi++) {{
        const gid = `${{region}}-${{roundOf}}-${{gi}}`;
        const wA = prev[gi*2], wB = prev[gi*2+1];
        if (!wA||!wB) {{ next.push(wA||wB); continue; }}
        if (!lockedPicks[gid] && !simPicks[gid]) {{
          const sA = getStats(wA.team, wA.seed), sB = getStats(wB.team, wB.seed);
          const pred = predictGame(sA, sB);
          simPicks[gid] = shouldPickUpset(pred.probA, wA.seed, wB.seed, agg) ? wA.team : wB.team;
        }}
        const pick = lockedPicks[gid] || simPicks[gid];
        if (pick !== wA.team && pick !== wB.team) {{
          delete simPicks[gid];
          const sA = getStats(wA.team, wA.seed), sB = getStats(wB.team, wB.seed);
          const pred = predictGame(sA, sB);
          simPicks[gid] = shouldPickUpset(pred.probA, wA.seed, wB.seed, agg) ? wA.team : wB.team;
        }}
        const finalPick = lockedPicks[gid] || simPicks[gid];
        next.push(finalPick === wA.team ? wA : wB);
      }}
      prev = next;
    }});
  }});
  recomputeBracket();
  const rw2 = {{}};
  REGIONS.forEach(r => {{
    const st = bracketState[`${{r}}-8-0`];
    if (st && st.pick) rw2[r] = getStats(st.pick, st.pick===st.teamA ? st.seedA : st.seedB);
  }});
  const ffPairs = [['FF-4-0', QUADRANT[0], QUADRANT[3]], ['FF-4-1', QUADRANT[1], QUADRANT[2]]];
  ffPairs.forEach(([gid, rA, rB]) => {{
    const tA = rw2[rA], tB = rw2[rB];
    if (!tA||!tB) return;
    if (!lockedPicks[gid] && !simPicks[gid]) {{
      const pred = predictGame(tA, tB);
      simPicks[gid] = shouldPickUpset(pred.probA, tA.seed, tB.seed, upsetAggression) ? tA.team : tB.team;
    }}
  }});
  recomputeBracket();
  const s0 = bracketState['FF-4-0'], s1 = bracketState['FF-4-1'];
  if (s0?.pick && s1?.pick) {{
    if (!lockedPicks['FF-2-0'] && !simPicks['FF-2-0']) {{
      const cA = getStats(s0.pick, s0.pick===s0.teamA ? s0.seedA : s0.seedB);
      const cB = getStats(s1.pick, s1.pick===s1.teamA ? s1.seedA : s1.seedB);
      const pred = predictGame(cA, cB);
      simPicks['FF-2-0'] = shouldPickUpset(pred.probA, cA.seed, cB.seed, upsetAggression) ? cA.team : cB.team;
    }}
  }}
  recomputeBracket();
}}

function renderBracket() {{
  document.querySelectorAll('.game').forEach(el => {{
    const gid = el.dataset.gameId;
    const st = bracketState[gid];
    const slots = el.querySelectorAll('.team-slot');
    const infoBtn = el.querySelector('.info-btn');
    if (st && (st.teamA || st.teamB)) {{
      const teams = [{{ name:st.teamA, seed:st.seedA }}, {{ name:st.teamB, seed:st.seedB }}];
      const alert = gameAlerts[gid];
      el.classList.toggle('has-alert', !!alert);
      const seedDiff = Math.abs((st.seedA||0) - (st.seedB||0));
      const hasUpsetPick = st.pick && (
        (st.pick === st.teamA && (st.seedA||0) > (st.seedB||0)) ||
        (st.pick === st.teamB && (st.seedB||0) > (st.seedA||0))
      );
      const underdogIdx = st.probA !== null && st.probA >= .5 ? 1 : 0;
      const favIdx = 1 - underdogIdx;
      slots.forEach((slot, idx) => {{
        const badgeEl = slot.querySelector('.upset-badge');
        if (badgeEl) {{
          let show = false, level = 'strong', label = 'Upset';
          if (alert) {{
            const onUnderdog = alert.badgeOnUnderdog !== false;
            show = (onUnderdog && idx === underdogIdx) || (!onUnderdog && idx === favIdx);
            level = alert.level || 'strong';
            label = level === 'strong' ? '!' : level === 'mild' ? '~' : level === 'blowout' ? 'Blowout' : 'Upset';
          }} else if (hasUpsetPick && teams[idx].name === st.pick) {{
            show = true; level = 'strong'; label = 'Upset';
          }}
          badgeEl.style.display = show ? 'inline' : 'none';
          badgeEl.textContent = label;
          badgeEl.className = 'upset-badge ' + (level || 'strong');
        }}
      }});
      if (infoBtn) {{
        infoBtn.style.display = (st.teamA && st.teamB) ? 'flex' : 'none';
        infoBtn.title = alert ? alert.reason : 'Matchup details';
      }}
      slots.forEach((slot, idx) => {{
        const t = teams[idx];
        if (t.name) {{
          slot.dataset.team = t.name; slot.dataset.seed = t.seed || '';
          slot.querySelector('.sd').textContent = t.seed || '';
          const displayName = DISPLAY_NAMES[t.name] || t.name;
          slot.querySelector('.tm').textContent = displayName;
          slot.querySelector('.tm').title = displayName !== t.name ? t.name : '';
          slot.classList.remove('empty');
          slot.classList.toggle('picked', st.pick === t.name);
          slot.classList.toggle('locked', st.isLocked && st.pick === t.name);
          slot.onclick = null;
          const isUpset = st.pick === t.name && hasUpsetPick;
          slot.classList.toggle('upset-pick', isUpset);
        }} else {{
          slot.dataset.team = ''; slot.dataset.seed = '';
          slot.querySelector('.sd').textContent = '';
          slot.querySelector('.tm').textContent = '\\u2014';
          slot.classList.add('empty');
          slot.classList.remove('picked','locked','upset-pick');
          slot.onclick = null;
        }}
      }});
    }} else {{
      el.classList.remove('has-alert');
      slots.forEach(slot => {{
        slot.dataset.team = ''; slot.dataset.seed = '';
        slot.querySelector('.sd').textContent = '';
        slot.querySelector('.tm').textContent = '\\u2014';
        slot.classList.add('empty');
        slot.classList.remove('picked','locked','upset-pick');
        slot.onclick = null;
        const b = slot.querySelector('.upset-badge');
        if (b) b.style.display = 'none';
      }});
      if (infoBtn) infoBtn.style.display = 'none';
    }}
  }});
  const cs = bracketState['FF-2-0'];
  const cn = cs?.pick || '\\u2014';
  document.getElementById('champ-name').textContent = cn;
  const b = document.getElementById('champ-banner-name');
  if (b) b.textContent = cn;
}}

function handleGameClick(e, gameId) {{
  const slot = e.target.closest('.team-slot');
  if (slot && !slot.classList.contains('empty') && slot.dataset.team) {{
    e.stopPropagation();
    togglePick(gameId, slot.dataset.team);
  }} else {{
    e.stopPropagation();
    showAnalysis(gameId);
  }}
}}

function togglePick(gameId, teamName) {{
  if (!teamName) return;
  const currentPick = lockedPicks[gameId] || simPicks[gameId] || null;
  if (currentPick === teamName) {{
    delete lockedPicks[gameId]; delete simPicks[gameId];
    clearDownstream(gameId, teamName);
  }} else if (currentPick && currentPick !== teamName) {{
    const oldWinner = currentPick;
    delete simPicks[gameId]; lockedPicks[gameId] = teamName;
    clearDownstream(gameId, oldWinner);
  }} else {{
    lockedPicks[gameId] = teamName;
  }}
  recomputeBracket();
}}

function clearDownstream(gameId, oldWinner) {{
  if (!oldWinner) return;
  const nextId = getNextGameId(gameId);
  if (!nextId) return;
  const st = bracketState[nextId];
  if (!st) return;
  if (st.teamA === oldWinner || st.teamB === oldWinner) {{
    const nextPick = lockedPicks[nextId] || simPicks[nextId] || null;
    delete lockedPicks[nextId]; delete simPicks[nextId];
    if (nextPick) clearDownstream(nextId, nextPick);
  }}
}}

function resetPicks() {{
  lockedPicks = {{}}; simPicks = {{}};
  recomputeBracket();
}}

function switchTab(tabId) {{
  document.querySelectorAll('.tab').forEach(t => {{ t.classList.toggle('active', t.dataset.tab === tabId); }});
  document.querySelectorAll('.tab-pane').forEach(p => {{ p.classList.toggle('active', p.id === 'tab-' + tabId); }});
  document.getElementById('controls').style.display = tabId === 'bracket' ? 'flex' : 'none';
}}

function onYearChange(y) {{
  window.location.href = 'bracket_' + y + '.html';
}}

function onUpsetChange(val) {{
  upsetAggression = val / 100;
  document.getElementById('upset-val').textContent = val + '%';
  const labels = ['All Chalk','Slight Chaos','Moderate Chaos','Heavy Chaos','Maximum Chaos'];
  document.getElementById('upset-label').textContent = labels[Math.min(4, Math.floor(val / 25))];
  if (_simTimer) clearTimeout(_simTimer);
  _simTimer = setTimeout(() => {{
    simPicks = {{}};
    simulateAll();
  }}, 400);
}}

function buildBracket() {{
  const wrap = document.getElementById('bracket-wrap');
  let html = '<div class="bracket-top">';
  html += buildRegionHTML(QUADRANT[0], false);
  html += buildFinalFourHTML();
  html += buildRegionHTML(QUADRANT[1], true);
  html += '</div><div class="bracket-bottom">';
  html += buildRegionHTML(QUADRANT[3], false);
  html += '<div class="ff-center"></div>';
  html += buildRegionHTML(QUADRANT[2], true);
  html += '</div>';
  wrap.innerHTML = html;
}}

function buildRegionHTML(region, flipped) {{
  const teams = BRACKET[region];
  if (!teams) return '<div class="region-bracket"></div>';
  const rounds = ['R64','R32','S16','E8'], gamesPerRound = [8,4,2,1];
  let html = `<div class="region-bracket ${{flipped?'flipped':''}}">`;
  html += `<div class="region-label">${{region}}</div><div class="rounds">`;
  rounds.forEach((rnd, ri) => {{
    const roundOf = [64,32,16,8][ri];
    const isLast = ri === rounds.length - 1;
    html += `<div class="round ${{isLast?'round-last':''}}">`;
    html += `<div class="round-header">${{rnd}}</div>`;
    const numGames = gamesPerRound[ri];
    for (let gi = 0; gi < numGames; gi++) {{
      if (gi % 2 === 0) html += '<div class="game-pair">';
      const gid = `${{region}}-${{roundOf}}-${{gi}}`;
      html += `<div class="game" data-game-id="${{gid}}" onclick="handleGameClick(event,'${{gid}}')">`;
      if (ri === 0) {{
        const tA = teams[gi*2], tB = teams[gi*2+1];
        html += `<div class="team-slot" data-team="${{tA?.team||''}}" data-seed="${{tA?.seed||''}}"><span class="sd">${{tA?.seed||''}}</span><span class="tm">${{tA?.team||'\\u2014'}}</span><span class="upset-badge" style="display:none"></span></div>`;
        html += `<div class="team-slot" data-team="${{tB?.team||''}}" data-seed="${{tB?.seed||''}}"><span class="sd">${{tB?.seed||''}}</span><span class="tm">${{tB?.team||'\\u2014'}}</span><span class="upset-badge" style="display:none"></span></div>`;
      }} else {{
        html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="upset-badge" style="display:none"></span></div>';
        html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="upset-badge" style="display:none"></span></div>';
      }}
      html += `<div class="info-btn" style="display:${{ri===0?'flex':'none'}}" onclick="event.stopPropagation();showAnalysis('${{gid}}')">i</div>`;
      html += '</div>';
      if (gi % 2 === 1 || gi === numGames - 1) html += '</div>';
    }}
    html += '</div>';
  }});
  html += '</div></div>';
  return html;
}}

function buildFinalFourHTML() {{
  let html = '<div class="ff-center"><div class="ff-label">Final Four</div>';
  ['FF-4-0','FF-4-1'].forEach(gid => {{
    html += `<div class="game" data-game-id="${{gid}}" onclick="handleGameClick(event,'${{gid}}')">`;
    html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="upset-badge" style="display:none"></span></div>';
    html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="upset-badge" style="display:none"></span></div>';
    html += `<div class="info-btn" style="display:none" title="Matchup details">i</div>`;
    html += '</div>';
  }});
  html += '<div class="ff-label">Championship</div>';
  html += '<div class="game" data-game-id="FF-2-0" onclick="handleGameClick(event,\'FF-2-0\')">';
  html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="upset-badge" style="display:none"></span></div>';
  html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="upset-badge" style="display:none"></span></div>';
  html += `<div class="info-btn" style="display:none" title="Matchup details">i</div>`;
  html += '</div>';
  html += '<div class="champ-banner"><div class="trophy">&#127942;</div><div class="champ-team" id="champ-banner-name">\\u2014</div></div>';
  html += '</div>';
  return html;
}}

function showAnalysis(gameId) {{
  const st = bracketState[gameId];
  if (!st || !st.teamA || !st.teamB) return;
  const panel = document.getElementById('analysis-panel');
  const overlay = document.getElementById('analysis-overlay');
  const content = document.getElementById('analysis-content');
  const pA = st.probA !== null ? st.probA : .5;
  const probA = (pA*100).toFixed(0), probB = ((1-pA)*100).toFixed(0);
  const spreadAmt = Math.abs(st.margin).toFixed(1);
  const favTeam = pA >= 0.5 ? st.teamA : st.teamB;
  const dogTeam = pA >= 0.5 ? st.teamB : st.teamA;
  const spreadFav = '-' + spreadAmt;
  const spreadDog = '+' + spreadAmt;
  const p = Math.max(pA, 1-pA);
  const conf = p>=.9?'lock':p>=.75?'strong':p>=.6?'lean':'tossup';
  const sA = TEAM_STATS[st.teamA], sB = TEAM_STATS[st.teamB];
  let orig = null;
  for (const pp of PICKS) {{
    if ((pp.team_a===st.teamA&&pp.team_b===st.teamB)||(pp.team_b===st.teamA&&pp.team_a===st.teamB)) {{ orig=pp; break; }}
  }}
  const margin = Math.abs(st.margin || 0);
  let insight = orig?.insight;
  let factors = orig?.key_factors || [];
  if (!orig) {{
    insight = `Favorite ${{spreadFav}}, Underdog ${{spreadDog}}.`;
    if (margin >= 12) insight += ' ' + (pA >= .5 ? st.teamA : st.teamB) + ' dominates on efficiency.';
    else if (margin >= 5) insight += ' Solid efficiency edge to the favorite.';
    else if (margin >= 2) insight += ' Slight edge to the favorite.';
    else insight += ' Razor-thin margin — essentially a coin flip.';
    if (margin >= 5) factors = [(pA >= .5 ? st.teamA : st.teamB) + ': ' + margin.toFixed(1) + '-pt efficiency edge'];
    else if (st.seedA && st.seedB && Math.abs(st.seedA - st.seedB) >= 3) factors = ['Seed differential suggests upset potential'];
  }}
  const hist = orig?.historical || null;
  let statsHtml = '';
  if (sA) {{
    statsHtml += `<div class="stat-row"><span>${{st.teamA}} Adj O/D</span><span class="val">${{sA.adj_o.toFixed(1)}} / ${{sA.adj_d.toFixed(1)}}</span></div>`;
    if (sA.wab!=null) statsHtml += `<div class="stat-row"><span>${{st.teamA}} WAB</span><span class="val">${{sA.wab.toFixed(1)}}</span></div>`;
  }}
  if (sB) {{
    statsHtml += `<div class="stat-row"><span>${{st.teamB}} Adj O/D</span><span class="val">${{sB.adj_o.toFixed(1)}} / ${{sB.adj_d.toFixed(1)}}</span></div>`;
    if (sB.wab!=null) statsHtml += `<div class="stat-row"><span>${{st.teamB}} WAB</span><span class="val">${{sB.wab.toFixed(1)}}</span></div>`;
  }}
  if (sA?.sos!=null && sB?.sos!=null) {{
    statsHtml += `<div class="stat-row"><span>SOS</span><span class="val">${{sA.sos.toFixed(3)}} vs ${{sB.sos.toFixed(3)}}</span></div>`;
  }}
  content.innerHTML = `
    <h3>(${{st.seedA}}) ${{st.teamA}} vs (${{st.seedB}}) ${{st.teamB}} <span class="conf-badge conf-${{conf}}">${{conf.toUpperCase()}}</span></h3>
    <div class="matchup-line"><span class="team-side ${{pA>=.5?'fav':''}}">(${{st.seedA}}) ${{st.teamA}}</span><span style="font-weight:700">${{probA}}%</span></div>
    <div class="prob-bar-wrap"><div class="prob-bar-a" style="width:${{probA}}%"></div><div class="prob-bar-b" style="width:${{probB}}%"></div></div>
    <div class="matchup-line"><span class="team-side ${{pA<.5?'fav':''}}">(${{st.seedB}}) ${{st.teamB}}</span><span style="font-weight:700">${{probB}}%</span></div>
    <div class="stat-row"><span>Spread</span><span class="val">${{favTeam}} ${{spreadFav}}, ${{dogTeam}} ${{spreadDog}}</span></div>
    ${{st.pick?`<div class="stat-row"><span>Pick</span><span class="val" style="color:var(--green)">${{st.pick}}</span></div>`:''}}
    ${{(()=>{{
      const alert = gameAlerts[gameId];
      if (!alert) return '';
      const labels = {{strong:'Upset danger',mild:'Close margin',blowout:'Blowout'}};
      const colors = {{strong:'var(--red)',mild:'var(--gold)',blowout:'var(--green)'}};
      const c = colors[alert.level]||'#888';
      return `<div class="matchup-alert" style="background:${{c}}12;border-left:3px solid ${{c}};padding:8px 10px;margin:8px 0;font-size:.8rem;">
        <div style="font-weight:700;color:${{c}};margin-bottom:2px;">${{labels[alert.level]||'Note'}}</div>
        <div style="color:var(--text);">${{alert.reason}}</div></div>`;
    }})()}}
    ${{(()=>{{
      const h2h = orig?.head_to_head;
      if (!h2h) return '';
      const hasSeason = h2h.this_season?.length > 0;
      const hasPast = h2h.past_tournament?.length > 0;
      if (!hasSeason && !hasPast) return '<div class="h2h-block" style="margin:8px 0;font-size:.8rem;color:var(--muted);">No head-to-head matchups between these teams.</div>';
      let html = '<div class="h2h-block" style="margin:8px 0;font-size:.8rem;">';
      if (hasSeason) {{
        html += '<div style="font-weight:700;margin-bottom:4px;">This season</div>';
        h2h.this_season.forEach(g => {{
          const loc = g.location ? ' @ ' + g.location : (g.date ? ' ' + g.date : '');
          const loser = g.team_a === g.winner ? g.team_b : g.team_a;
          html += '<div style="margin-bottom:2px;">' + (g.date||'') + loc + ': ' + g.winner + ' ' + (g.score_a||0) + '–' + (g.score_b||0) + ' ' + loser + '</div>';
        }});
      }}
      if (hasPast) {{
        html += '<div style="font-weight:700;margin:8px 0 4px 0;">Past tournaments</div>';
        h2h.past_tournament.forEach(g => {{
          const loc = g.region ? ' (' + g.region + ')' : '';
          const loser = g.team_a === g.winner ? g.team_b : g.team_a;
          html += '<div style="margin-bottom:2px;">' + g.year + ' ' + (g.round_name||'') + loc + ': ' + g.winner + ' ' + (g.score_a||0) + '–' + (g.score_b||0) + ' ' + loser + '</div>';
        }});
      }}
      html += '</div>';
      return html;
    }})()}}
    ${{statsHtml}}
    ${{hist?`<div class="stat-row"><span>Historical</span><span class="val">${{hist}}</span></div>`:''}}
    <div class="insight">${{insight}}</div>
    ${{factors.length?`<ul class="factors">${{factors.map(f=>`<li>${{f}}</li>`).join('')}}</ul>`:''}}
  `;
  panel.classList.add('show');
  overlay.classList.add('show');
}}
function closeAnalysis() {{
  document.getElementById('analysis-panel').classList.remove('show');
  document.getElementById('analysis-overlay').classList.remove('show');
}}

PICKS.forEach(p => {{
  const region = p.region, roundOf = p.round;
  let gid = null;
  if (region && roundOf >= 8) {{
    const same = PICKS.filter(pp => pp.region===region && pp.round===roundOf);
    const gi = same.indexOf(p);
    if (gi >= 0) gid = `${{region}}-${{roundOf}}-${{gi}}`;
  }} else if (roundOf === 4) {{
    const ff = PICKS.filter(pp => pp.round===4);
    const gi = ff.indexOf(p);
    if (gi >= 0) gid = `FF-4-${{gi}}`;
  }} else if (roundOf === 2) {{
    gid = 'FF-2-0';
  }}
  if (gid) simPicks[gid] = p.pick;
}});

buildBracket();
recomputeBracket();
document.addEventListener('keydown', e => {{ if (e.key==='Escape') closeAnalysis(); }});
</script>
</body>
</html>"""
    return html


def _generate_all_years(args):
    """Generate HTML for every bracket_YYYY.json found in data/."""
    import subprocess
    years = _available_bracket_years()
    print(f"Generating brackets for {len(years)} years: {years}")
    for y in years:
        bracket_path = os.path.join(DATA_DIR, f"bracket_{y}.json")
        cmd = [sys.executable, __file__,
               "--bracket", bracket_path,
               "--sims", str(args.sims),
               "--upset", str(args.upset)]
        print(f"\n{'='*60}\n  Year {y}\n{'='*60}")
        subprocess.run(cmd, cwd=ROOT)
    print(f"\nAll {len(years)} years generated in output/")


def main():
    parser = argparse.ArgumentParser(description="Bracket Brain — March Madness Bracket Picks")
    parser.add_argument("--sims", type=int, default=DEFAULT_NUM_SIMS,
                        help=f"Monte Carlo simulations (default: {DEFAULT_NUM_SIMS})")
    parser.add_argument("--bracket", type=str, default="data/bracket_2026.json",
                        help="Path to bracket JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML file path (default: output/bracket_YYYY.html)")
    parser.add_argument("--upset", type=float, default=0.0,
                        help="Upset aggressiveness 0.0-1.0 (default: 0.0)")
    parser.add_argument("--best-bets", action="store_true",
                        help="Fetch today's odds and include Top Bets tab (requires ODDS_API_KEY)")
    parser.add_argument("--all", action="store_true",
                        help="Generate HTML for all available bracket years")
    args = parser.parse_args()

    if args.all:
        _generate_all_years(args)
        return

    print("=" * 60)
    print("BRACKET BRAIN — Interactive Bracket Picker")
    print("=" * 60)
    print(f"\nBracket: {args.bracket}")
    print(f"Monte Carlo sims: {args.sims:,}")
    print(f"Upset aggressiveness: {args.upset:.0%}")

    if not os.path.exists(args.bracket):
        print(f"\nERROR: Bracket file not found at {args.bracket}")
        sys.exit(1)

    year = _year_from_bracket_path(args.bracket)
    if args.output is None:
        args.output = f"output/bracket_{year}.html"
    bracket, ff_matchups, quadrant_order = load_bracket(args.bracket, data_dir=DATA_DIR, year=year)

    total_teams = sum(len(region) for region in bracket.values())
    print(f"Loaded bracket: {len(bracket)} regions, {total_teams} teams\n")

    placeholders = sum(1 for r in bracket.values() for t in r.values() if t["team"].startswith("TEAM_"))
    if placeholders > 0:
        print(f"WARNING: {placeholders} placeholder teams found.\n")

    config = ModelConfig(num_sims=args.sims)
    cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
    if os.path.isfile(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        for k, v in cal.items():
            if hasattr(config, k):
                setattr(config, k, v)
        print("Loaded calibrated model parameters")

    print("Generating bracket picks (63 games)...")
    bracket_result = generate_bracket_picks(bracket, config, upset_aggression=args.upset, quadrant_order=quadrant_order,
                                           data_dir=DATA_DIR, year=year)
    picks = bracket_result["picks"]

    print(f"\n  Champion: {bracket_result['champion']}")
    print(f"  Final Four: {', '.join(bracket_result['final_four'])}")

    # LLM-powered analysis (when ANTHROPIC_API_KEY is set)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            sys.path.insert(0, os.path.join(ROOT, "scripts"))
            from generate_analysis import generate_analysis_for_picks
            cache_path = os.path.join(DATA_DIR, f"analysis_cache_{year}.json")
            team_stats = {}
            for fname in [f"teams_merged_{year}.json", f"torvik_{year}.json"]:
                tpath = os.path.join(DATA_DIR, fname)
                if os.path.isfile(tpath):
                    with open(tpath) as f:
                        tdata = json.load(f)
                    if isinstance(tdata, list):
                        for t in tdata:
                            if t.get("team"):
                                team_stats[t["team"]] = t
                    break
            print("\nGenerating Claude-powered analysis...")
            picks = generate_analysis_for_picks(picks, team_stats, cache_path)
            bracket_result["picks"] = picks
        except Exception as e:
            print(f"  Claude analysis failed: {e} — using template analysis")
    else:
        print("\n  (Set ANTHROPIC_API_KEY for Claude-powered analysis)")

    conf_counts = {}
    for p in picks:
        conf_counts[p["confidence"]] = conf_counts.get(p["confidence"], 0) + 1
    print(f"  Confidence: {conf_counts.get('lock',0)} locks, {conf_counts.get('strong',0)} strong, "
          f"{conf_counts.get('lean',0)} leans, {conf_counts.get('tossup',0)} tossups")

    print(f"\nRunning {args.sims:,} Monte Carlo simulations...")
    mc_results = run_monte_carlo(bracket, config=config)

    print("\n  Championship odds (top 8):")
    for i, (team, prob) in enumerate(list(mc_results["champion_probs"].items())[:8]):
        seed = _find_team_seed(team, bracket)
        print(f"    {i+1:2}. ({seed}) {team:20s} {prob*100:5.1f}%")

    best_bets = []
    if args.best_bets:
        api_key = os.environ.get("ODDS_API_KEY", "")
        if api_key:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "best_bets", os.path.join(ROOT, "scripts", "best_bets.py"))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                print("Fetching today's odds for Top Bets tab...")
                best_bets = mod.get_best_bets_json(api_key, year=year)
                print(f"  Found {len(best_bets)} qualifying bet(s)")
            except Exception as e:
                print(f"  Warning: Could not fetch best bets: {e}")
        else:
            print("  ODDS_API_KEY not set — skipping Top Bets tab")

    print("\nGenerating HTML...")
    html = generate_html(bracket_result, mc_results, bracket, config, args.sims, args.upset, quadrant_order, year=year, best_bets=best_bets)

    os.makedirs(os.path.dirname(args.output) or "output", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"  Saved to {args.output}")

    json_path = args.output.replace(".html", "_data.json")
    output_data = {
        "bracket_picks": bracket_result,
        "monte_carlo": mc_results,
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Raw data saved to {json_path}")

    print(f"\nDone! Open {args.output} in your browser.")


if __name__ == "__main__":
    main()
