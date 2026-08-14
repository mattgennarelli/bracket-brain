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
import hashlib
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import re

from engine import (
    generate_bracket_picks, run_monte_carlo, load_bracket, analyze_matchup,
    REGIONS, FIRST_ROUND_MATCHUPS, DEFAULT_NUM_SIMS, SEED_WEIGHT,
    ModelConfig, DEFAULT_CONFIG, build_locked_picks_from_results, resolve_ff_pairs,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def _prediction_inputs_hash(year: int) -> str:
    """Stable hash of files that affect bracket and Monte Carlo predictions."""
    paths = [
        os.path.join(DATA_DIR, f"bracket_{year}.json"),
        os.path.join(DATA_DIR, f"teams_merged_{year}.json"),
        os.path.join(DATA_DIR, f"injuries_{year}.json"),
        os.path.join(DATA_DIR, f"results_{year}.json"),
        os.path.join(DATA_DIR, "calibrated_config.json"),
    ]
    h = hashlib.sha256()
    for path in paths:
        h.update(path.encode("utf-8"))
        if not os.path.isfile(path):
            h.update(b"<missing>")
            continue
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _year_from_bracket_path(path):
    m = re.search(r"bracket_(\d{4})", path)
    return int(m.group(1)) if m else 2026


def _load_results_games(year: int):
    path = os.path.join(DATA_DIR, f"results_{year}.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, dict):
        games = data.get("games", [])
    elif isinstance(data, list):
        games = data
    else:
        games = []
    return [game for game in games if isinstance(game, dict)]


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



def generate_html(bracket_result, mc_results, bracket, config, num_sims,
                  upset_aggression=0.0, quadrant_order=None, year=2026):
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
            if team.get("injuries"):
                ts["injuries"] = team["injuries"]
                if team.get("injury_impact") is not None:
                    ts["injury_impact"] = team["injury_impact"]
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

    def _injury_badge(team_name):
        imp = team_stats.get(team_name, {}).get("injury_impact", 0) or 0
        if imp > 1:
            return f' <span class="injury-badge" title="Key injuries ({imp:.1f} pts impact)">!</span>'
        return ""

    max_champ = max(champ_probs.values()) if champ_probs else 1
    champ_rows = ""
    for i, (team, prob) in enumerate(list(champ_probs.items())[:16]):
        seed = _find_team_seed(team, bracket)
        bar_w = prob / max_champ * 100
        champ_rows += (f'<tr><td class="rank">{i+1}</td>'
                       f'<td><span class="seed-badge">{seed}</span>{team}{_injury_badge(team)}</td>'
                       f'<td class="bar-cell"><div class="bar" style="width:{bar_w:.0f}%"></div></td>'
                       f'<td class="pct">{prob*100:.1f}%</td></tr>\n')

    max_ff = max(ff_probs.values()) if ff_probs else 1
    ff_rows = ""
    for i, (team, prob) in enumerate(list(ff_probs.items())[:16]):
        seed = _find_team_seed(team, bracket)
        bar_w = prob / max_ff * 100
        ff_rows += (f'<tr><td class="rank">{i+1}</td>'
                    f'<td><span class="seed-badge">{seed}</span>{team}{_injury_badge(team)}</td>'
                    f'<td class="bar-cell"><div class="bar" style="width:{bar_w:.0f}%"></div></td>'
                    f'<td class="pct">{prob*100:.1f}%</td></tr>\n')

    conf_counts = {"lock": 0, "strong": 0, "lean": 0, "tossup": 0}
    for p in picks:
        conf_counts[p["confidence"]] = conf_counts.get(p["confidence"], 0) + 1

    quadrant_order_json = json.dumps(quadrant_order)

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
.injury-badge {{ color:#dc2626; font-size:.75rem; font-weight:700; margin-left:2px; cursor:help; }}

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
    <span>Calibrated on 1,071 games</span>
    <span>{timestamp}</span>
  </div>
</div>

<div class="tabs">
  <div class="tab active" data-tab="bracket" onclick="switchTab('bracket')">Bracket</div>
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

<div class="footer">
  <p>Built by Matt Gennarelli &middot; Calibrated on 1,071 tournament games (2008-2025) &middot; Data: Bart Torvik T-Rank</p>
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
          const injSlot = slot.querySelector('.injury-badge-slot');
          if (injSlot) {{
            const imp = TEAM_STATS[t.name]?.injury_impact;
            if (imp != null && imp > 1) {{
              injSlot.innerHTML = '<span class="injury-badge" title="Key injuries (' + imp.toFixed(1) + ' pts impact)">!</span>';
              injSlot.style.display = 'inline';
            }} else {{
              injSlot.innerHTML = '';
              injSlot.style.display = 'none';
            }}
          }}
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
          const injSlot = slot.querySelector('.injury-badge-slot');
          if (injSlot) {{ injSlot.innerHTML = ''; injSlot.style.display = 'none'; }}
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
        const injSlot = slot.querySelector('.injury-badge-slot');
        if (injSlot) {{ injSlot.innerHTML = ''; injSlot.style.display = 'none'; }}
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
  if (!wrap) return;
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
  let teams = BRACKET[region];
  if (!teams && region) {{
    const key = Object.keys(BRACKET).find(k => k.toLowerCase() === region.toLowerCase());
    if (key) teams = BRACKET[key];
  }}
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
        html += `<div class="team-slot" data-team="${{tA?.team||''}}" data-seed="${{tA?.seed||''}}"><span class="sd">${{tA?.seed||''}}</span><span class="tm">${{tA?.team||'\\u2014'}}</span><span class="injury-badge-slot"></span><span class="upset-badge" style="display:none"></span></div>`;
        html += `<div class="team-slot" data-team="${{tB?.team||''}}" data-seed="${{tB?.seed||''}}"><span class="sd">${{tB?.seed||''}}</span><span class="tm">${{tB?.team||'\\u2014'}}</span><span class="injury-badge-slot"></span><span class="upset-badge" style="display:none"></span></div>`;
      }} else {{
        html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="injury-badge-slot"></span><span class="upset-badge" style="display:none"></span></div>';
        html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="injury-badge-slot"></span><span class="upset-badge" style="display:none"></span></div>';
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
    html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="injury-badge-slot"></span><span class="upset-badge" style="display:none"></span></div>';
    html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="injury-badge-slot"></span><span class="upset-badge" style="display:none"></span></div>';
    html += `<div class="info-btn" style="display:none" title="Matchup details">i</div>`;
    html += '</div>';
  }});
  html += '<div class="ff-label">Championship</div>';
  html += '<div class="game" data-game-id="FF-2-0" onclick="handleGameClick(event,\\'FF-2-0\\')">';
  html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="injury-badge-slot"></span><span class="upset-badge" style="display:none"></span></div>';
  html += '<div class="team-slot empty" data-team="" data-seed=""><span class="sd"></span><span class="tm">\\u2014</span><span class="injury-badge-slot"></span><span class="upset-badge" style="display:none"></span></div>';
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
      const injA = sA?.injuries, injB = sB?.injuries;
      const impA = sA?.injury_impact, impB = sB?.injury_impact;
      if ((!injA || injA.length===0) && (!injB || injB.length===0)) return '';
      const fmt = (team, injs, imp) => {{
        if (!injs || injs.length===0) return '';
        const list = injs.map(i => i.player + ' (' + (i.status||'out') + ')').join(', ');
        const pts = (imp!=null && imp>0) ? ' — ' + imp.toFixed(1) + ' pts impact' : '';
        return `<div style="margin-bottom:6px;"><strong>${{team}}</strong>: ${{list}}${{pts}}</div>`;
      }};
      return `<div class="injury-block" style="background:rgba(220,38,38,.08);border-left:3px solid #dc2626;padding:8px 10px;margin:8px 0;font-size:.8rem;">
        <div style="font-weight:700;color:#dc2626;margin-bottom:4px;">Injuries</div>
        ${{fmt(st.teamA, injA, impA)}}
        ${{fmt(st.teamB, injB, impB)}}
      </div>`;
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

document.addEventListener('DOMContentLoaded', function() {{
  try {{
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
  }} catch (e) {{ console.warn('Picks init:', e); }}
  buildBracket();
  try {{
    recomputeBracket();
  }} catch (e) {{ console.warn('Recompute:', e); }}
  document.addEventListener('keydown', e => {{ if (e.key==='Escape') closeAnalysis(); }});
}});
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
    parser.add_argument("--all", action="store_true",
                        help="Generate HTML for all available bracket years")
    parser.add_argument("--write-monte-carlo", action="store_true", default=True,
                        help="Write data/monte_carlo_YYYY.json for API (default: True)")
    parser.add_argument("--no-write-monte-carlo", action="store_false", dest="write_monte_carlo",
                        help="Skip writing monte_carlo file")
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

    # grading_locks: real results, used ONLY to grade the model's own picks
    # (actual_winner/is_correct per game) -- never drives which team advances.
    # The bracket is a simulator: it always shows the model's own picks. See
    # generate_bracket_picks(locked_picks=None, ...) below.
    grading_locks = build_locked_picks_from_results(
        bracket,
        _load_results_games(year),
        quadrant_order=quadrant_order,
        ff_matchups=ff_matchups,
    )
    if grading_locks:
        print(f"Grading model picks against {len(grading_locks)} completed tournament games")

    print("Generating bracket picks (63 games)...")
    bracket_result = generate_bracket_picks(bracket, config, upset_aggression=args.upset,
                                           quadrant_order=quadrant_order, ff_matchups=ff_matchups,
                                           data_dir=DATA_DIR, year=year, locked_picks=None,
                                           grading_locks=grading_locks)
    picks = bracket_result["picks"]

    print(f"\n  Champion: {bracket_result['champion']}")
    print(f"  Final Four: {', '.join(bracket_result['final_four'])}")

    # Precomputed artifact for the API's default page-load path (upset_aggression=0.0
    # only -- the chaos slider is a deliberate live action and stays live-computed).
    # Served as-is by GET /bracket/{year}, no per-request computation.
    if args.upset == 0.0:
        picks_path = os.path.join(DATA_DIR, f"bracket_picks_{year}.json")
        ff_pairs = [list(pair) for pair in resolve_ff_pairs(quadrant_order, ff_matchups)]
        picks_export = {
            "year": year,
            "upset_aggression": 0.0,
            "prediction_inputs_hash": _prediction_inputs_hash(year),
            "picks": bracket_result,
            "quadrant_order": quadrant_order,
            "ff_pairs": ff_pairs,
        }
        with open(picks_path, "w") as f:
            json.dump(picks_export, f)
        print(f"  Wrote {picks_path} for API")

        # Historical "what actually happened" view -- real results drive
        # advancement here (this is the one place they're allowed to), always
        # at upset_aggression=0.0 since real results don't depend on chaos.
        # Self-graded: is_correct compares the model's own independent pick
        # against the very same real outcome.
        actual_result = generate_bracket_picks(bracket, config, upset_aggression=0.0,
                                               quadrant_order=quadrant_order, ff_matchups=ff_matchups,
                                               data_dir=DATA_DIR, year=year, locked_picks=grading_locks,
                                               grading_locks=grading_locks)
        actual_path = os.path.join(DATA_DIR, f"bracket_actual_{year}.json")
        actual_export = {
            "year": year,
            "upset_aggression": 0.0,
            "prediction_inputs_hash": _prediction_inputs_hash(year),
            "picks": actual_result,
            "quadrant_order": quadrant_order,
            "ff_pairs": ff_pairs,
        }
        with open(actual_path, "w") as f:
            json.dump(actual_export, f)
        print(f"  Wrote {actual_path} for API")

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
    mc_results = run_monte_carlo(
        bracket,
        config=config,
        year=year,
        quadrant_order=quadrant_order,
        ff_matchups=ff_matchups,
    )

    print("\n  Championship odds (top 8):")
    for i, (team, prob) in enumerate(list(mc_results["champion_probs"].items())[:8]):
        seed = _find_team_seed(team, bracket)
        print(f"    {i+1:2}. ({seed}) {team:20s} {prob*100:5.1f}%")

    if args.write_monte_carlo and args.sims == 10000:
        mc_path = os.path.join(DATA_DIR, f"monte_carlo_{year}.json")
        # final_four_by_region: top 8 per-region teams by F4 probability. Computed
        # here (once, at generation time) and embedded in the file so the API can
        # serve it as a pure static read -- see api.py's now-removed
        # _add_final_four_by_region, which used to redo this work on every request.
        ff_probs = mc_results["final_four_probs"]
        final_four_by_region = {}
        for region, teams in bracket.items():
            team_probs = []
            for seed, team in teams.items():
                t = team.get("team") if isinstance(team, dict) else None
                if t and t in ff_probs:
                    team_probs.append((t, ff_probs[t]))
            team_probs.sort(key=lambda x: -x[1])
            final_four_by_region[region] = [[t, round(p, 4)] for t, p in team_probs[:8]]
        mc_export = {
            "year": year,
            "num_simulations": args.sims,
            "prediction_inputs_hash": _prediction_inputs_hash(year),
            "champion_probs": mc_results["champion_probs"],
            "final_four_probs": mc_results["final_four_probs"],
            "elite_eight_probs": mc_results["elite_eight_probs"],
            "sweet_sixteen_probs": mc_results["sweet_sixteen_probs"],
            "round_of_32_probs": mc_results["round_of_32_probs"],
            "final_four_by_region": final_four_by_region,
        }
        with open(mc_path, "w") as f:
            json.dump(mc_export, f)
        print(f"  Wrote {mc_path} for API")

    print("\nGenerating HTML...")
    html = generate_html(bracket_result, mc_results, bracket, config, args.sims, args.upset, quadrant_order, year=year)

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
