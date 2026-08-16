# Bracket Brain — March Madness Prediction Engine

**Live demo: [bracket-brain.onrender.com](https://bracket-brain.onrender.com)**

A prediction engine that generates a complete 63-game NCAA tournament bracket — picks, projected spreads, win probabilities, and matchup analysis for every game — calibrated against 1,071 historical tournament games (2008–2025) and validated against a genuine, held-out 2026 season.

## Methodology

The model computes a win probability for each matchup from tempo-adjusted efficiency, schedule strength, coaching pedigree, and possession-level factors, then converts the projected margin to a probability via a Gaussian CDF blended with historical seed-performance priors.

Parameters are tuned with **leave-one-year-out cross-validated calibration**: each candidate parameter set is scored by averaging its Brier score across every tournament year held out in turn, and the search (`scipy.optimize.differential_evolution`) directly minimizes that cross-validated score rather than in-sample error. A true sequential walk-forward mode also exists (`scripts/calibrate.py --no-cv`, training strictly on prior years only) but is not the default and is not what produced the deployed model — the codebase's own docs mark it "legacy... may overfit." The config actually served is a separate final fit on all years except the single most recent one — currently 2026, which is used only in cross-validation and never touches the trained parameters.

**Cross-validated Brier score: 0.1661 · Accuracy: 74.8%** (1,071 games, 2008–2025, 2020 excluded — no tournament was played that year)

**2026 held out entirely from the final fit, scored as a true out-of-sample test: Brier 0.1339 · Accuracy 81.0%** (63 games) — the model generalizes to a season it never trained on at least as well as its in-sample average, not worse.

Against a pure seed-only baseline (always pick the better seed, probability from the historical seed-gap win rate) the model beats chalk by **+4.1 points of accuracy and a 14.5% lower Brier score** on 2008–2025, and by **+6.4 points of accuracy and a 20.0% lower Brier score** on the 2026 holdout specifically.

```bash
python backtest.py 2017 2019 2021 2023 2024 2025   # score picks against actual results
python scripts/reliability_diagram.py               # per-probability-bin calibration check
```

Backtest and reliability output isn't shipped in the repo (it's tens of thousands of lines of per-game data) — regenerate it with the commands above.

### Known limitations
Cross-validation folds are single tournament years (63 games each), so per-year Brier estimates carry real variance — the aggregate 1,071-game figure is more stable than any individual year's number. The "upset aggressiveness" slider is a user-facing heuristic that shifts which side of a close call gets picked; it doesn't change the underlying win probabilities. A tournament year still in progress (before its Championship game has been played) runs on incomplete injury and roster data, so accuracy improves as the season fills in — 2026 is now complete and reflects final-season data.

## What It Does

- **Interactive bracket** — traditional bracket layout you can click to make picks
- Picks every game in the bracket (all 63), not just the championship
- Projects a spread and score for each matchup
- Rates each pick with a confidence tier: Lock / Strong / Lean / Tossup
- **Pick provenance labeling** — for any year with completed games, each matchup is marked as an actual historical result or a model projection, with a running scoreboard of the model's own picks against real outcomes, overall and per round
- **Upset aggressiveness slider** — tune how many upsets the model picks (0% = chalk, 100% = chaos)
- **Manual pick locking** — click any team to lock your pick, then re-simulate the rest
- **Claude-powered analysis** — optional LLM-generated matchup insights (with caching)
- Runs Monte Carlo simulations for championship and Final Four probabilities

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data Pipeline

### Step 1: Torvik Ratings

```bash
curl -sL -o "data/torvik_YYYY_raw.csv" \
  "https://barttorvik.com/YYYY_team_results.csv" \
  -H "User-Agent: Mozilla/5.0"
python scripts/fetch_torvik.py --from-csv data/torvik_YYYY_raw.csv YYYY
```

### Step 2: Merge Team Data

```bash
python scripts/fetch_data.py YYYY
```

### Step 3: Bracket

```bash
# Projected bracket from BracketMatrix
python scripts/fetch_brackets.py projected 2026
cp data/bracket_2026_projected.json data/bracket_2026.json

# Or historical (danvk 2010-2024, Sports-Reference 2025+)
python scripts/fetch_brackets.py historical YYYY
python scripts/merge_bracket_stats.py YYYY
```

### Step 4: Run

```bash
python run.py                          # generates output/index.html
python run.py --sims 50000            # more Monte Carlo sims
python run.py --upset 0.3             # 30% upset aggressiveness
python run.py --bracket data/bracket_2025.json
```

### Step 5 (Optional): Claude Analysis

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python run.py                          # automatically uses Claude when key is set
# Or generate analysis separately:
python scripts/generate_analysis.py --data output/index_data.json --year 2026
```

Analysis is cached in `data/analysis_cache_YYYY.json` to avoid redundant API calls.

## Training and Calibration

### Extract historical results

```bash
python scripts/extract_results.py    # downloads 2008-2026 results (danvk + Sports-Reference)
```

Produces `data/results_all.json` with 1,134 games (18 tournament years, 2020 excluded) including teams, seeds, scores, winners, margins, and upset flags. Calibration itself still trains on 1,071 games (2008–2025) — the most recent complete year (2026) is automatically held out of the final fit and used only for cross-validation; see Methodology above.

### Calibrate model parameters

```bash
python scripts/calibrate.py
```

Optimizes `ModelConfig` parameters via leave-one-year-out cross-validated Brier score minimization (see Methodology above). Saves to `data/calibrated_config.json`.

### Backtest

```bash
python backtest.py 2017 2019 2021 2023 2024 2025
```

## How the Model Works

### Per-Game Prediction
1. Computes expected efficiency for each team (offense vs opponent defense, tempo-adjusted)
2. Adjusts for schedule strength, coaching pedigree, program history, and possession metrics
3. Calculates win probability via Gaussian CDF of the adjusted margin
4. Blends with historical seed performance priors (weight learned via calibration)

### Upset Aggressiveness
At aggression 0, always picks the favorite. At aggression > 0, uses stochastic sampling with a probability shift toward underdogs proportional to the seed difference.

### Interactive Bracket
The HTML output includes a full client-side prediction engine (JavaScript port of the Python model) that enables:
- Click any team to lock it as your pick
- Adjust upset slider and re-simulate
- See analysis for any matchup by clicking the game cell
- Champion and downstream picks update instantly

## Output Structure

`output/index.html` — Interactive bracket picker with traditional bracket layout, click-to-pick, upset slider, analysis panels, and Monte Carlo odds.

`output/index_data.json` — Raw data for all picks and Monte Carlo results.

## Project Structure

```
engine.py                    # Prediction model, bracket generation, calibrated config
run.py                       # Main entry — generates interactive bracket HTML
backtest.py                  # Score picks against actual historical results
scripts/
  extract_results.py         # Extract game results (danvk 2010-2024, SR 2025+)
  calibrate.py               # Optimize model parameters via cross-validated Brier score
  generate_analysis.py       # Claude-powered matchup analysis with caching
  fetch_torvik.py            # Torvik T-Rank CSV parser
  fetch_conf_tourney.py      # Conference tournament results (champions/finalists)
  fetch_data.py               # Merge team data sources
  fetch_brackets.py          # Historical + projected brackets
  merge_bracket_stats.py     # Enrich bracket with team stats
  reliability_diagram.py     # Per-probability-bin calibration check
  sources/
    sports_reference.py      # SR bracket scraper (brackets + results)
    danvk_brackets.py        # danvk GitHub data fetcher
    bracket_matrix.py        # BracketMatrix projected bracket scraper
data/
  calibrated_config.json     # Trained model parameters
  results_all.json           # 1,134 historical game results (2008-2026)
  analysis_cache_YYYY.json   # Cached Claude analyses
  torvik_YYYY.json           # Parsed team stats
  conf_tourney_YYYY.json     # Conference tournament results (optional)
  teams_merged_YYYY.json     # Merged team data
  bracket_YYYY.json          # Tournament brackets
output/
  index.html                 # Interactive bracket page
  index_data.json            # Raw output data
```

## Troubleshooting

### Torvik download fails
Use `curl` with a browser User-Agent header, or download in your browser. See Step 1.

### Bracket has placeholder stats
Run `python scripts/merge_bracket_stats.py YYYY` after building teams_merged.

### Want to re-calibrate
Run `python scripts/extract_results.py` then `python scripts/calibrate.py`.

### Claude analysis not working
Ensure `ANTHROPIC_API_KEY` is set. The script falls back to template analysis gracefully.

## Built By
Matt Gennarelli
