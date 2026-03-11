# Bracket Brain — March Madness Bracket Picker

A prediction engine that generates a complete 63-game bracket with picks, projected spreads, confidence ratings, and matchup analysis for every game. Trained on 945 historical tournament games (2010-2025) with calibrated model parameters.

## What It Does

- **Interactive bracket** — traditional bracket layout you can click to make picks
- Picks every game in the bracket (all 63), not just the championship
- Projects a spread and score for each matchup
- Rates each pick with a confidence tier: Lock / Strong / Lean / Tossup
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
python scripts/extract_results.py    # downloads 2010-2025 results (danvk + Sports-Reference)
```

Produces `data/results_all.json` with 945 games including teams, seeds, scores, winners, margins, and upset flags.

### Calibrate model parameters

```bash
python scripts/calibrate.py
```

Optimizes `ModelConfig` parameters against all historical games using Brier score minimization. Saves to `data/calibrated_config.json`.

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
  calibrate.py               # Optimize model parameters via Brier score
  generate_analysis.py       # Claude-powered matchup analysis with caching
  fetch_torvik.py            # Torvik T-Rank CSV parser
  fetch_data.py              # Merge team data sources
  fetch_brackets.py          # Historical + projected brackets
  merge_bracket_stats.py     # Enrich bracket with team stats
  sources/
    sports_reference.py      # SR bracket scraper (brackets + results)
    danvk_brackets.py        # danvk GitHub data fetcher
    bracket_matrix.py        # BracketMatrix projected bracket scraper
data/
  calibrated_config.json     # Trained model parameters
  results_all.json           # 945 historical game results (2010-2025)
  analysis_cache_YYYY.json   # Cached Claude analyses
  torvik_YYYY.json           # Parsed team stats
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
