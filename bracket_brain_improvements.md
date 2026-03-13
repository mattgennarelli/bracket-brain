# Bracket Brain — Engine & Betting Improvements
# Instructions for Claude Code

You are improving a March Madness prediction engine (Bracket Brain). The codebase lives in this repo. The core files are:
- `engine.py` — prediction model, bracket generation, Monte Carlo sims
- `scripts/calibrate.py` — walk-forward parameter optimization
- `scripts/best_bets.py` — compares model predictions to Vegas lines
- `scripts/save_bets.py` — saves daily picks to ledger
- `scripts/settle_bets.py` — settles picks against actual scores
- `backtest.py` — scores model against historical results

Read engine.py, calibrate.py, backtest.py, and best_bets.py fully before starting. Understand the flow: `_predict_base_score` → factor adjustments → `predict_game` → `generate_bracket_picks`.

Work through these phases in order. After each phase, run `python backtest.py 2023 2024 2025` and `pytest -x` to confirm nothing is broken and metrics improve (or at least don't regress).

---

## PHASE 1: Fix the OVER Bias (score_scale per round + tempo)

### Problem
`score_scale` is a single global constant (0.942) applied to all predicted scores. Tournament games score lower than regular season, but not uniformly — R64 games between mismatched seeds play near regular-season pace, while Elite 8 grinds between two elite defenses score much lower. This flat discount causes 65% of total picks to be OVER.

### Goal
Replace the single `score_scale` with per-round scaling that also accounts for the tempo of the specific matchup. OVER rate on backtested 2023-2025 totals should drop below 55%.

### Steps

1. **Add per-round score_scale fields to `ModelConfig`:**
   ```
   score_scale_r64: float = 0.960
   score_scale_r32: float = 0.950
   score_scale_s16: float = 0.935
   score_scale_e8: float = 0.920
   score_scale_ff: float = 0.910
   ```
   Keep the existing `score_scale: float = 0.942` as a fallback for when round context isn't available.

2. **Add a helper function `_get_round_score_scale(round_name, config)`** that maps round names ("Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship") to the corresponding config field. Return `config.score_scale` as default for unrecognized rounds.

3. **Add tempo-based score adjustment.** When both teams have `adj_tempo` data, compute a tempo factor:
   ```python
   avg_tempo = (team_a["adj_tempo"] + team_b["adj_tempo"]) / 2
   tempo_ratio = avg_tempo / config.national_avg_tempo
   # Slow games (tempo_ratio < 1) get additional scoring discount
   # Fast games (tempo_ratio > 1) get less discount
   tempo_adjust = 0.5 * (tempo_ratio - 1.0)  # e.g., 5% slow = -0.025 additional discount
   effective_scale = round_scale * (1.0 + tempo_adjust)
   # Clamp to reasonable range
   effective_scale = max(0.88, min(1.0, effective_scale))
   ```

4. **Modify `predict_game` signature** to accept an optional `round_name=None` parameter. When provided, use the round-specific score_scale for the predicted_score_a and predicted_score_b output. When None, fall back to `config.score_scale`.

5. **Thread round_name through the call chain.** In `generate_bracket_picks` and `_make_pick_dict`, pass the `round_name` to `predict_game`. In `backtest.py`'s `score_all_games`, pass the game's `round_name` to `predict_game`. In `best_bets.py`, pass `round_name=None` (Vegas bets don't have round context for regular season games, and tournament games should pass the actual round).

6. **Add `score_scale` params to calibrate.py's PARAM_SPEC** so the optimizer can tune them:
   ```python
   ("score_scale_r64", 0.90, 1.00),
   ("score_scale_r32", 0.90, 1.00),
   ("score_scale_s16", 0.88, 0.98),
   ("score_scale_e8", 0.86, 0.96),
   ("score_scale_ff", 0.85, 0.95),
   ```

### Checkpoint
Run `python backtest.py 2023 2024 2025`. Verify the per-round predicted scores are now lower for later rounds. The overall Brier should stay ≤ 0.165. The OVER rate check is only meaningful once we add total backtesting (Phase 5), but the predicted totals should visibly decrease for E8/FF games.

---

## PHASE 2: Cap Factor Margin + Round Stdev Inflation

### Problem
12 intangible factor bonuses (experience, coach, pedigree, etc.) plus 4 margin adjustments (possession, FT, foul_rate, SOS) all stack linearly with no cap. In extreme cases, a blue-blood team with a great coach, high experience, momentum, etc. can accumulate +10-15 points of adjustments on top of the efficiency margin. This is unrealistic — no intangible package is worth 15 points.

Additionally, later tournament rounds (E8, FF) have miscalibrated win probabilities. The model gives 65-70% to favorites when true probability is closer to 55-60%. Inflating stdev for later rounds pushes probabilities toward 0.5 where they belong.

### Goal
Cap total factor_margin at ±6 points using a soft cap (tanh compression). Add per-round stdev inflation. Elite 8 Brier should improve from 0.231 toward 0.20.

### Steps

1. **Add to `ModelConfig`:**
   ```
   factor_margin_cap: float = 6.0
   round_stdev_inflation_r64: float = 1.00
   round_stdev_inflation_r32: float = 1.03
   round_stdev_inflation_s16: float = 1.06
   round_stdev_inflation_e8: float = 1.10
   round_stdev_inflation_ff: float = 1.12
   ```

2. **In `predict_game`, after computing `factor_margin`**, apply a soft cap using tanh:
   ```python
   # Soft-cap factor margin to prevent intangible stacking
   cap = config.factor_margin_cap
   factor_margin = math.tanh(factor_margin / cap) * cap
   ```
   This smoothly compresses: a raw +4 becomes ~3.6, a raw +8 becomes ~5.7, a raw +15 becomes ~6.0. It preserves ordering but prevents extremes.

3. **Add a helper `_get_round_stdev_inflation(round_name, config)`** similar to the score_scale helper.

4. **In `predict_game`, modify the game_stdev calculation:**
   ```python
   round_inflation = _get_round_stdev_inflation(round_name, config)
   game_stdev = config.base_scoring_stdev * math.sqrt(poss / config.national_avg_tempo) * vol * round_inflation
   ```

5. **Add the stdev inflation params to PARAM_SPEC in calibrate.py** with reasonable bounds:
   ```python
   ("round_stdev_inflation_e8", 1.0, 1.25),
   ("round_stdev_inflation_ff", 1.0, 1.30),
   ```
   Don't calibrate R64/R32/S16 inflation — keep those fixed to reduce param count.

### Checkpoint
Run `python backtest.py 2023 2024 2025`. Check the per-round breakdown. Elite 8 Brier should improve. Win probabilities for later-round games should be closer to 0.5 (less extreme). Overall accuracy may dip 1-2% (fewer confident picks) but Brier should improve (better calibrated).

---

## PHASE 3: Evaluate Dropping the Seed Prior Blend

### Problem
`_blend_probs` mixes the efficiency-derived probability with a seed-based historical prior at 18% weight. But the efficiency model already implicitly captures seed information (higher seeds have better efficiency). The calibrator excluded `seed_weight` as low-signal, suggesting it's not helping. Double-counting seeds can bias the model toward chalk and hurt upset detection.

### Goal
Test whether removing or reducing the seed blend improves CV Brier.

### Steps

1. **Create a branch or backup.** This is an experiment — we may revert.

2. **Run backtest with current seed_weight (0.18):**
   ```bash
   python backtest.py 2023 2024 2025
   ```
   Record: overall Brier, per-round Brier, accuracy, number of correct upsets.

3. **Set `seed_weight` to 0.0 in `data/calibrated_config.json`** (or in ModelConfig default). Run backtest again. Compare.

4. **Also test seed_weight = 0.10** (half current). Run backtest. Compare.

5. **Pick whichever value produces the best CV Brier.** If 0.0 and 0.18 are within 0.002 Brier of each other, prefer 0.0 (simpler model, one fewer parameter).

6. **If dropping seed_weight, also remove it from PARAM_SPEC** in calibrate.py so the optimizer doesn't waste time on it.

### Checkpoint
Document the results of all three configs. If seed_weight = 0.0 wins or ties, commit it. The model should still pick heavy favorites in 1v16 matchups even without the seed prior — the efficiency gap does that work.

---

## PHASE 4: Add Kelly Criterion Sizing to Betting Pipeline

### Problem
Every bet is flat (1 unit) regardless of edge size. A 7% edge and a 25% edge get the same allocation. This is suboptimal for bankroll growth.

### Goal
Add Kelly fraction to each pick. Display it in the picks output. Use quarter-Kelly for safety.

### Steps

1. **Add a `kelly_fraction` function to `best_bets.py`:**
   ```python
   def kelly_fraction(model_prob, decimal_odds, fraction=0.25):
       """Compute fractional Kelly bet size.
       
       Args:
           model_prob: our estimated probability of winning
           decimal_odds: payout in decimal format (e.g., 2.0 for even money)
           fraction: Kelly fraction (0.25 = quarter Kelly, conservative)
       
       Returns:
           Fraction of bankroll to bet (0.0 if no edge), capped at 5%.
       """
       # Full Kelly: f = (p * odds - 1) / (odds - 1)
       if decimal_odds <= 1.0:
           return 0.0
       edge = model_prob * decimal_odds - 1.0
       if edge <= 0:
           return 0.0
       full_kelly = edge / (decimal_odds - 1.0)
       sized = full_kelly * fraction
       return min(sized, 0.05)  # hard cap at 5% of bankroll
   ```

2. **In the bet generation loop (where ML/spread/total bets are created)**, compute kelly for each bet. For ML bets, use model_prob and the moneyline decimal odds. For spread bets, estimate the probability of covering (use the model margin vs spread to derive a cover probability via CDF). For totals, similarly estimate over/under probability.

   For spread cover probability:
   ```python
   # model_margin is how much we think team wins by
   # vegas_spread is the line (negative = favorite)
   # cover_margin = model_margin + vegas_spread (for home team bet)
   # P(cover) = Phi(cover_margin / game_stdev)
   from math import erf, sqrt
   def cover_prob(cover_margin, stdev=11.0):
       return 0.5 * (1.0 + erf(cover_margin / (stdev * sqrt(2))))
   ```

3. **Add `kelly_size`, `kelly_units` fields to each pick dict.** `kelly_units` = kelly_fraction * 100 (so 2.5% = 2.5 units on a 100-unit bankroll).

4. **In `save_bets.py`, include kelly_size in the saved pick.**

5. **In `settle_bets.py`, compute `units_won` and `units_lost`:**
   - Win: `units_won = kelly_units * (decimal_odds - 1)`
   - Loss: `units_lost = kelly_units`
   - Push: 0

6. **Update `compute_stats` in settle_bets.py** to track total units wagered, units won, units lost, and ROI% (= net_units / total_wagered).

7. **In the print_report output**, show Kelly size next to each pick:
   ```
   ★★★ [ML] Duke (-180)  edge: +12.3%  kelly: 3.2 units
   ```

### Checkpoint
Run `python scripts/best_bets.py --api-key ... --year 2026` (or with test data). Verify kelly_size appears on each pick. Verify that higher-edge bets get larger Kelly sizing. Verify the 5% cap works (no pick exceeds 5 units on a 100-unit bankroll).

---

## PHASE 5: Add Spread and Total Backtesting

### Problem
The backtest only evaluates win/loss prediction accuracy. But the betting pipeline also makes spread and total picks. Without backtesting those, we have no idea if spread/total picks would be profitable historically.

### Goal
Add ATS (against-the-spread) and total backtesting to backtest.py using historical data.

### Steps

1. **Create `scripts/compute_historical_lines.py`** that generates approximate historical lines from actual results:
   - For each historical game in results_all.json, the model's predicted margin IS our "model line"
   - We can approximate "Vegas lines" by using the seed-implied spread (e.g., 1v16 ~= 20 pts, 5v12 ~= 4 pts). This isn't perfect but gives a rough ATS backtest.
   - For totals, the actual combined score is the ground truth. The model's predicted total is our line.

2. **Add `score_spread_picks` and `score_total_picks` functions to backtest.py:**
   ```python
   def score_total_picks(actual_games, teams, config):
       """For each historical game, predict the total and compare to actual."""
       over_picks = 0
       over_correct = 0
       under_picks = 0
       under_correct = 0
       for g in actual_games:
           # ... run predict_game, get predicted_total
           actual_total = g["score_a"] + g["score_b"]
           if predicted_total > actual_total:
               over_picks += 1
               # model predicted higher than actual = OVER pick
               # was model right? only if we'd have picked OVER vs some line
           # Track predicted vs actual totals for bias analysis
       return {"over_rate": over_picks / total, "mae": mean_abs_error, ...}
   ```

3. **In the backtest output, add a section showing:**
   - Total predicted vs actual: MAE, bias (average predicted - actual)
   - OVER rate: what % of games did the model predict higher than actual?
   - Margin predicted vs actual: MAE, bias
   - Per-round breakdown of all the above

4. **This is diagnostic, not a betting backtest.** We can't reconstruct actual Vegas lines for historical games without paid data. But the bias analysis tells us if the model systematically over/under-predicts scores and margins, which directly maps to betting edge.

### Checkpoint
Run `python backtest.py 2023 2024 2025`. The new output should show total/margin bias. If OVER rate is still >58% after Phase 1 changes, the score_scale values need further tuning. Target: predicted total bias < ±2 points, OVER rate 48-55%.

---

## PHASE 6: Raise Betting Edge Thresholds

### Problem
Default thresholds (7% ML edge, 5pt spread edge, 8pt total edge) are too loose for an NCAAB model with only 945 training games. The model's probability estimates have substantial error, so small "edges" over the market are likely noise.

### Goal
Raise thresholds and add confidence-based filtering so only genuinely strong edges become picks.

### Steps

1. **Update default thresholds in `best_bets.py`:**
   ```python
   DEFAULT_ML_EDGE = 0.10       # was 0.07 — raise to 10%
   DEFAULT_SPREAD_EDGE = 7.0    # was 5.0 — raise to 7 pts
   DEFAULT_TOTAL_EDGE = 10.0    # was 8.0 — raise to 10 pts
   ```

2. **Add a confidence filter based on model certainty.** Only make a bet when the model is relatively confident in its own prediction. Add a parameter:
   ```python
   DEFAULT_MIN_MODEL_CONFIDENCE = 0.58  # don't bet ML if model prob < 58%
   ```
   In the ML edge check, after computing model_prob and edge, also require:
   ```python
   if model_prob < args.min_confidence:
       continue  # model too uncertain, skip even if edge exists
   ```

3. **Add a star-rating minimum for auto-saves.** In `save_bets.py`, only save picks that are ★★ or better (skip ★ picks). This reduces noise in the ledger.

4. **Update the help text and CLI args** to document the new defaults.

### Checkpoint
Run `python scripts/best_bets.py` with test data. Verify fewer picks are generated (higher quality, not quantity). The star ratings should cluster at ★★ and ★★★.

---

## PHASE 7: Add Calibration Reliability Diagram

### Problem
Brier score is a single number that doesn't tell you WHERE the model is miscalibrated. A reliability diagram (predicted prob bins vs actual win rate) reveals if the model is overconfident, underconfident, or well-calibrated at different probability levels.

### Goal
Add a reliability diagram generator that can be run after calibration or backtest.

### Steps

1. **Create `scripts/reliability_diagram.py`:**
   ```python
   """
   Generate a calibration reliability diagram from backtest results.
   
   Usage:
     python scripts/reliability_diagram.py 2023 2024 2025
   
   Outputs: data/reliability_diagram.json with binned calibration data
   """
   ```

2. **The script should:**
   - Load results + team data for each year
   - Run predict_game on every historical matchup
   - Bin predictions into buckets: 50-55%, 55-60%, 60-65%, 65-70%, 70-75%, 75-80%, 80-85%, 85-90%, 90-95%, 95-100%
   - For each bin: count predictions, count actual wins, compute actual win rate
   - Output a JSON with the bins + a simple ASCII table to stdout:
     ```
     Predicted    N_games    Actual_win%    Calibration_error
     50-55%       45         52.1%          +0.1%  (good)
     55-60%       38         58.2%          +0.7%  (good)
     60-65%       52         59.3%          -3.2%  (overconfident)
     ...
     ```

3. **Key diagnostic: if actual win rate is consistently BELOW predicted probability, the model is overconfident.** This means you should:
   - Inflate stdev (Phase 2 helps)
   - Reduce factor bonuses
   - Raise betting thresholds even further
   
   If actual win rate is consistently ABOVE, the model is underconfident and you can bet more aggressively.

4. **Save the JSON output** so it can optionally be displayed on the Picks tab later.

### Checkpoint
Run `python scripts/reliability_diagram.py 2023 2024 2025`. Review the table. Every bin's calibration error should be < 5%. If any bin is off by >5%, that's a zone where the model needs adjustment.

---

## PHASE 8: Expand Walk-Forward CV to More Folds

### Problem
Current CV uses only 3 test folds (2023, 2024, 2025) with ~63 games each. A single upset going differently can swing a fold's Brier by 0.01+. More folds = more stable estimate.

### Steps

1. **In `calibrate.py`'s `calibrate_walk_forward` function**, change the test years from just the last 3 to all years from 2017 onward that have both results and team data:
   ```python
   # Current: test_years = [2023, 2024, 2025]
   # New: test_years = [y for y in sorted(pairs_by_year.keys()) if y >= 2017]
   ```
   This gives up to 7+ folds (2017, 2019, 2021, 2022, 2023, 2024, 2025 — skip 2020 no tournament).

2. **Ensure the training set for each fold only includes years STRICTLY before the test year.** The current `_fold_pairs` function already does this — verify it.

3. **Report per-fold Brier and the average.** The average across 7 folds is a much more stable estimate than across 3.

4. **The optimizer should still train on the LARGEST training set** (all years before the latest test year) for the final saved config. The per-fold metrics are for evaluation only.

### Checkpoint
Run `python scripts/calibrate.py`. The per-fold Brier scores should be reported for each test year. The average CV Brier should be stable (likely 0.16-0.18 range). The 2023 fold may have higher Brier (more upsets that year) while 2025 is lower (chalk year).

---

## PHASE 9: Add score_scale to Calibration

### Problem
`score_scale` is the single most impactful parameter for total/over-under predictions, but it's a fixed constant — the calibrator doesn't optimize it. Even the per-round variants from Phase 1 should be calibratable.

### Steps

1. **Add `score_scale` to PARAM_SPEC:**
   ```python
   ("score_scale", 0.88, 1.00),
   ("score_scale_r64", 0.90, 1.00),
   ("score_scale_r32", 0.90, 1.00),
   ("score_scale_s16", 0.88, 0.98),
   ("score_scale_e8", 0.86, 0.96),
   ("score_scale_ff", 0.85, 0.95),
   ```

2. **Modify the score_model function in calibrate.py** to also track predicted total vs actual total for each game (if actual scores are available). Add a total_bias metric to the output:
   ```python
   total_errors = []
   for each game:
       predicted_total = result["predicted_score_a"] + result["predicted_score_b"]
       actual_total = actual_score_a + actual_score_b  # if available
       if actual_total:
           total_errors.append(predicted_total - actual_total)
   metrics["total_bias"] = sum(total_errors) / len(total_errors) if total_errors else 0
   metrics["total_mae"] = sum(abs(e) for e in total_errors) / len(total_errors) if total_errors else 0
   ```

3. **Consider adding total_bias to the optimization objective.** Currently the objective is pure Brier score (which only cares about win probability). To also optimize score prediction accuracy, use a blended objective:
   ```python
   # Brier optimizes win probability calibration
   # total_mae optimizes score prediction accuracy
   # Blend: primarily Brier, with small weight on total accuracy
   objective = brier_score + 0.001 * total_mae
   ```
   The 0.001 weight means total accuracy is a tiebreaker, not the primary goal. Adjust if needed.

### Checkpoint
Run `python scripts/calibrate.py`. The optimized score_scale values should vary by round. Check that total_bias in the output is close to 0 (±2 points). The per-round score_scales should show a decreasing pattern (R64 highest, FF lowest).

---

## PHASE 10: Silent Fallback Warning for Missing Team Data

### Problem
When `enrich_team` can't find a team in the merged data, it silently returns a team with default/zero stats. This produces completely wrong predictions with no warning.

### Steps

1. **In the `enrich_team` function**, add a check after enrichment:
   ```python
   import logging
   logger = logging.getLogger("bracketbrain")
   
   # At the end of enrich_team, before returning:
   adj_o = team.get("adj_o")
   adj_d = team.get("adj_d")
   if adj_o is None or adj_d is None or (adj_o == _BAD_DEFAULTS.get("adj_o") and adj_d == _BAD_DEFAULTS.get("adj_d")):
       team_name = team.get("team", "UNKNOWN")
       logger.warning(f"Team '{team_name}' has no real efficiency data — using defaults. Prediction will be unreliable.")
   ```

2. **In `best_bets.py`'s team lookup**, when a team can't be matched, already logs to `unmatched`. Also add a prominent warning in the bet output if any bet involves a team with default stats.

3. **In `predict_game`**, if BOTH teams have default stats, return a result with `win_prob_a = 0.5` and add a `"warning": "both teams have default stats"` field. This prevents garbage predictions from becoming confident picks.

### Checkpoint
Manually test with a fake team name that won't match any real team. Verify the warning appears in logs. Verify predict_game returns 50/50 when both teams are unknown.

---

## FINAL VALIDATION

After all phases are complete, run the full validation suite:

```bash
# 1. All tests pass
pytest -x

# 2. Backtest shows improvement
python backtest.py 2017 2019 2021 2023 2024 2025

# 3. Calibration with expanded CV
python scripts/calibrate.py

# 4. Reliability diagram looks good
python scripts/reliability_diagram.py 2023 2024 2025

# 5. Best bets generates reasonable output
python scripts/best_bets.py --api-key $ODDS_API_KEY --year 2026
```

**Target metrics after all phases:**
- CV Brier (avg across all folds): ≤ 0.165
- Elite 8 per-round Brier: ≤ 0.210 (down from 0.231)
- Overall accuracy: ≥ 73%
- Total prediction bias: ≤ ±2 points
- OVER rate on backtested games: 48-55%
- Reliability diagram: all bins within 5% calibration error
- All tests pass
- Kelly sizing appears on every bet
- Betting thresholds raised (10% ML, 7pt spread, 10pt total)
