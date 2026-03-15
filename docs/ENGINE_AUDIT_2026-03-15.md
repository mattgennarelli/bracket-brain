# Bracket Brain 2026 Pre-Drop Audit (Engine + Data + Betting)

_Date:_ 2026-03-15  
_Scope:_ model quality, training data sanity, 2008/2009 integration, betting quality controls, web integration risk areas, and a concrete forward execution plan.

## 1) Executive Readout

- **Core model is stable and test-clean** (55/55 tests pass).
- **Tournament prediction quality remains strong overall** (2010–2025 aggregate backtest: **73.3% accuracy, 0.1674 Brier**).
- **Biggest current model weakness is still upset calibration in close games**, especially 5/12, 6/11, 7/10 and Elite 8 confidence.
- **2008/2009 data is now active and appears structurally usable**, but has **name-normalization gaps** (e.g., WKU / UT Arlington variants) that can silently degrade matching and calibration quality if unaddressed.
- **Betting integration is the highest-risk layer today**: ledger shows **negative ROI**, with **3-star picks currently underperforming badly** and totals being especially weak.
- **Primary immediate priority (next 24h):** tighten betting selection guards and confidence gating so the site avoids pushing weak high-conviction picks before bracket lock.

---

## 2) What was audited

### Code / architecture reviewed
- `engine.py` (prediction stack, normalization, bracket enrichment, upset logic).
- `backtest.py` (historical scoring and total-bias diagnostics).
- `scripts/calibrate.py`, `scripts/benchmark.py`, `scripts/diagnose_model.py` (training/eval diagnostics).
- `run.py` and betting data outputs (`data/bets_ledger.json`, card files).
- Existing plans in `docs/MILESTONE_PLAN.md` and `ROADMAP.md`.

### Checks executed
- Unit/integration tests.
- Data completeness validation.
- Multi-year backtest sweep.
- Benchmark comparison (seed-only vs efficiency vs full model).
- Global diagnostic report over all training/eval games.
- Focused 2008/2009 training-data sanity checks.
- Betting ledger quality review (star tiers, favorites vs dogs, bet-type performance).

---

## 3) Current performance baseline (from this audit run)

## 3.1 Engine quality
- **Backtest (2010–2025):** 693/945 (**73.3%**) with **0.1674 Brier**.
- **Recent years:**
  - 2023: 71.4%, Brier 0.1853
  - 2024: 74.6%, Brier 0.1797
  - 2025: 81.0%, Brier 0.1229

Interpretation:
- Baseline predictive strength is real.
- Year-to-year volatility persists (notably 2023/2024 upset-heavy paths).

## 3.2 Diagnostics
- Overall on 2008–2025 dataset (1,071 games): **0.1646 Brier**, **73.9% accuracy**.
- Hardest segment remains upset detection:
  - Upsets Brier: **0.3855** (very high)
  - Chalk Brier: **0.0770**
- Close games are noisy:
  - |margin| < 2: Brier 0.2429
  - 2–5: Brier 0.2418
- Elite 8 still weak calibration zone (Brier 0.2094).

## 3.3 Benchmark snapshot (2025 year check)
- Efficiency-only outperformed full model on plain Brier in 2025 (0.1151 vs 0.1229), indicating some full-model factors can overfit in certain years.
- Full model still best on champion ranking in that sample (rank 3, 15.4% champion probability for actual champ).

Implication:
- Keep full model, but add stricter regularization and objective balancing (already aligned with M1/M2 roadmap).

---

## 4) 2008 + 2009 data audit findings

## 4.1 Structural sanity
- `results_2008.json` and `results_2009.json` each contain 63 games and full round distribution.
- Champion games match historical outcomes (2008 Kansas, 2009 North Carolina).
- Seed ranges are valid across both years.

## 4.2 Training impact sanity check
- With current calibrated config:
  - **All years (2008–2025, 1,071 games):** Brier 0.1646, acc 73.95%
  - **Without 2008/2009 (945 games):** Brier 0.1674, acc 73.33%
  - **Only 2008/2009 (126 games):** Brier 0.1433, acc 78.57%

Interpretation:
- Adding 2008/2009 is not causing obvious degradation; metrics improve slightly on aggregate.
- 2008/2009 may be somewhat “easier” samples; monitor over-weighting in calibration.

## 4.3 Data quality issues discovered
- Name matching misses exist for historical teams (e.g., **WKU**, **Texas–Arlington**, **Cal State Fullerton**, **Cal State Northridge**, **Miss Valley St.**).
- These misses can reduce feature richness for affected games and contaminate calibration signal.

Action taken in this patch:
- Added explicit normalization aliases in `engine.py` for historical/team-name variants to improve 2008/2009 (and similar) team joins.

---

## 5) Betting system audit findings (highest priority)

From `data/bets_ledger.json` current aggregate stats:
- **33 settled picks**
- **16W / 17L** (48.5% hit)
- **ROI: -25.1%**
- By type:
  - ML: +6.87% ROI
  - Spread: -70.03% ROI
  - Total: -52.28% ROI

Critical behavior flags:
- **3-star picks: 0-3** (0% hit) → confidence ladder is currently inverted/noisy at top tier.
- **ML favorites vs underdogs split:**
  - Favorites: 90% hit
  - Underdogs: 30% hit
  This suggests edge-model thresholding is letting too many weak dog/value claims through.
- High model-probability picks (>=0.80 proxy): only 66.7% hit in this small sample (too low for that confidence bin).

Conclusion:
- The betting recommendation layer needs immediate risk controls and confidence recalibration before scaling.

---

## 6) 2026 readiness risks

### High risk (must address first)
1. **Overconfident upset/favorite framing in close games** can produce bad public picks and trust loss.
2. **3-star confidence mapping is not reliable** in live ledger outcomes.
3. **Spread/total selection rules are too permissive** relative to measured ROI.

### Medium risk
1. 2026 bracket data has two unmatched teams in validation (Tennessee State, Portland State) in projected bracket context.
2. Possible factor over-stacking in specific year profiles (efficiency-only occasionally better on Brier).

### Low risk
1. Core API/app stability (tests pass).
2. Historical training ingestion generally consistent post-2008/2009 expansion.

---

## 7) Forward plan (thorough, priority-ordered)

## Phase A — Ship-safe controls before bracket lock (today)

1. **Confidence-tier recalibration (betting picks).**
   - Re-map stars by empirical win-rate bins from settled history + backtest-derived expected value buckets.
   - Hard cap 3★ issuance until minimum sample confidence achieved.

2. **Underdog/favorite policy guards.**
   - Add stricter gating for ML dogs: require stronger edge + probability floor + anti-volatility checks.
   - Add “no-pick” zone for noisy spreads/totals when model disagreement or score uncertainty is high.

3. **Bet-type throttling by live performance.**
   - Temporarily down-weight/disable totals if rolling ROI or hit rate stays below threshold.
   - Keep ML as primary lane until spread/total confidence is revalidated.

4. **Expose confidence diagnostics in UI/API.**
   - Show why a pick earned its star tier (edge source, volatility penalties, calibration bin).

## Phase B — Engine optimization for upset realism (next 2–4 days)

1. **Complete M6 underseeded indicator implementation + test.**
   - Compute and persist tournament `eff_rank`.
   - Add as upset tolerance signal with guardrails to avoid runaway dog bias.

2. **Close-game calibration objective expansion (M2).**
   - Add objective that jointly optimizes Brier + upset-Brier + bracket quality (champ rank/champ%).
   - Use reduced Monte Carlo for calibration loops to keep runtime manageable.

3. **Round-aware uncertainty regularization.**
   - Re-check late-round stdev inflation and dampening with 2008–2025 folds.
   - Target Elite 8 Brier improvement without hurting R64 upset calibration.

4. **Factor ablation sweep (full vs efficiency-only parity control).**
   - Systematically test each intangible factor’s incremental value by fold.
   - Freeze or shrink factors with unstable year-over-year contribution.

## Phase C — Data moat + quality hardening (next 1–2 weeks)

1. **2008/2009 validation artifact.**
   - Add `docs/data_validation_2008_2009.md` with spot-check evidence, alias mapping list, and final pass/fail checklist.

2. **Conference tournament momentum (M3).**
   - Build ingestion + merge + calibration for conf-tourney momentum signal.

3. **Automated data QA gates in CI.**
   - Add checks for name-match coverage %, placeholder stat rate, and per-year outlier detection.

4. **Betting model audit table generation.**
   - Daily generated report: hit rate/ROI by type, stars, favorite/dog, and edge deciles.

## Phase D — Product integration + trust UX (next sprint)

1. **Pick explainability cards** (feature-level contribution snippets).
2. **Public reliability dashboard** (probability bins vs actual outcomes).
3. **Risk-aware defaults** in UI (safe mode before tournament starts; aggressive mode opt-in).

---

## 8) Suggested concrete target metrics

For launch-readiness gates:
- Engine CV Brier (2008–2025): maintain <= 0.165.
- Upset Brier: improve from 0.3855 toward <= 0.36.
- Elite 8 Brier: improve from 0.2094 toward <= 0.20.
- 3★ picks: enforce positive expected value and observed hit-rate > overall baseline over rolling sample.
- Totals ROI: keep disabled/limited until trailing sample shows non-negative ROI.

---

## 9) Immediate next actions checklist

- [ ] Implement star-tier recalibration + issuance caps.
- [ ] Add underdog ML gating and no-pick filters for spreads/totals.
- [ ] Add daily betting diagnostics export and wire to web “Picks” panel.
- [ ] Run quick calibration ablation (efficiency-only parity test + top factor trims).
- [ ] Execute full pre-drop smoke run (`validate_data`, `pytest`, backtest slice, benchmark year check).

