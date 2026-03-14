# Model Improvement Log

> Tracks experiments, results, and decisions for the Bracket Brain prediction model.

---

## Phase 0: Baseline Capture (2026-03-13)

### Calibration Baseline
- **Games:** 945 matchable
- **Accuracy:** 74.0% (699/945)
- **Brier score:** 0.1707
- **Log loss:** 0.5113
- **Spread MAE:** 8.6 pts
- **Total bias:** -0.6 pts, MAE: 13.2 pts

### Per-Round Breakdown (Baseline)
| Round       | Accuracy | Brier  |
|-------------|----------|--------|
| Round of 64 | 76.7%    | 0.1573 |
| Round of 32 | 74.6%    | 0.1742 |
| Sweet 16    | 63.3%    | 0.1936 |
| Elite 8     | 68.3%    | 0.2193 |
| Final Four  | 76.7%    | 0.1551 |
| Championship| 80.0%    | 0.1993 |

### Benchmark Baseline
- Stored in `docs/benchmark_baseline_2026-03-13.json`
- CV summary: seed-only, efficiency, full model Brier and accuracy
- Bracket quality: champion rank, champion %, FF hit rate per year

### Success Criteria for Future Phases
- CV Brier ≤ baseline + 0.002 (0.1707 + 0.002 = 0.1727 max acceptable)
- Champion rank and FF hit rate not worse
- All changes documented here

---

## Phase 1: Data Extension (2008–2009) — COMPLETED (2026-03-14)

**Results extraction:** Success. Extracted 63 games each for 2008 and 2009 (126 total). Combined: 1,071 games across 17 years.

**Torvik / teams_merged:** User added `torvik_2008_raw.csv` and `torvik_2009_raw.csv`. Processed with:
- `python scripts/fetch_torvik.py --from-csv data/torvik_2008_raw.csv 2008`
- `python scripts/fetch_torvik.py --from-csv data/torvik_2009_raw.csv 2009`
- `python scripts/fetch_data.py --no-fetch 2008` and `2009`

**Result:** 1,071 matchable games (up from 945). Calibration now uses 18 years: 2008–2026.

---

## Phase 2–4: Calibration Pipeline Upgrades (2026-03-13)

**Implemented:**
- CV-based Phase 1 optimization (optimize on cross-validated Brier, not in-sample)
- Log loss objective (`--objective logloss`)
- Default multi-start=2, maxiter=100, popsize=16
- Warm start from calibrated_config.json
- `--no-cv` flag for legacy per-fold train optimization
- `--recency-weight` and `--round-weight` for weighted scoring
- `--save-report PATH` for JSON report output
- `scripts/run_full_eval.sh` for calibration + benchmark + log append

---

## Phase 5: Final Validation (2026-03-13)

**Run:** maxiter=25, popsize=10, multi-start=1 (reduced for speed)

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| Brier (in-sample) | 0.1685 | 0.1682 | +0.0003 |
| Accuracy | 73.4% | 74.2% | +0.8% |
| CV Brier (honest) | — | 0.1647 | — |

**Success criteria:** CV Brier ≤ baseline + 0.002 (0.1707) → **0.1647 ✓**  
Champion rank and FF hit rate: to be verified via benchmark.

**Config saved to** `data/calibrated_config.json`. Report: `docs/cal_report_phase5.json`.

---

## Experiments

*(Entries added as phases complete)*
