# Bracket Brain — Milestone Plan

> Addresses open items from calibration/benchmark work and outlines implementation roadmap.

---

## 1. Possession & Conference Params — Keep, Investigate

**Decision:** Keep possession and conference in the model for now.

**Why they go to zero:** The optimizer (differential evolution) drives `possession_edge_max_bonus` and `conf_rating_max_bonus` to near-zero because:
- Bounds allow 0 (e.g. `possession_edge_max_bonus`: 0–8, `conf_rating_max_bonus`: 0–4)
- CV Brier improves when these are small; the signals may be weak or noisy
- Different runs produce different params at zero due to optimizer variance (multi-start, random seed)

**Actions:**
- [x] **M1.1** Run calibration with fixed non-zero floor (e.g. min 0.1) for possession/conf — added `--min-floor 0.1` flag
- [x] **M1.2** Document conference strength grading in `fetch_data.py`; formula measures per-team conference boost; alternatives: raw conf_adj, KenPom conf ratings
- [x] **M1.3** Add a short note in `MODEL_IMPROVEMENT_LOG.md` that possession/conf are intentionally kept despite calibration driving them low; revisit after M1.1–M1.2

---

## 2. Bracket-Quality Objective — Optimize to Champion Rank + Champ % + FF Hit

**Goal:** Add optimization targets beyond Brier: champion rank, champion %, Final Four hit rate.

**Approach:** Extend `calibrate.py` to support a composite objective that includes bracket-quality metrics. Run Monte Carlo on a held-out year (or subset of years) and optimize params to improve champion rank, champion %, and FF hit rate.

**Actions:**
- [ ] **M2.1** Add `--objective bracket-quality` (or `brier+f4`) that combines Brier with champion rank, champion %, FF hit rate
- [ ] **M2.2** Define composite formula, e.g. `score = w1*brier + w2*champ_rank_penalty + w3*(1 - champ_pct) + w4*(1 - ff_hit_rate)` with tunable weights
- [ ] **M2.3** Integrate `_compute_bracket_quality` (or equivalent) into calibration loop — requires running Monte Carlo per candidate config, which is expensive; consider evaluating only on 1–2 held-out years or a reduced sim count
- [ ] **M2.4** Benchmark: compare Brier-only vs Brier+F4 calibration on champion rank and FF hit rate

---

## 3. Conference Tournament Momentum — Implementation Plan

**Goal:** Use conference tournament results as a momentum signal (recent form before NCAA tournament).

### 3.1 Data

| Source | Format | Availability |
|-------|--------|--------------|
| Torvik / BartTorvik | Conference tournament results | Manual scrape or API |
| Sports Reference (CBB) | Conf tourney brackets/results | Web scrape |
| KenPom | Conf tourney pages | May require subscription |

**Actions:**
- [ ] **M3.1** Research and document where to fetch conference tournament results (dates, matchups, winners) for 2008–2026
- [ ] **M3.2** Create `scripts/fetch_conf_tourney.py` or extend `fetch_data.py` to pull conf tourney results into `data/conf_tourney_YYYY.json`
- [ ] **M3.3** Define momentum metric: e.g. `conf_tourney_momentum = (actual_finish - expected_finish) / expected_finish` or binary "won conf tourney" / "lost in final" / "lost earlier"
- [ ] **M3.4** Merge conf tourney momentum into `teams_merged_YYYY` (new field `conf_tourney_momentum` or `conf_tourney_result`)
- [ ] **M3.5** Add `calc_conf_tourney_bonus(team, config)` in `engine.py` and wire into factor list
- [ ] **M3.6** Add config param `conf_tourney_max_bonus` and include in calibration
- [ ] **M3.7** Validate: compare calibration/Brier with and without conf tourney momentum

---

## 4. Florida 2025 Clarification

**Context:** 2025 champion was Florida. Benchmark shows:
- **Seed-only:** champion_rank 1, champion_pct 9.0%
- **Efficiency:** champion_rank 4, champion_pct 11.65%
- **Full model:** champion_rank 4, champion_pct 14.6%

**Clarification:** The full model gave Florida the highest champion % among the three models (14.6%). If using seed-only, Florida was the 4th 1-seed (by region/bracket order). Seed-only’s champion_rank 1 for Florida is driven by Monte Carlo path — all 1-seeds get similar pairwise win probs, but Florida’s bracket path yielded more simulated titles.

**Actions:**
- [ ] **M4.1** Add a note in `docs/benchmark_baseline_*.txt` or `MODEL_IMPROVEMENT_LOG.md`: "2025: Full model gave Florida highest champ % (14.6%); seed-only had Florida as 4th 1-seed but path-dependent Monte Carlo ranked them #1"
- [ ] **M4.2** Optionally add `champion_pct` to benchmark summary table so it’s clear full model had highest Florida %

---

## 5. Validate 2008/2009 Data

**Goal:** Confirm the added 2008/2009 Torvik and results data is correct.

**Actions:**
- [x] **M5.1** Spot-check: verify 5–10 teams from `torvik_2008.json` / `torvik_2009.json` against Torvik/BartTorvik historical pages (barthag, adj_o, adj_d)
- [x] **M5.2** Cross-check `results_2008.json` / `results_2009.json` against NCAA official brackets (winners, seeds, regions)
- [x] **M5.3** Run calibration with 2008/2009 excluded; compare CV Brier and round-level stats to full dataset — large swings may indicate data issues
- [x] **M5.4** Document validation results in `MODEL_IMPROVEMENT_LOG.md` or `docs/data_validation.md`

---

## 6. Underseeded Upset Indicator

**Idea:** When the underdog is significantly underseeded (efficiency rank &lt; actual seed), add an upset indicator.

**Definition:** `eff_rank` = 1-indexed rank of team by barthag among all tournament teams (1 = best). Underseeded = `eff_rank < seed` (e.g. eff_rank 6, seed 12 → team is better than seed suggests).

**Actions:**
- [x] **M6.1** Compute `eff_rank` when building bracket: rank all 68 teams by barthag descending, assign rank 1 to best. Store in team dict during `enrich_bracket_with_teams` or equivalent
- [x] **M6.2** Add indicator #16 in `_calc_upset_tolerance_bonus`: if `dog.get("eff_rank")` and `dog.get("seed")` exist and `dog["eff_rank"] < dog["seed"]`, count as indicator (dog is underseeded)
- [x] **M6.3** Ensure `eff_rank` is available in `teams_merged` or bracket enrichment for all years with Torvik data
- [ ] **M6.4** Re-run calibration and benchmark; check if upset Brier or champion rank improves

---

## 7. Milestone Summary

| ID | Milestone | Priority | Effort |
|----|-----------|----------|--------|
| M1 | Possession/conf: keep, investigate bounds & grading | Medium | 1–2 days |
| M2 | Bracket-quality objective (champ rank, %, FF) | High | 2–3 days |
| M3 | Conference tournament momentum | High | 2–4 days |
| M4 | Florida 2025 documentation | Low | &lt;1 hr |
| M5 | 2008/2009 data validation | Medium | 1 day |
| M6 | Underseeded upset indicator | Medium | 0.5–1 day |

**Suggested order:** M4 (quick doc) → M5 (data validation) → M6 (underseeded) → M1 (investigation) → M2 (bracket objective) → M3 (conf tourney momentum).

---

## Appendix: Key File Paths

- `scripts/calibrate.py` — calibration, PARAM_SPEC, objectives
- `scripts/benchmark.py` — champion rank, champ %, FF hit rate
- `scripts/fetch_data.py` — `compute_conf_strength_scores`, merge logic
- `scripts/compute_momentum.py` — momentum from season_games
- `engine.py` — `_calc_upset_tolerance_bonus`, `calc_conf_bonus`, `enrich_bracket_with_teams`
- `data/calibrated_config.json` — current config
- `docs/MODEL_IMPROVEMENT_LOG.md` — experiment log
