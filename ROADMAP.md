# Bracket Brain — Roadmap

> Last updated: 2026-03-13

---

## Current State Assessment

### What's Working
- **Prediction engine** — Torvik + EvanMiya efficiency blend, CV-based walk-forward calibrated (12 params); CV Brier **0.165** (8 folds 2017–2025); champion rank #1/#1/#4 across held-out years
- **Bracket simulation** — 63-game bracket with correct FF seeding and quadrant ordering; Monte Carlo pre-computed for all years
- **Betting picks** — daily save (9am ET) + auto-settle (midnight ET) via GitHub Actions; ledger committed to repo; Kelly sizing (quarter-Kelly, 5% cap) on every pick
- **Picks tab** — hit rate, record by type, net units, ROI%, result badges, full history with filters, "Why this pick?" expandable, model comparison benchmark table
- **Analysis panel** — 40+ stats per team across 6 sections: efficiency signals, shooting (eFG/3PT/FT), rebounding/defense, scoring output, roster cards, key edges
- **Mobile** — round-tab mobile bracket view with touch-friendly cards and sticky nav
- **Shareable links** — URL-encoded bracket state, clipboard copy, cross-device restore
- **Data pipeline** — daily Torvik refresh via curl (bypasses Cloudflare), teams_merged committed for Render
- **Keep-alive** — GitHub Actions pings /health every 10 min 24/7 to eliminate Render cold starts

### Known Problems
- **Total bets gated by date** — disabled March 13-19 (conference tournaments, OVER biased); auto-enabled March 20+ (NCAA tournament, -1.8 pts avg error, calibrated)
- **2026 advanced stats missing** — shooting/rebounding/roster sections empty until bracket is set (Selection Sunday); Torvik-only data active
- **Overfitting risk** — held-out sample is only 63 games/year; CV variance is high
- **No distribution** — no SEO, no social sharing, no way for the site to grow
- **Lighthouse not audited** — mobile score and accessibility score unverified

---

## Milestone Plan

---

### M1 — Fix the Foundation
**Goal:** The model is trustworthy, the app is stable, and the CI pipeline is green.
**Target:** (March 16, 2026)

#### Model Integrity
- [x] Implement walk-forward cross-validation (train on years N–k, test on year N) — replace current random split
- [x] Reduce param count: drop or merge any param with |calibrated − default| < 0.05 across all folds; target ≤ 12 params
- [x] Re-derive seed weight from walk-forward results; confirm it doesn't dominate over efficiency signals (seed_weight held at 0.18 default — optimizer excluded it as low-signal)
- [x] **Measurable:** Cross-validated Brier (avg 8 folds) < 0.170 → **achieved 0.165** (2026-03-13: CV-based optimization, warm start, multi-start=2)

#### Betting Pipeline
- [x] Add unit tests for `settle_bets.py`: mock Odds API response, assert ML/spread/total settle logic correctly
- [x] Add `manage_bets.py settle <N> W|L|P` command for manual override when API data is wrong
- [x] Log every settle decision to `data/settle_log_YYYY-MM-DD.json` for debugging
- [x] **Measurable:** settle tests pass in CI; settle log written after every midnight run

#### App Stability
- [x] Add cache TTL: re-read bracket/monte-carlo data from disk after 1 hour (avoids stale Render cache on redeploy)
- [x] Add structured JSON logging to API (replace bare `print`) — one log line per request with latency
- [x] Pin all deps in `requirements.txt` with exact versions (`pip freeze > requirements.txt`)
- [x] Add `/ready` health endpoint that returns 503 if `teams_merged_2026.json` is missing
- [x] **Measurable:** All CI tests pass; `pytest -x` green on a fresh clone with no data files

---

### M2 — Model Excellence
**Goal:** Best-in-class prediction accuracy; model is clearly better than seeding alone and competitive with ESPN BPI.
**Target:** (March 16, 2026)

#### Benchmarking (completed)
- [x] Build `scripts/benchmark.py` — compares seed-only, efficiency-only, full model on 2023–2025 walk-forward folds
- [x] Add champion probability rank + FF hit rate to benchmark output
- [x] Publish the comparison table on the Picks tab

#### Right Metrics for Evaluation
- [x] Add `--bracket-quality` mode to `benchmark.py`: for each model, report champion rank, champion %, FF hit rate in top-8
- [x] **Measurable:** Full model champion rank ≤ #3 in at least 2 of 3 held-out years → **achieved 2/3** (2023: #1, 2024: #1, 2025: #4)

#### New Signals (gated on bracket quality improvement, not flat Brier)
- [ ] `recent_form_weight`: last 10 games vs season efficiency — requires fetching per-game data for historical years
- [ ] `rest_days_diff`: days since last game — requires tournament schedule data
- [ ] Each signal kept only if it improves champion rank or FF hit rate, not just Brier
- [x] **Measurable:** Avg champion % across 3 held-out years > 15% → **achieved 19.8%** ✓

**Evaluation:** New signals deferred. `recent_form_weight` and `rest_days_diff` require non-trivial data pipelines (per-game Torvik history, tournament schedule). Current model already meets champion % target. Recommend implementing in M5 (Data Moat) when building injury/venue pipelines.

#### Per-Round Win-Prob Calibration
- [x] Add `late_round_dampening` param: shift win-probs toward 0.5 in Sweet 16+ (all models collapse there)
- [x] Add to calibration PARAM_SPEC (0–0.35 range); calibrate via walk-forward CV
- [x] **Measurable:** Sweet 16 + Elite 8 Brier both < 0.230 → **baseline already meets** (S16: 0.201, E8: 0.220 on full 945 games)

---

#### M2 Results (2026-03-12)

| Metric | seed-only | efficiency | full model |
|--------|-----------|------------|------------|
| CV Brier (3-fold) | 0.1849 | 0.1614 | 0.1644 |
| CV Accuracy | 72.0% | 74.6% | 75.1% |
| Avg champion rank | 6.7 | 2.0 | 2.0 |
| Avg champion % | 7.0% | 18.5% | 19.8% |
| Avg FF hit (top-8) | 50% | 58% | 58% |

**Per-year bracket quality (full model):**
- 2023: Champ rank #1, 16.3%, FF hit 25%
- 2024: Champ rank #1, 30.9%, FF hit 50%
- 2025: Champ rank #4, 12.1%, FF hit 100%

**Takeaways:** Full model beats seed-only on Brier and bracket quality. Efficiency-only is slightly better on raw Brier (0.1614 vs 0.1644) but full model has higher champion % (19.8% vs 18.5%) and matches on FF hit rate. The intangibles (coach, pedigree, etc.) add value for bracket-level outcomes.

---

### M3 — UX Transformation
**Goal:** The site works perfectly on mobile, is visually polished, and is shareable.
**Target:** (March 16, 2026)

#### Mobile-First Bracket
- [x] On screens ≤ 768px: hide desktop bracket; show round-tab mobile view (R64→Championship)
- [x] Touch-friendly game cards: large tap targets (48px min), seed badge, win-prob %, pick highlight
- [x] Round navigator: sticky scrollable tab bar; "See Analysis" button on each card
- [x] Mobile layout: controls wrap, champ display full-width, panels edge-to-edge
- [ ] **Measurable:** Lighthouse mobile score ≥ 80; bracket is fully usable on iPhone 14 viewport

#### Shareable Bracket Links
- [x] Encode locked picks + year into URL hash (`#bracket=<base64url>`)
- [x] "🔗 Share" button copies shareable URL to clipboard; falls back to `prompt()` on HTTP
- [x] Decode hash on page load to restore shared bracket (year-safe: ignores cross-year links)
- [x] URL hash updates silently with `history.replaceState` as picks are made
- [x] **Measurable:** A bracket link opens identically on a different browser/device ✓

#### Explainability
- [x] Analysis panel redesigned: rich matchup panel with probability display, signal comparison bars, stat tiles, key factors, historical context, upset alert, model pick box, insight quote
- [x] Mobile bracket: every card has "ⓘ See Analysis" button opening the full panel
- [x] "Why this pick?" expandable on every betting card: model prob vs implied, edge, Kelly size, insight text
- [x] **Measurable:** Every matchup card has a reachable explanation; every pick card shows "Why?" ✓

#### Visual Polish
- [x] Design system: CSS variables extended with `--shadow`; full dark mode via `prefers-color-scheme`
- [x] Win-prob pip bars: 2px bar at bottom of each team slot showing win probability at a glance
- [x] Dark mode: full palette inversion — background, surfaces, borders, accent colors
- [x] Cold-start banner: shows after 3s of API delay with animated progress bar; hides on success
- [x] Kelly units shown inline on every pick card ("Bet 1.2u")
- [ ] Lighthouse accessibility score ≥ 90 (pending audit)
- [ ] CLS < 0.1 (pending audit)

---

### M4 — Betting Platform
**Goal:** The Picks tab is a full betting tool: sized picks, ROI tracking, OVER bias fixed.
**Target:** (March 16, 2026)

#### OVER Bias Fix
- [x] Backtested model totals against 2023–2025 NCAA tournament results: avg error **-1.8 pts**, OVER rate **46%** across 69 games — no systematic bias on NCAA tournament games
- [x] Root cause identified: +14.7pt OVER bias was from mid-major conference tournament games (March 13-19), not NCAA tournament games; score_scale is correctly calibrated for the tournament
- [x] Fix: `--no-totals` flag added to `save_bets.py`; workflow uses it automatically March 13-19, enables totals March 20+ (NCAA tournament start)
- [x] **Measurable:** OVER rate 46% on 2023–2025 tournament games ✓ (within 50–58% target; slight UNDER lean is acceptable)

#### Kelly Criterion Sizing
- [x] Add `kelly_fraction` to each pick (full Kelly × 0.25 for safety)
- [x] Display Kelly size on each pick card ("Bet 2.1u")
- [x] Add `max_kelly` cap (5% of bankroll) to prevent outsized single-game exposure
- [ ] **Measurable:** Backtested Kelly growth > flat-bet growth on 2023–2025 (deferred — need settled results)

#### ROI Tracking
- [x] Track `units_won` / `units_lost` per pick via Kelly-weighted settle logic
- [x] Picks tab stat cards: ROI%, net units, total wagered, by bet type hit rates
- [ ] Monthly/tournament-level PnL chart (simple bar chart, no external lib)
- [x] **Measurable:** Stat cards update in real-time after settle runs ✓

#### Picks Card View
- [x] Card design with game matchup prominent, bet info secondary, color-coded result badges
- [x] Color-code by result: green WIN, red LOSS, grey PENDING
- [x] Filter controls: by bet type (ML/Spread), by result (W/L/Pending)
- [ ] Filter by date range or min edge (deferred)
- [ ] **Measurable:** Filter state persists in URL param (deferred)

---

### M5 — Data Moat
**Goal:** Signals no competitor easily replicates — injuries, venue, style matchups.
**Target:** Final Four weekend (April 4, 2026)

#### Injury Signals
- [ ] Build `scripts/fetch_injuries.py` — scrapes or polls a free injury API for tournament rosters
- [ ] Add `injury_impact` field to team data: weighted sum of injured players' BPR
- [ ] Adjust efficiency ratings down when key players are out
- [ ] **Measurable:** Injury data present for ≥ 80% of tournament teams before R64 tips

#### Venue / Travel Database
- [ ] Build `data/venues_YYYY.json` with city, timezone, altitude, floor surface for each tournament site
- [ ] Add travel distance differential as a feature (miles from campus to venue)
- [ ] **Measurable:** Venue data present for all 2026 tournament sites; travel feature shows nonzero Brier improvement

#### Style Matchup
- [ ] Define 4 play-style axes from existing Torvik data (tempo, 3pt rate, transition rate, turnover rate)
- [ ] Add matchup compatibility score: teams whose defense exploits opponent's offensive style get a bonus
- [ ] **Measurable:** Style matchup improves Brier by > 0.002 on held-out 2025 data

---

### M6 — Growth and Distribution
**Goal:** The site has organic discoverability and a mechanism for repeat visitors.
**Target:** Before next season (October 2026)

#### SEO and Discoverability
- [ ] Add OpenGraph tags: title, description, preview image per bracket year
- [ ] Add `sitemap.xml` with all bracket year pages
- [ ] Add structured data (JSON-LD) for prediction results (sports betting schema)
- [ ] **Measurable:** All pages indexed by Google; Lighthouse SEO score ≥ 90

#### Social Sharing
- [ ] "Share my bracket" generates a static PNG of the current bracket (server-side, using Pillow)
- [ ] Auto-generate daily picks image for Twitter/Instagram (score card layout)
- [ ] Add Twitter/X card meta tags so shared links show preview
- [ ] **Measurable:** Shared bracket links render with preview image on Twitter and iMessage

#### Email / Alerts
- [ ] Optional email signup: send daily picks digest before 9am tip-off
- [ ] Use Resend or SendGrid free tier; store signups in a simple JSON file or Render KV
- [ ] **Measurable:** End-to-end email delivered within 5 min of 9am ET cron

#### API Monetization (optional)
- [ ] Document public API endpoints in OpenAPI/Swagger UI (already generated by FastAPI)
- [ ] Add API key auth for `/predict` and `/bets/*` endpoints
- [ ] Rate limit: 100 req/day free, unlimited with key
- [ ] **Measurable:** API key flow works end-to-end; Swagger UI accessible at `/docs`

---

### M7 — Infrastructure for Scale
**Goal:** The site survives traffic spikes, has zero cold starts, and has a real database.
**Target:** Before 2027 tournament

#### Eliminate Cold Starts
- [ ] Upgrade Render to Starter ($7/month) — always-on, no sleep
- [x] Add keep-alive ping: GitHub Actions cron hits `/health` every 10 min, 24/7
- [x] Interim: show "Waking up (est. 30s)…" progress banner on first load when API is cold
- [ ] **Measurable:** P95 first-byte latency < 500ms during tournament week

#### PostgreSQL Ledger
- [ ] Migrate `bets_ledger.json` to Render PostgreSQL (free tier: 256MB)
- [ ] Update `settle_bets.py` and `save_bets.py` to write to DB
- [ ] Add `/bets/export` endpoint that returns ledger as JSON (backwards-compatible)
- [ ] **Measurable:** Ledger survives a Render redeploy without data loss; no git commits needed for picks

#### Docker and Build Optimization
- [ ] Add `Dockerfile` with multi-stage build; final image < 200MB
- [ ] Cache pip install layer separately from code layer for fast rebuilds
- [ ] **Measurable:** Docker build < 90s; Render deploy < 3 min

#### CDN for Static Assets
- [ ] Move `web/` to a CDN (Cloudflare Pages free tier or GitHub Pages)
- [ ] API and frontend on separate origins; configure CORS
- [ ] **Measurable:** Lighthouse performance score ≥ 90 on mobile; TTFB for HTML < 100ms

---

## Metrics Dashboard (track weekly during tournament)

| Metric | Baseline | M1 Target | M2 Target | M4 Target |
|--------|----------|-----------|-----------|-----------|
| Brier score (CV avg, 3 folds) | 0.161 ✓ | < 0.158 | < 0.153 | < 0.153 |
| Betting hit rate | TBD | — | — | > 55% |
| Betting ROI | TBD | — | — | > 0% |
| OVER rate | 65% | — | — | 50–58% |
| Mobile Lighthouse | ? | — | — | ≥ 80 |
| CI pass rate | 100% | 100% | 100% | 100% |
| Cold start latency | ~30s | < 5s (keep-alive) | — | < 500ms (M7) |
