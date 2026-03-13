# Bracket Brain — Roadmap

> Last updated: 2026-03-12

---

## Current State Assessment

### What's Working
- **Prediction engine** — Torvik + EvanMiya efficiency blend, walk-forward calibrated (12 params); in-sample Brier 0.171 (74.2% / 945 games), cross-validated Brier **0.161** (avg of 2023/2024/2025 folds: 0.194 / 0.171 / 0.118)
- **Bracket simulation** — 63-game bracket with correct FF seeding, Monte Carlo pre-computed for all years
- **Betting picks** — daily save (9am ET) + auto-settle (midnight ET) via GitHub Actions; ledger committed to repo
- **Picks tab** — hit rate, record by type, result badges, full history
- **Data pipeline** — daily Torvik refresh via curl (bypasses Cloudflare), teams_merged committed for Render

### Known Problems
- **Cold start** — Render free tier takes 30s to wake up; users bounce before the bracket loads
- **Mobile** — bracket SVG/table layout is completely broken on small screens
- **OVER bias** — 30/46 picks (65%) are OVER even after score_scale fix; tempo scaling incomplete
- **Overfitting risk** — walk-forward validation done, params reduced to 12; held-out sample is only 63 games/year so variance is high
- **No explainability** — picks and predictions show numbers but not *why*
- **Picks UX** — no Kelly sizing shown, no PnL in units, no per-round breakdown
- **No distribution** — no SEO, no social sharing, no way for the site to grow

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
- [x] **Measurable:** Cross-validated Brier (avg 3 folds) < 0.170 → **achieved 0.161** (per-fold: 2023=0.194, 2024=0.171, 2025=0.118; 2025 was unusually chalk so average is the honest number)

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

- [ ] Add champion probability rank + FF hit rate to benchmark output
- [ ] Publish the comparison table on the Picks tab

#### Right Metrics for Evaluation
- [ ] Add `--bracket-quality` mode to `benchmark.py`: for each model, report champion rank, champion %, FF hit rate in top-8
- [ ] **Measurable:** Full model champion rank ≤ #3 in at least 2 of 3 held-out years (achieved: #2, #1, #2 ✓)

#### New Signals (now gated on bracket quality improvement, not flat Brier)
- [ ] `recent_form_weight`: last 10 games vs season efficiency — requires fetching per-game data for historical years
- [ ] `rest_days_diff`: days since last game — requires tournament schedule data
- [ ] Each signal kept only if it improves champion rank or FF hit rate, not just Brier
- [ ] **Measurable:** Avg champion % across 3 held-out years > 15% (currently: 16.1% avg across 2023/24/25 ✓)

#### Per-Round Win-Prob Calibration
- [ ] Add `late_round_dampening` param: shift win-probs toward 0.5 in Sweet 16+ (all models collapse there)
- [ ] Calibrate per round via walk-forward CV
- [ ] **Measurable:** Sweet 16 + Elite 8 Brier both < 0.230 (currently 0.267 and 0.248)

---

### M3 — UX Transformation
**Goal:** The site works perfectly on mobile, is visually polished, and is shareable.
**Target:** (March 16, 2026)

#### Mobile-First Bracket
- [ ] Rebuild bracket layout using CSS Grid instead of fixed-width tables
- [ ] On screens < 768px: show bracket as vertical scrollable list grouped by round
- [ ] Touch-friendly game cards: large tap targets, swipe between rounds
- [ ] **Measurable:** Lighthouse mobile score ≥ 80; bracket is fully usable on iPhone 14 viewport

#### Shareable Bracket Links
- [ ] Encode current bracket state (all 63 picks) into URL hash (`#bracket=<base64>`)
- [ ] Add "Copy link" button that copies the current bracket URL to clipboard
- [ ] Decode hash on page load to restore shared bracket
- [ ] **Measurable:** A bracket link opens identically on a different browser/device

#### Explainability
- [ ] Add "Why?" tooltip/drawer on every matchup showing: efficiency edge, seed differential, recent form, predicted margin
- [ ] Add "Why this pick?" panel on each betting card showing the factors that drove the edge
- [ ] **Measurable:** Every matchup card has a reachable explanation; user study shows picks feel justified

#### Visual Polish
- [ ] Design system: define 6-color palette, 3 font sizes, consistent spacing scale in CSS variables
- [ ] Replace raw numbers with visual indicators: win-prob bar, seed badge, trend arrow
- [ ] Dark mode: respect `prefers-color-scheme`
- [ ] Loading state: skeleton cards while data fetches; "Waking up…" banner on cold start with progress indicator
- [ ] **Measurable:** Lighthouse accessibility score ≥ 90; no layout shift on load (CLS < 0.1)

---

### M4 — Betting Platform
**Goal:** The Picks tab is a full betting tool: sized picks, ROI tracking, OVER bias fixed.
**Target:** (March 16, 2026)

#### OVER Bias Fix
- [ ] Stratify historical games by tempo quartile; compute `score_scale` per quartile
- [ ] Apply tempo-stratified scale in `engine.py` when tempo data is available in `teams_merged`
- [ ] Re-run pick generation on 2023–2025 games; verify OVER rate drops below 55%
- [ ] **Measurable:** Backtested OVER rate 50–58% on 2023–2025 tournament games

#### Kelly Criterion Sizing
- [ ] Add `kelly_fraction` to each pick (full Kelly × 0.25 for safety)
- [ ] Display Kelly size on each pick card (e.g., "Bet 2.1 units")
- [ ] Add `max_kelly` cap (5 units) to prevent outsized single-game exposure
- [ ] **Measurable:** All picks show kelly size; backtested Kelly growth > flat-bet growth on 2023–2025

#### ROI Tracking
- [ ] Track `units_won` / `units_lost` per pick (using kelly size or flat 1 unit)
- [ ] Picks tab stat cards: ROI%, total units wagered, units won/lost, by bet type
- [ ] Monthly/tournament-level PnL chart (simple bar chart, no external lib)
- [ ] **Measurable:** Stat cards update in real-time after settle runs; ROI visible per bet type

#### Picks Card View
- [ ] Redesign picks list as cards (not table rows): game matchup prominent, bet info secondary
- [ ] Color-code by result: green WIN, red LOSS, yellow PUSH, grey PENDING
- [ ] Filter controls: by date range, bet type, result, min edge
- [ ] **Measurable:** Picks card view renders correctly on mobile; filter state persists in URL param

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
- [ ] Add keep-alive ping: GitHub Actions cron hits `/health` every 10 min during tournament
- [ ] Interim: show "Waking up (est. 30s)…" progress banner on first load when API is cold
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
