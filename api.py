"""
Bracket Brain API — FastAPI wrapper around engine.py.

Endpoints:
  GET  /health                      — server status + data freshness
  GET  /ready                       — 503 if teams_merged_2026.json missing (for load balancers)
  GET  /teams/{year}                — all teams with stats for a given year
  POST /predict                     — predict one game
  GET  /bracket/{year}              — full bracket picks
  GET  /bracket/{year}/monte-carlo  — championship probabilities (N sims)

Usage:
  uvicorn api:app --reload
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import io
import json
import logging
import os
import sys
import contextlib
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import (
    predict_game, enrich_team, run_monte_carlo, load_bracket,
    generate_bracket_picks, load_teams_merged, _normalize_team_for_match,
    ModelConfig, DEFAULT_CONFIG,
)
from espn_scores import fetch_scores_for_picks, _scores_key as espn_scores_key
from best_bets import get_full_card_json

LEDGER_PATH = os.path.join(DATA_DIR, "bets_ledger.json")
CARD_LEDGER_PATH = os.path.join(DATA_DIR, "card_ledger.json")
# Cache: card endpoint hits the Odds API — 15 min TTL during game days
CARD_CACHE_TTL = 900

# Cache TTL: 1 hour — avoids stale Render cache on redeploy
CACHE_TTL_SECONDS = 3600
# Scores cache: 90 seconds for live score freshness
SCORES_CACHE_TTL = 90

# Cache: key -> {"result": ..., "cached_at": timestamp}
_cache: dict = {}

# Structured JSON logging
_log = logging.getLogger("api")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
_log.addHandler(_handler)
_log.setLevel(logging.INFO)

app = FastAPI(
    title="Bracket Brain",
    description="NCAA tournament prediction engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000)
    log_line = json.dumps({
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": latency_ms,
    })
    _log.info(log_line)
    return response


WEB_DIR = os.path.join(ROOT, "web")


def _cache_get(key: str):
    """Return cached result if not expired, else None."""
    entry = _cache.get(key)
    if not entry:
        return None
    cached_at = entry.get("cached_at", 0)
    if time.time() - cached_at > CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    return entry.get("result")


def _cache_set(key: str, result):
    _cache[key] = {"result": result, "cached_at": time.time()}


@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available_years() -> list[int]:
    years = []
    for fname in os.listdir(DATA_DIR):
        m = re.match(r"bracket_(\d{4})\.json$", fname)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def _data_freshness(year: int) -> Optional[str]:
    path = os.path.join(DATA_DIR, f"teams_merged_{year}.json")
    if not os.path.isfile(path):
        return None
    mtime = os.path.getmtime(path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _load_bracket_for_year(year: int):
    path = os.path.join(DATA_DIR, f"bracket_{year}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"No bracket found for {year}")
    bracket, ff_matchups, quadrant_order = load_bracket(path, data_dir=DATA_DIR, year=year)
    return bracket, ff_matchups, quadrant_order


def _lookup_team(name: str, year: int) -> dict:
    teams = load_teams_merged(DATA_DIR, year)
    if not teams:
        raise HTTPException(status_code=404, detail=f"No team data for {year}")
    key = _normalize_team_for_match(name)
    stats = teams.get(key)
    if stats is None:
        for k, v in teams.items():
            if len(key) <= len(k) and key in k and len(key) >= len(k) * 0.65:
                stats = v
                break
    if stats is None:
        raise HTTPException(status_code=404, detail=f"Team not found: '{name}' (year {year})")
    return stats


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    team_a: str
    team_b: str
    year: int = 2026
    game_site: Optional[str] = None  # "home_a", "home_b", or None (neutral)
    seed_a: Optional[int] = None
    seed_b: Optional[int] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/ready")
def ready():
    """Returns 503 if teams_merged_2026.json is missing (for load balancer health checks)."""
    path = os.path.join(DATA_DIR, "teams_merged_2026.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=503, detail="teams_merged_2026.json not found")
    return {"ready": True}


@app.get("/health")
def health():
    years = _available_years()
    current = max(years) if years else None
    return {
        "status": "ok",
        "version": "2.0.0",
        "available_years": years,
        "current_year": current,
        "data_updated_at": _data_freshness(current) if current else None,
    }


@app.get("/teams/{year}")
def get_teams(year: int):
    teams = load_teams_merged(DATA_DIR, year)
    if not teams:
        raise HTTPException(status_code=404, detail=f"No team data for {year}")
    # Return as sorted list
    out = sorted(teams.values(), key=lambda t: t.get("barthag", 0), reverse=True)
    return {"year": year, "count": len(out), "teams": out}


@app.post("/predict")
def predict(req: PredictRequest):
    raw_a = _lookup_team(req.team_a, req.year)
    raw_b = _lookup_team(req.team_b, req.year)

    # Apply seeds if provided
    if req.seed_a is not None:
        raw_a = dict(raw_a, seed=req.seed_a)
    if req.seed_b is not None:
        raw_b = dict(raw_b, seed=req.seed_b)

    team_a = enrich_team(raw_a)
    team_b = enrich_team(raw_b)

    result = predict_game(team_a, team_b, game_site=req.game_site)

    return {
        "team_a": team_a["team"],
        "team_b": team_b["team"],
        "seed_a": team_a.get("seed"),
        "seed_b": team_b.get("seed"),
        "win_prob_a": round(result["win_prob_a"], 4),
        "win_prob_b": round(result["win_prob_b"], 4),
        "predicted_margin": round(result["predicted_margin"], 1),
        "favorite": team_a["team"] if result["win_prob_a"] >= 0.5 else team_b["team"],
        "year": req.year,
    }


@app.get("/bracket/{year}")
def get_bracket(
    year: int,
    upset_aggression: float = Query(default=0.0, ge=0.0, le=1.0),
):
    cache_key = f"bracket_{year}_{upset_aggression:.2f}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    bracket, ff_matchups, quadrant_order = _load_bracket_for_year(year)
    with contextlib.redirect_stdout(io.StringIO()):
        picks = generate_bracket_picks(
            bracket,
            upset_aggression=upset_aggression,
            quadrant_order=quadrant_order,
        )

    # ff_pairs: [(TL_region, BL_region), (TR_region, BR_region)]
    qo = quadrant_order
    result = {
        "year": year,
        "upset_aggression": upset_aggression,
        "picks": picks,
        "quadrant_order": qo,
        "ff_pairs": [[qo[0], qo[3]], [qo[1], qo[2]]],
    }
    _cache_set(cache_key, result)
    return result


@app.get("/bets/today")
def get_bets_today():
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.isfile(LEDGER_PATH):
        return {"date": today, "picks": [], "available": False}
    with open(LEDGER_PATH) as f:
        ledger = json.load(f)
    picks = [p for p in ledger.get("picks", []) if p.get("date") == today]
    picks = sorted(picks, key=lambda p: p.get("commence_time", ""))
    return {"date": today, "picks": picks, "available": bool(picks)}


@app.get("/bets/history")
def get_bets_history():
    if not os.path.isfile(LEDGER_PATH):
        return {"picks": [], "stats": {}}
    with open(LEDGER_PATH) as f:
        return json.load(f)


@app.get("/bets/scores")
def get_bets_scores():
    """
    Fetch live/final scores from ESPN for games we have picks on.
    Returns { "home_team|away_team": { home_score, away_score, completed, status_detail, display_clock } }.
    Cached 90s to avoid hammering ESPN.
    """
    cache_key = "bets_scores"
    entry = _cache.get(cache_key)
    if entry and (time.time() - entry.get("cached_at", 0)) < SCORES_CACHE_TTL:
        return entry.get("result", {})

    if not os.path.isfile(LEDGER_PATH):
        return {"scores": {}}

    with open(LEDGER_PATH) as f:
        ledger = json.load(f)

    picks = ledger.get("picks", [])
    # Focus on recent picks (today + last 2 days)
    today = datetime.now().strftime("%Y-%m-%d")
    recent = [p for p in picks if p.get("date") and p.get("date") >= _days_ago(2)]

    scores_by_key = fetch_scores_for_picks(recent, days=2)

    # Build response keyed by pick's home_team|away_team for frontend lookup
    result = {"scores": {}}
    for p in recent:
        key = espn_scores_key(p.get("home_team", ""), p.get("away_team", ""))
        rec = scores_by_key.get(key)
        if rec:
            pick_key = f"{p['home_team']}|{p['away_team']}"
            result["scores"][pick_key] = {
                "home_score": rec.get("home_score"),
                "away_score": rec.get("away_score"),
                "completed": rec.get("completed", False),
                "status_detail": rec.get("status_detail", ""),
                "display_clock": rec.get("display_clock", ""),
                "period": rec.get("period", 0),
            }
            # Also add flipped key in case frontend looks up away|home
            pick_key_flip = f"{p['away_team']}|{p['home_team']}"
            result["scores"][pick_key_flip] = {
                "home_score": rec.get("away_score"),
                "away_score": rec.get("home_score"),
                "completed": rec.get("completed", False),
                "status_detail": rec.get("status_detail", ""),
                "display_clock": rec.get("display_clock", ""),
                "period": rec.get("period", 0),
            }

    _cache[cache_key] = {"result": result, "cached_at": time.time()}
    return result


def _days_ago(n):
    """Return YYYY-MM-DD for n days ago."""
    d = datetime.now(timezone.utc) - timedelta(days=n)
    return d.strftime("%Y-%m-%d")


@app.get("/benchmark")
def get_benchmark():
    """Return model comparison (seed-only vs efficiency vs full) for Picks tab."""
    path = os.path.join(DATA_DIR, "benchmark_results.json")
    if not os.path.isfile(path):
        return {"available": False, "message": "Run: python scripts/benchmark.py --bracket-quality"}

    with open(path) as f:
        data = json.load(f)
    data["available"] = True
    return data


@app.get("/bracket/{year}/monte-carlo")
def get_monte_carlo(
    year: int,
    sims: int = Query(default=10000, ge=100, le=100000),
):
    cache_key = f"mc_{year}_{sims}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Use pre-computed file when available (sims=10000 default)
    precomputed_path = os.path.join(DATA_DIR, f"monte_carlo_{year}.json")
    if sims == 10000 and os.path.isfile(precomputed_path):
        with open(precomputed_path) as f:
            result = json.load(f)
        _cache_set(cache_key, result)
        return result

    bracket, _, _ = _load_bracket_for_year(year)
    config = ModelConfig(num_sims=sims)
    with contextlib.redirect_stdout(io.StringIO()):
        mc = run_monte_carlo(bracket, config=config)

    result = {
        "year": year,
        "num_simulations": sims,
        "champion_probs": mc["champion_probs"],
        "final_four_probs": mc["final_four_probs"],
        "elite_eight_probs": mc["elite_eight_probs"],
        "sweet_sixteen_probs": mc["sweet_sixteen_probs"],
        "round_of_32_probs": mc["round_of_32_probs"],
    }
    _cache_set(cache_key, result)
    return result


@app.get("/bets/card")
def get_bets_card(year: int = Query(default=2026)):
    """Return today's full card — every NCAAB game with the model's best lean per market.

    Reads from saved daily file (data/card_YYYY-MM-DD.json) written by save_card.py /
    GitHub Actions.  Falls back to live Odds API fetch only if no saved file exists.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Serve from saved daily snapshot (committed by GH Actions each morning)
    daily_path = os.path.join(DATA_DIR, f"card_{today}.json")
    if os.path.isfile(daily_path):
        with open(daily_path) as f:
            data = json.load(f)
        games = data.get("games", [])
        return {"date": today, "games": games, "available": bool(games)}

    # Fall back to live Odds API if no saved file (e.g. before Actions runs)
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return {"games": [], "available": False,
                "message": "No card data yet. Run save_card.py each morning."}

    cache_key = f"card_{year}_live"
    entry = _cache.get(cache_key)
    if entry and (time.time() - entry.get("cached_at", 0)) < CARD_CACHE_TTL:
        return entry.get("result", {})

    games = get_full_card_json(api_key, year=year)
    result = {"date": today, "games": games, "available": bool(games)}
    _cache[cache_key] = {"result": result, "cached_at": time.time()}
    return result


@app.get("/bets/card/history")
def get_bets_card_history():
    """Return card ledger — all card picks with results and stats."""
    if not os.path.isfile(CARD_LEDGER_PATH):
        return {"picks": [], "stats": {}}
    with open(CARD_LEDGER_PATH) as f:
        return json.load(f)


@app.get("/bets/card/scores")
def get_bets_card_scores():
    """Live/final scores from ESPN for today's card games.  Cached 90s."""
    cache_key = "card_scores"
    entry = _cache.get(cache_key)
    if entry and (time.time() - entry.get("cached_at", 0)) < SCORES_CACHE_TTL:
        return entry.get("result", {})

    if not os.path.isfile(CARD_LEDGER_PATH):
        return {"scores": {}}

    with open(CARD_LEDGER_PATH) as f:
        ledger = json.load(f)

    picks = ledger.get("picks", [])
    recent = [p for p in picks if p.get("date") and p.get("date") >= _days_ago(2)]
    scores_by_key = fetch_scores_for_picks(recent, days=2)

    result = {"scores": {}}
    for p in recent:
        key = espn_scores_key(p.get("home_team", ""), p.get("away_team", ""))
        rec = scores_by_key.get(key)
        if rec:
            pick_key = f"{p['home_team']}|{p['away_team']}"
            result["scores"][pick_key] = {
                "home_score": rec.get("home_score"),
                "away_score": rec.get("away_score"),
                "completed": rec.get("completed", False),
                "status_detail": rec.get("status_detail", ""),
                "display_clock": rec.get("display_clock", ""),
                "period": rec.get("period", 0),
            }
    _cache[cache_key] = {"result": result, "cached_at": time.time()}
    return result
