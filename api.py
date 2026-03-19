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

import collections
import hashlib
import io
import json
import logging
import os
import sys
import contextlib
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import (
    predict_game, enrich_team, run_monte_carlo, load_bracket,
    generate_bracket_picks, load_teams_merged, _normalize_team_for_match,
    get_matchup_analysis_display,
    ModelConfig, DEFAULT_CONFIG,
)
from espn_scores import fetch_scores_for_picks, _scores_key as espn_scores_key
from best_bets import get_full_card_json
from odds_provider import get_api_key

LEDGER_PATH = os.path.join(DATA_DIR, "bets_ledger.json")
CARD_LEDGER_PATH = os.path.join(DATA_DIR, "card_ledger.json")
# Cache: card endpoint hits the Odds API — 15 min TTL during game days
CARD_CACHE_TTL = 900

# Cache TTL: 1 hour — avoids stale Render cache on redeploy
CACHE_TTL_SECONDS = 3600
# Scores cache: 90 seconds for live score freshness
SCORES_CACHE_TTL = 90

# Cache: key -> {"result": ..., "cached_at": timestamp}
# Uses OrderedDict so we can evict the oldest entry when size limit is hit.
_CACHE_MAX_ENTRIES = 50
_cache: collections.OrderedDict = collections.OrderedDict()

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

_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://localhost:3000,http://127.0.0.1:8000"
).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
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


def _cache_get(key: str, ttl: int = CACHE_TTL_SECONDS):
    """Return cached result if not expired, else None. Moves hit to end (LRU)."""
    entry = _cache.get(key)
    if not entry:
        return None
    cached_at = entry.get("cached_at", 0)
    if time.time() - cached_at > ttl:
        _cache.pop(key, None)
        return None
    # Move to end so it's the most recently used
    _cache.move_to_end(key)
    return entry.get("result")


def _cache_set(key: str, result):
    """Store result in cache, evicting the oldest entry if at capacity."""
    if key in _cache:
        _cache.move_to_end(key)
    _cache[key] = {"result": result, "cached_at": time.time()}
    while len(_cache) > _CACHE_MAX_ENTRIES:
        _cache.popitem(last=False)  # evict oldest


# ---------------------------------------------------------------------------
# Rate limiting: per-IP request tracking for expensive endpoints
# ---------------------------------------------------------------------------
_RATE_LIMIT_WINDOW = 60  # seconds
_MC_RATE_LIMIT = 10       # max Monte Carlo requests per IP per minute
_rate_limit_store: dict = {}  # ip -> [(timestamp, ...)]


def _check_rate_limit(ip: str, limit: int = _MC_RATE_LIMIT) -> bool:
    """Return True if the request is allowed, False if rate limit exceeded."""
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW
    hits = _rate_limit_store.get(ip, [])
    hits = [t for t in hits if t > window_start]
    if len(hits) >= limit:
        _rate_limit_store[ip] = hits
        return False
    hits.append(now)
    _rate_limit_store[ip] = hits
    return True


def _load_config(num_sims: int = 10000) -> ModelConfig:
    """Load ModelConfig with calibrated parameters (same as run.py)."""
    config = ModelConfig(num_sims=num_sims)
    cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
    if os.path.isfile(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        for k, v in cal.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return config


def _config_mtime() -> str:
    """Return mtime of calibrated_config.json for cache busting."""
    cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
    if os.path.isfile(cal_path):
        return str(int(os.path.getmtime(cal_path)))
    return "0"


def _add_final_four_by_region(result: dict, year: int) -> dict:
    """Compute final_four_by_region from bracket + final_four_probs. Modifies result in place."""
    ff_probs = result.get("final_four_probs", {})
    if not ff_probs:
        result["final_four_by_region"] = {}
        return result
    try:
        bracket, _, _ = _load_bracket_for_year(year)
    except Exception:
        result["final_four_by_region"] = {}
        return result
    by_region = {}
    for region, teams in bracket.items():
        team_probs = []
        for seed, team in teams.items():
            t = team.get("team") if isinstance(team, dict) else None
            if t and t in ff_probs:
                team_probs.append((t, ff_probs[t]))
        team_probs.sort(key=lambda x: -x[1])
        by_region[region] = [(t, round(p, 4)) for t, p in team_probs[:8]]
    result["final_four_by_region"] = by_region
    return result


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


def _data_hash_and_meta(year: int) -> dict:
    """Return data hash and bracket meta for diagnostic (compare local vs Render)."""
    out = {"data_hash": None, "quadrant_order": None, "sample_teams": {}}
    h = hashlib.sha256()
    files = [
        os.path.join(DATA_DIR, f"bracket_{year}.json"),
        os.path.join(DATA_DIR, f"teams_merged_{year}.json"),
        os.path.join(DATA_DIR, "calibrated_config.json"),
    ]
    for path in files:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(f.read())
    out["data_hash"] = h.hexdigest()[:16]

    try:
        bracket, _, quadrant_order = _load_bracket_for_year(year)
        out["quadrant_order"] = quadrant_order
        for rname, teams in list(bracket.items())[:2]:
            out["sample_teams"][rname] = [t.get("team") for s, t in list(teams.items())[:4]]
    except Exception:
        pass
    return out


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
    # Deep copy to prevent mutation of cached/shared data between requests
    import copy
    return copy.deepcopy(stats)


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


class SimulateRequest(BaseModel):
    upset_aggression: float = 0.0
    locked_picks: Dict[str, str] = Field(default_factory=dict)  # game_id -> team_name


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
    meta = _data_hash_and_meta(current) if current else {}
    return {
        "status": "ok",
        "version": "2.0.0",
        "available_years": years,
        "current_year": current,
        "data_updated_at": _data_freshness(current) if current else None,
        "data_hash": meta.get("data_hash"),
        "quadrant_order": meta.get("quadrant_order"),
        "sample_teams": meta.get("sample_teams"),
    }


@app.get("/debug/picks-sample")
def debug_picks_sample(year: int = 2026, upset_aggression: float = 0.0):
    """Return first 8 R64 picks for direct local vs Render comparison."""
    bracket, _, quadrant_order = _load_bracket_for_year(year)
    config = _load_config()
    with contextlib.redirect_stdout(io.StringIO()):
        result = generate_bracket_picks(
            bracket,
            config=config,
            upset_aggression=upset_aggression,
            quadrant_order=quadrant_order,
            data_dir=DATA_DIR,
            year=year,
        )
    picks = [p for p in result["picks"] if p.get("round") == 64][:8]
    return {
        "year": year,
        "upset_aggression": upset_aggression,
        "champion": result["champion"],
        "quadrant_order": quadrant_order,
        "sample_picks": [
            {"region": p["region"], "team_a": p["team_a"], "team_b": p["team_b"],
             "pick": p["pick"], "win_prob_a": p.get("win_prob_a")}
            for p in picks
        ],
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
    cache_key = f"bracket_{year}_{upset_aggression:.2f}_{_config_mtime()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    bracket, ff_matchups, quadrant_order = _load_bracket_for_year(year)
    config = _load_config()
    with contextlib.redirect_stdout(io.StringIO()):
        picks = generate_bracket_picks(
            bracket,
            config=config,
            upset_aggression=upset_aggression,
            quadrant_order=quadrant_order,
            data_dir=DATA_DIR,
            year=year,
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


@app.post("/bracket/{year}/simulate")
def simulate_bracket(year: int, req: SimulateRequest):
    """Generate bracket picks respecting locked picks. Use when user has manually
    locked some picks and wants to re-simulate the rest."""
    bracket, ff_matchups, quadrant_order = _load_bracket_for_year(year)
    config = _load_config()
    with contextlib.redirect_stdout(io.StringIO()):
        bracket_result = generate_bracket_picks(
            bracket,
            config=config,
            upset_aggression=req.upset_aggression,
            quadrant_order=quadrant_order,
            data_dir=DATA_DIR,
            year=year,
            locked_picks=req.locked_picks,
        )
    qo = quadrant_order
    return {
        "year": year,
        "upset_aggression": req.upset_aggression,
        "picks": bracket_result,
        "quadrant_order": qo,
        "ff_pairs": [[qo[0], qo[3]], [qo[1], qo[2]]],
    }


@app.get("/analyze")
def analyze_matchup_endpoint(
    team_a: str = Query(...),
    team_b: str = Query(...),
    year: int = Query(default=2026),
    seed_a: Optional[int] = Query(default=None),
    seed_b: Optional[int] = Query(default=None),
    region: Optional[str] = Query(default=None),
    round_name: Optional[str] = Query(default=None),
    injury_overrides: Optional[str] = Query(default=None),
):
    """Return full matchup analysis for display. Used when matchup is not in
    pre-generated picks (e.g. downstream game after user changed an upstream pick).
    injury_overrides: JSON string mapping "team_name|player_name" -> new_status.
      e.g. '{"Duke|Kyle Filipowski":"out","Duke|Jared McCain":"healthy"}'
      Use status "healthy" to remove a player from the injury list entirely.
    """
    raw_a = _lookup_team(team_a, year)
    raw_b = _lookup_team(team_b, year)
    if seed_a is not None:
        raw_a = dict(raw_a, seed=seed_a)
    if seed_b is not None:
        raw_b = dict(raw_b, seed=seed_b)
    config = _load_config()
    # Always recompute injury_impact for both teams (the stored value in teams_merged
    # may be stale — e.g. computed with old BPR data or different config)
    from engine import calc_injury_penalty
    # Apply injury status overrides from user
    if injury_overrides:
        import json as _json
        try:
            overrides = _json.loads(injury_overrides)
        except (ValueError, TypeError):
            overrides = {}
        for team_dict, team_name in [(raw_a, team_a), (raw_b, team_b)]:
            if "injuries" not in team_dict:
                continue
            new_injuries = []
            for inj in team_dict["injuries"]:
                key = f"{team_name}|{inj.get('player', '')}"
                if key in overrides:
                    new_status = overrides[key]
                    if new_status.lower() in ("healthy", "active", "playing", "remove"):
                        continue  # Remove from injury list
                    inj = dict(inj, status=new_status)
                new_injuries.append(inj)
            team_dict = dict(team_dict, injuries=new_injuries)
            team_dict["injury_impact"] = calc_injury_penalty(team_dict, config)
            if team_name == team_a:
                raw_a = team_dict
            else:
                raw_b = team_dict
    # Recompute injury_impact for both teams to ensure consistency
    # (even teams without overrides need fresh computation)
    raw_a = dict(raw_a, injury_impact=calc_injury_penalty(raw_a, config))
    raw_b = dict(raw_b, injury_impact=calc_injury_penalty(raw_b, config))
    analysis = get_matchup_analysis_display(
        raw_a, raw_b, data_dir=DATA_DIR, year=year,
        region=region, round_name=round_name,
        config=config,
    )
    return analysis


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
        return {"picks": [], "stats": {}, "model_epoch": None}
    with open(LEDGER_PATH) as f:
        data = json.load(f)
    # Attach model_epoch: the date calibrated_config.json was last written.
    # Picks generated before this date used a different (possibly worse) model.
    cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
    if os.path.isfile(cal_path):
        epoch_ts = os.path.getmtime(cal_path)
        data["model_epoch"] = datetime.fromtimestamp(epoch_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    else:
        data["model_epoch"] = None
    return data


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


@app.get("/injuries/{year}")
def get_injuries(year: int):
    """Return current injury data for all teams."""
    path = os.path.join(DATA_DIR, f"injuries_{year}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


@app.post("/injuries/{year}/update")
def update_injury(
    year: int,
    team: str = Query(...),
    player: str = Query(...),
    status: str = Query(..., description="out, doubtful, questionable, probable, day-to-day, healthy (removes)"),
):
    """Add, update, or remove a single injury entry. BPR data auto-populated from EvanMiya.

    Examples:
      POST /injuries/2026/update?team=Duke&player=Patrick Ngongba&status=out
      POST /injuries/2026/update?team=UCLA&player=Donovan Dent&status=healthy
    """
    path = os.path.join(DATA_DIR, f"injuries_{year}.json")
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
    else:
        data = {}

    # Ensure team entry exists
    if team not in data:
        data[team] = {"injuries": [], "roster": []}
    elif isinstance(data[team], list):
        data[team] = {"injuries": data[team], "roster": []}

    injuries = data[team].get("injuries", [])

    # Remove existing entry for this player (case-insensitive)
    injuries = [i for i in injuries if i.get("player", "").lower() != player.lower()]

    if status.lower() not in ("healthy", "active", "playing", "remove"):
        # Add/update: look up BPR data
        sys.path.insert(0, ROOT) if ROOT not in sys.path else None
        from scripts.fetch_injuries_live import _load_bpr_data, _find_bpr_team, FALLBACK_BPR_SHARE
        from datetime import timezone as _tz
        bpr_data = _load_bpr_data(year)
        team_bpr = _find_bpr_team(team, bpr_data)

        # Match player in BPR data
        player_match = team_bpr.get(player.lower())
        if player_match is None:
            last = player.lower().split()[-1] if player else ""
            for k, v in team_bpr.items():
                if last and last in k:
                    player_match = v
                    break
        if player_match is None:
            parts = player.lower().split()
            if len(parts) >= 2:
                for k, v in team_bpr.items():
                    if parts[0] in k and parts[-1] in k:
                        player_match = v
                        break

        if player_match:
            bpr_val = round(float(player_match["bpr"]), 4)
            poss_val = round(float(player_match.get("poss", 0)), 0)
            share_val = round(float(player_match.get("bpr_share", 0)), 4)
        else:
            bpr_val = round(FALLBACK_BPR_SHARE * 10, 4)
            poss_val = 0.0
            share_val = FALLBACK_BPR_SHARE

        injuries.append({
            "player": player,
            "status": status.lower(),
            "bpr": bpr_val,
            "poss": poss_val,
            "bpr_share": share_val,
            "importance": share_val,
            "source": "manual",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        action = f"set {player} to {status}"
    else:
        action = f"removed {player}"

    data[team]["injuries"] = injuries
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    # Also rebuild teams_merged so the engine picks it up
    try:
        from scripts.fetch_data import main as rebuild_data
        rebuild_data(year)
    except Exception:
        pass  # non-critical

    return {"ok": True, "team": team, "action": action, "injuries": injuries}


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
    request: Request,
    year: int,
    sims: int = Query(default=10000, ge=100, le=100000),
):
    # Rate limit live simulations (pre-computed file is unaffected)
    if not (sims == 10000 and os.path.isfile(os.path.join(DATA_DIR, f"monte_carlo_{year}.json"))):
        ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(ip):
            raise HTTPException(status_code=429, detail="Too many Monte Carlo requests. Try again in a minute.")

    cache_key = f"mc_{year}_{sims}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Use pre-computed file when available (sims=10000) for fast load; fallback to live
    precomputed_path = os.path.join(DATA_DIR, f"monte_carlo_{year}.json")
    if sims == 10000 and os.path.isfile(precomputed_path):
        with open(precomputed_path) as f:
            result = json.load(f)
        result = _add_final_four_by_region(result, year)
        _cache_set(cache_key, result)
        return result

    bracket, _, _ = _load_bracket_for_year(year)
    config = _load_config(num_sims=sims)
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
    result = _add_final_four_by_region(result, year)
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

    # Fall back to live odds fetch if no saved file (e.g. before Actions runs)
    api_key = get_api_key()
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
