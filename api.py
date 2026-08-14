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
from functools import lru_cache
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import (
    predict_game, enrich_team, run_monte_carlo, load_bracket,
    generate_bracket_picks, load_teams_merged, _normalize_team_for_match,
    get_matchup_analysis_display, is_ncaa_tournament_game, _strip_mascot,
    ModelConfig, DEFAULT_CONFIG, resolve_ff_pairs, build_locked_picks_from_results,
)
from espn_scores import fetch_espn_scoreboard
from scripts.fetch_data import build_merged_teams

# Cache TTL: 1 hour — avoids stale Render cache on redeploy
CACHE_TTL_SECONDS = 3600
# Scores cache: 90 seconds for live score freshness
SCORES_CACHE_TTL = 90
BRACKET_RESPONSE_VERSION = 2
ET_TZ = ZoneInfo("America/New_York")
ROUND_NAME_BY_SIZE = {
    68: "First Four",
    64: "Round of 64",
    32: "Round of 32",
    16: "Sweet 16",
    8: "Elite 8",
    4: "Final Four",
    2: "Championship",
}
REGION_R64_SEED_PAIRS = (
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
)

# Cache: key -> {"result": ..., "cached_at": timestamp}
# Uses OrderedDict so we can evict the oldest entry when size limit is hit.
_CACHE_MAX_ENTRIES = 50
_cache: collections.OrderedDict = collections.OrderedDict()

# In-process cache for precomputed per-year artifacts (bracket picks, Monte Carlo).
# Keyed simply by year -- populated once per process lifetime from a static file
# read, so repeat requests within the same deploy cost a dict lookup, not even a
# file read. Cold-start-safe: a free-tier instance restarting just means the next
# request repopulates it, same file read as always.
_static_artifact_cache: dict = {}

# Precomputed, rarely-changing artifacts are safe for browsers/Cloudflare to cache
# and re-serve without ever reaching this (CPU-constrained, free-tier) origin.
_STATIC_CACHE_CONTROL = "public, max-age=3600"


def _static_json_response(data: dict) -> JSONResponse:
    return JSONResponse(content=data, headers={"Cache-Control": _STATIC_CACHE_CONTROL})

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


def _resolve_game_site(game_site, team_a: dict, team_b: dict):
    """Convert API game_site values into the lat/lon tuple expected by engine.py."""
    if game_site in (None, "", "neutral"):
        return None
    if isinstance(game_site, (list, tuple)) and len(game_site) == 2:
        return tuple(game_site)
    if not isinstance(game_site, str):
        raise HTTPException(status_code=422, detail="game_site must be null, 'home_a', 'home_b', or 'lat,lon'")

    site = game_site.strip().lower()
    if site == "home_a":
        loc = team_a.get("location")
        return tuple(loc) if isinstance(loc, (list, tuple)) and len(loc) == 2 else None
    if site == "home_b":
        loc = team_b.get("location")
        return tuple(loc) if isinstance(loc, (list, tuple)) and len(loc) == 2 else None

    parts = [p.strip() for p in game_site.split(",")]
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass

    raise HTTPException(status_code=422, detail="game_site must be null, 'home_a', 'home_b', or 'lat,lon'")


def _config_mtime() -> str:
    """Return mtime of calibrated_config.json for cache busting."""
    cal_path = os.path.join(DATA_DIR, "calibrated_config.json")
    if os.path.isfile(cal_path):
        return str(int(os.path.getmtime(cal_path)))
    return "0"


def _prediction_inputs_mtime(year: int) -> str:
    """Return the newest mtime among files that affect predictions for a given year."""
    paths = [
        os.path.join(DATA_DIR, f"bracket_{year}.json"),
        os.path.join(DATA_DIR, f"teams_merged_{year}.json"),
        os.path.join(DATA_DIR, f"injuries_{year}.json"),
        os.path.join(DATA_DIR, f"results_{year}.json"),
        os.path.join(DATA_DIR, "calibrated_config.json"),
    ]
    latest = 0
    for path in paths:
        if os.path.isfile(path):
            latest = max(latest, int(os.path.getmtime(path)))
    return str(latest)


def _prediction_inputs_hash(year: int) -> str:
    """Stable hash of files that affect bracket and Monte Carlo predictions."""
    paths = [
        os.path.join(DATA_DIR, f"bracket_{year}.json"),
        os.path.join(DATA_DIR, f"teams_merged_{year}.json"),
        os.path.join(DATA_DIR, f"injuries_{year}.json"),
        os.path.join(DATA_DIR, f"results_{year}.json"),
        os.path.join(DATA_DIR, "calibrated_config.json"),
    ]
    h = hashlib.sha256()
    for path in paths:
        h.update(path.encode("utf-8"))
        if not os.path.isfile(path):
            h.update(b"<missing>")
            continue
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


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


def _load_results_games(year: int) -> list[dict]:
    path = os.path.join(DATA_DIR, f"results_{year}.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, dict):
        games = data.get("games", [])
    elif isinstance(data, list):
        games = data
    else:
        games = []
    return [game for game in games if isinstance(game, dict)]


def _completed_tournament_locked_picks(year: int, bracket: dict, quadrant_order=None, ff_matchups=None) -> Dict[str, str]:
    return build_locked_picks_from_results(
        bracket,
        _load_results_games(year),
        quadrant_order=quadrant_order,
        ff_matchups=ff_matchups,
    )


@lru_cache(maxsize=8)
def _load_bracket_file(year: int) -> dict:
    path = os.path.join(DATA_DIR, f"bracket_{year}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"No bracket found for {year}")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=8)
def _tournament_team_map(year: int) -> Dict[str, str]:
    """Return normalized team name -> canonical bracket name for the tournament field."""
    raw = _load_bracket_file(year)
    out: Dict[str, str] = {}
    for entries in raw.get("regions", {}).values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            team = entry.get("team")
            if team:
                out[_normalize_team_for_match(team)] = team
                out[_normalize_team_for_match(_strip_mascot(team))] = team
    for ff in raw.get("first_four", []):
        if not isinstance(ff, dict):
            continue
        for key in ("team_a", "team_b"):
            team = ff.get(key)
            if team:
                out[_normalize_team_for_match(team)] = team
                out[_normalize_team_for_match(_strip_mascot(team))] = team
    return out


def _resolve_tournament_team(team_map: Dict[str, str], *names: str) -> Optional[str]:
    candidates = []
    for raw_name in names:
        if not raw_name:
            continue
        for candidate in (raw_name, _strip_mascot(raw_name)):
            norm = _normalize_team_for_match(candidate)
            if norm in team_map:
                return team_map[norm]
            if norm:
                candidates.append(norm)
    for candidate in candidates:
        for key, team in team_map.items():
            shorter, longer = (candidate, key) if len(candidate) <= len(key) else (key, candidate)
            if len(shorter) >= 6 and shorter in longer and len(shorter) >= len(longer) * 0.80:
                return team
    return None


def _build_bracket_scores_from_results(results_games: list[dict]) -> dict:
    """Build the scores response straight from results_{year}.json games -- zero
    network calls, zero team-name resolution. Used whenever a year's tournament is
    fully recorded (all 63 games), which is true for every year in this deployment.
    ESPN's live scoreboard is only needed for a genuinely in-progress season."""
    result = {"scores": {}}
    for game in results_games:
        team_a, team_b = game.get("team_a"), game.get("team_b")
        if not team_a or not team_b:
            continue
        try:
            round_of = int(game.get("round"))
        except (TypeError, ValueError):
            continue
        rec = {
            "team_a": team_a, "team_b": team_b,
            "score_a": game.get("score_a"), "score_b": game.get("score_b"),
            "scheduled_at": None, "completed": True,
            "status_detail": "Final", "display_clock": "", "period": 0,
            "round_of": round_of,
        }
        result["scores"][f"{team_a}|{team_b}"] = rec
        result["scores"][f"{team_b}|{team_a}"] = {
            **rec, "team_a": team_b, "team_b": team_a,
            "score_a": game.get("score_b"), "score_b": game.get("score_a"),
        }
    return result


def _build_bracket_scores_result(year: int, days: int = 21) -> dict:
    # A fully recorded year never needs a live ESPN fetch -- this used to be the
    # dominant cost on this route: up to days+1 sequential ESPN HTTP calls, redone
    # every 90 seconds, for a tournament that ended long ago.
    results_games = _load_results_games(year)
    if len(results_games) >= 63:
        return _build_bracket_scores_from_results(results_games)

    team_map = _tournament_team_map(year)
    if not team_map:
        return {"scores": {}}

    today = datetime.now(timezone.utc)
    dates = [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(days + 1)]
    games = fetch_espn_scoreboard(dates)
    result = {"scores": {}}
    for game in games:
        home = _resolve_tournament_team(team_map, game.get("home_team", ""), *game.get("home_aliases", []))
        away = _resolve_tournament_team(team_map, game.get("away_team", ""), *game.get("away_aliases", []))
        if not home or not away or home == away:
            continue
        round_of = _exact_tournament_round_for_matchup(home, away, game.get("scheduled_at"), year=year)
        if round_of is None:
            continue
        rec = {
            "team_a": home,
            "team_b": away,
            "score_a": game.get("home_score"),
            "score_b": game.get("away_score"),
            "scheduled_at": game.get("scheduled_at"),
            "completed": game.get("completed", False),
            "status_detail": game.get("status_detail", ""),
            "display_clock": game.get("display_clock", ""),
            "period": game.get("period", 0),
            "round_of": round_of,
        }
        home_key = f"{home}|{away}"
        away_key = f"{away}|{home}"
        existing = result["scores"].get(home_key)
        if existing and str(existing.get("scheduled_at") or "") > str(rec.get("scheduled_at") or ""):
            continue
        result["scores"][home_key] = rec
        result["scores"][away_key] = {
            **rec,
            "team_a": away,
            "team_b": home,
            "score_a": game.get("away_score"),
            "score_b": game.get("home_score"),
        }
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
        os.path.join(DATA_DIR, f"results_{year}.json"),
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
    keys = []
    for candidate in (name, _strip_mascot(name)):
        key = _normalize_team_for_match(candidate)
        if key and key not in keys:
            keys.append(key)

    stats = None
    for key in keys:
        stats = teams.get(key)
        if stats is not None:
            break

    if stats is None:
        for key in keys:
            for team_key, team_stats in teams.items():
                if len(key) <= len(team_key) and key in team_key and len(key) >= len(team_key) * 0.65:
                    stats = team_stats
                    break
            if stats is not None:
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
    # /health is the first thing every page load blocks on. _data_hash_and_meta
    # hashes several data files (one over 1MB) -- cache it in-process so repeat
    # hits (the overwhelming majority) cost a dict lookup, not a re-hash.
    years = _available_years()
    current = max(years) if years else None
    if current is None:
        meta = {}
    else:
        meta = _static_artifact_cache.get(f"health_meta_{current}")
        if meta is None:
            meta = _data_hash_and_meta(current)
            _static_artifact_cache[f"health_meta_{current}"] = meta
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
    bracket, ff_matchups, quadrant_order = _load_bracket_for_year(year)
    config = _load_config()
    locked_picks = _completed_tournament_locked_picks(year, bracket, quadrant_order, ff_matchups)
    with contextlib.redirect_stdout(io.StringIO()):
        result = generate_bracket_picks(
            bracket,
            config=config,
            upset_aggression=upset_aggression,
            quadrant_order=quadrant_order,
            ff_matchups=ff_matchups,
            data_dir=DATA_DIR,
            year=year,
            locked_picks=locked_picks,
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

    result = predict_game(team_a, team_b, game_site=_resolve_game_site(req.game_site, team_a, team_b))

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
    # Default (chalk, upset_aggression=0.0) is the page-load path: serve the
    # precomputed artifact directly, no per-request generate_bracket_picks() call
    # (63 predict_game evaluations) at all. Any other value means the user moved
    # the chaos slider -- a deliberate live action, stays live-computed below.
    if upset_aggression == 0.0:
        cached = _static_artifact_cache.get(f"bracket_{year}")
        if cached is not None:
            return _static_json_response(cached)
        precomputed_path = os.path.join(DATA_DIR, f"bracket_picks_{year}.json")
        if os.path.isfile(precomputed_path):
            with open(precomputed_path) as f:
                precomputed = json.load(f)
            _static_artifact_cache[f"bracket_{year}"] = precomputed
            return _static_json_response(precomputed)

    cache_key = f"bracket_v{BRACKET_RESPONSE_VERSION}_{year}_{upset_aggression:.2f}_{_prediction_inputs_mtime(year)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    bracket, ff_matchups, quadrant_order = _load_bracket_for_year(year)
    config = _load_config()
    locked_picks = _completed_tournament_locked_picks(year, bracket, quadrant_order, ff_matchups)
    with contextlib.redirect_stdout(io.StringIO()):
        picks = generate_bracket_picks(
            bracket,
            config=config,
            upset_aggression=upset_aggression,
            quadrant_order=quadrant_order,
            ff_matchups=ff_matchups,
            data_dir=DATA_DIR,
            year=year,
            locked_picks=locked_picks,
        )
    ff_pairs = [list(pair) for pair in resolve_ff_pairs(quadrant_order, ff_matchups)]
    result = {
        "year": year,
        "upset_aggression": upset_aggression,
        "picks": picks,
        "quadrant_order": quadrant_order,
        "ff_pairs": ff_pairs,
    }
    _cache_set(cache_key, result)
    return result


@app.post("/bracket/{year}/simulate")
def simulate_bracket(year: int, req: SimulateRequest):
    """Generate bracket picks respecting locked picks. Use when user has manually
    locked some picks and wants to re-simulate the rest."""
    bracket, ff_matchups, quadrant_order = _load_bracket_for_year(year)
    config = _load_config()
    completed_locked = _completed_tournament_locked_picks(year, bracket, quadrant_order, ff_matchups)
    locked_picks = {**req.locked_picks, **completed_locked}
    with contextlib.redirect_stdout(io.StringIO()):
        bracket_result = generate_bracket_picks(
            bracket,
            config=config,
            upset_aggression=req.upset_aggression,
            quadrant_order=quadrant_order,
            ff_matchups=ff_matchups,
            data_dir=DATA_DIR,
            year=year,
            locked_picks=locked_picks,
        )
    ff_pairs = [list(pair) for pair in resolve_ff_pairs(quadrant_order, ff_matchups)]
    return {
        "year": year,
        "upset_aggression": req.upset_aggression,
        "picks": bracket_result,
        "quadrant_order": quadrant_order,
        "ff_pairs": ff_pairs,
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
    cache_key = None
    if not injury_overrides:
        cache_key = (
            f"analyze_{year}_{_prediction_inputs_mtime(year)}_"
            f"{_normalize_team_for_match(_strip_mascot(team_a))}_"
            f"{_normalize_team_for_match(_strip_mascot(team_b))}_"
            f"{seed_a}_{seed_b}_{region or ''}_{round_name or ''}"
        )
        cached = _cache_get(cache_key, ttl=300)
        if cached is not None:
            return cached

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
    if cache_key:
        _cache_set(cache_key, analysis)
    return analysis


def _nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int):
    d = datetime(year, month, 1, tzinfo=ET_TZ)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    d += timedelta(days=7 * (occurrence - 1))
    return d.date()


def _tournament_round_windows(year: int):
    r64_start = _nth_weekday_of_month(year, 3, 3, 3)  # third Thursday of March
    first_four_start = r64_start - timedelta(days=2)
    return {
        68: (first_four_start, first_four_start + timedelta(days=1)),
        64: (r64_start, r64_start + timedelta(days=1)),
        32: (r64_start + timedelta(days=2), r64_start + timedelta(days=3)),
        16: (r64_start + timedelta(days=7), r64_start + timedelta(days=8)),
        8: (r64_start + timedelta(days=9), r64_start + timedelta(days=10)),
        4: (_nth_weekday_of_month(year, 4, 5, 1), _nth_weekday_of_month(year, 4, 5, 1)),
        2: (_nth_weekday_of_month(year, 4, 0, 1), _nth_weekday_of_month(year, 4, 0, 1)),
    }


def _infer_tournament_round(year: int, scheduled_at: str):
    if not scheduled_at:
        return None
    try:
        dt = datetime.fromisoformat(str(scheduled_at).replace("Z", "+00:00")).astimezone(ET_TZ).date()
    except ValueError:
        return None
    for round_of, (start, end) in _tournament_round_windows(year).items():
        if start <= dt <= end:
            return round_of
    return None


def _matchup_key(team_a: str, team_b: str):
    a = _normalize_team_for_match(_strip_mascot(team_a))
    b = _normalize_team_for_match(_strip_mascot(team_b))
    if not a or not b or a == b:
        return None
    return tuple(sorted((a, b)))


def _add_matchups(matchups: set, teams_a, teams_b):
    for team_a in teams_a or []:
        for team_b in teams_b or []:
            key = _matchup_key(team_a, team_b)
            if key:
                matchups.add(key)


@lru_cache(maxsize=8)
def _exact_tournament_matchups(year: int):
    bracket_path = os.path.join(DATA_DIR, f"bracket_{year}.json")
    if not os.path.isfile(bracket_path):
        return {}

    raw_bracket = _load_bracket_file(year)
    regions = {}
    for region, teams in (raw_bracket.get("regions") or {}).items():
        seed_map = {}
        if isinstance(teams, list):
            for entry in teams:
                try:
                    seed = int(entry.get("seed"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if entry.get("team"):
                    seed_map[seed] = entry.get("team")
        elif isinstance(teams, dict):
            for seed, entry in teams.items():
                try:
                    seed = int(seed)
                except (TypeError, ValueError):
                    continue
                if isinstance(entry, dict) and entry.get("team"):
                    seed_map[seed] = entry.get("team")
        regions[region] = seed_map
    first_four_slots = {}
    matchups = collections.defaultdict(set)

    for ff in raw_bracket.get("first_four", []):
        team_a = ff.get("team_a")
        team_b = ff.get("team_b")
        key = _matchup_key(team_a, team_b)
        if key:
            matchups[68].add(key)
        region = ff.get("region")
        seed = ff.get("seed")
        if region and seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                continue
            slot_teams = [t for t in (team_a, team_b) if t]
            if slot_teams:
                first_four_slots[(region, seed)] = slot_teams

    region_winner_sets = {}
    for region, teams_by_seed in regions.items():
        def slot_teams(seed: int):
            out = set(first_four_slots.get((region, seed), []))
            entry = teams_by_seed.get(seed)
            if isinstance(entry, str):
                out.add(entry)
            return out

        round_slots = []
        for seed_a, seed_b in REGION_R64_SEED_PAIRS:
            teams_a = slot_teams(seed_a)
            teams_b = slot_teams(seed_b)
            _add_matchups(matchups[64], teams_a, teams_b)
            round_slots.append(set(teams_a) | set(teams_b))

        next_slots = []
        for idx in range(0, len(round_slots), 2):
            left = round_slots[idx]
            right = round_slots[idx + 1]
            _add_matchups(matchups[32], left, right)
            next_slots.append(set(left) | set(right))
        round_slots = next_slots

        next_slots = []
        for idx in range(0, len(round_slots), 2):
            left = round_slots[idx]
            right = round_slots[idx + 1]
            _add_matchups(matchups[16], left, right)
            next_slots.append(set(left) | set(right))
        round_slots = next_slots

        if len(round_slots) == 2:
            left, right = round_slots
            _add_matchups(matchups[8], left, right)
            region_winner_sets[region] = set(left) | set(right)

    quadrant_order = raw_bracket.get("quadrant_order") or list(regions.keys())
    ff_regions = resolve_ff_pairs(quadrant_order, raw_bracket.get("final_four_matchups"))
    semifinal_winner_sets = []
    for region_a, region_b in ff_regions:
        winners_a = region_winner_sets.get(region_a, set())
        winners_b = region_winner_sets.get(region_b, set())
        _add_matchups(matchups[4], winners_a, winners_b)
        semifinal_winner_sets.append(set(winners_a) | set(winners_b))

    if len(semifinal_winner_sets) == 2:
        _add_matchups(matchups[2], semifinal_winner_sets[0], semifinal_winner_sets[1])

    return matchups


def _exact_tournament_round_for_matchup(team_a: str, team_b: str, scheduled_at: str, year: int = 2026):
    round_of = _infer_tournament_round(year, scheduled_at)
    if round_of is None:
        return None
    matchup_key = _matchup_key(team_a, team_b)
    if matchup_key is None:
        return None
    matchups = _exact_tournament_matchups(year)
    if not matchups:
        return None
    return round_of if matchup_key in matchups.get(round_of, set()) else None


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
        build_merged_teams(year, skip_torvik_fetch=True)
        _cache.clear()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Injury updated, but teams rebuild failed: {exc}")

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
    # Default sims: always serve the precomputed artifact directly, no per-request
    # computation at all -- this is the page-load path and must be fast even on a
    # throttled free-tier instance. final_four_by_region is embedded in the file at
    # generation time (see run.py), so this is a pure static read. Freshness is a
    # regenerate-and-redeploy concern (`python run.py --all`), not a per-request
    # check: hashing 5 files (one of them 1MB+) on every hit was itself a real cost.
    # Live simulation is reserved for an explicit non-default sims value.
    if sims == 10000:
        cached = _static_artifact_cache.get(f"mc_{year}")
        if cached is not None:
            return _static_json_response(cached)
        precomputed_path = os.path.join(DATA_DIR, f"monte_carlo_{year}.json")
        if os.path.isfile(precomputed_path):
            with open(precomputed_path) as f:
                precomputed = json.load(f)
            if "final_four_by_region" not in precomputed:
                precomputed = _add_final_four_by_region(precomputed, year)
            _static_artifact_cache[f"mc_{year}"] = precomputed
            return _static_json_response(precomputed)

    ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Too many Monte Carlo requests. Try again in a minute.")

    inputs_mtime = _prediction_inputs_mtime(year)
    cache_key = f"mc_live_{year}_{sims}_{inputs_mtime}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    bracket, ff_matchups, quadrant_order = _load_bracket_for_year(year)
    config = _load_config(num_sims=sims)
    with contextlib.redirect_stdout(io.StringIO()):
        mc = run_monte_carlo(
            bracket,
            config=config,
            year=year,
            quadrant_order=quadrant_order,
            ff_matchups=ff_matchups,
        )

    result = {
        "year": year,
        "num_simulations": sims,
        "champion_probs": mc["champion_probs"],
        "final_four_probs": mc["final_four_probs"],
        "elite_eight_probs": mc["elite_eight_probs"],
        "sweet_sixteen_probs": mc["sweet_sixteen_probs"],
        "round_of_32_probs": mc["round_of_32_probs"],
        "prediction_inputs_hash": _prediction_inputs_hash(year),
    }
    result = _add_final_four_by_region(result, year)
    _cache_set(cache_key, result)
    return result


@app.get("/bracket/{year}/scores")
def get_bracket_scores(year: int, days: int = Query(default=21, ge=0, le=30)):
    """Live/final tournament scores keyed by bracket team names in both team orders."""
    # Fully recorded year -> static forever (results_{year}.json doesn't change
    # without a redeploy). Cached in-process and browser/Cloudflare-cacheable.
    cached = _static_artifact_cache.get(f"scores_{year}")
    if cached is not None:
        return _static_json_response(cached)
    results_games = _load_results_games(year)
    if len(results_games) >= 63:
        cached = _build_bracket_scores_from_results(results_games)
        _static_artifact_cache[f"scores_{year}"] = cached
        return _static_json_response(cached)

    cache_key = f"bracket_scores_{year}_{days}"
    entry = _cache.get(cache_key)
    if entry and (time.time() - entry.get("cached_at", 0)) < SCORES_CACHE_TTL:
        return entry.get("result", {})

    result = _build_bracket_scores_result(year, days=days)
    _cache[cache_key] = {"result": result, "cached_at": time.time()}
    return result

