"""
Bracket Brain API — FastAPI wrapper around engine.py.

Endpoints:
  GET  /health                      — server status + data freshness
  GET  /teams/{year}                — all teams with stats for a given year
  POST /predict                     — predict one game
  GET  /bracket/{year}              — full bracket picks
  GET  /bracket/{year}/monte-carlo  — championship probabilities (N sims)

Usage:
  uvicorn api:app --reload
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import io
import os
import sys
import json
import contextlib
import re
from datetime import datetime, timezone
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import (
    predict_game, enrich_team, run_monte_carlo, load_bracket,
    generate_bracket_picks, load_teams_merged, _normalize_team_for_match,
    ModelConfig, DEFAULT_CONFIG,
)

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

WEB_DIR = os.path.join(ROOT, "web")


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
    bracket, ff_matchups, quadrant_order = _load_bracket_for_year(year)

    # Suppress engine's stdout prints during pick generation
    with contextlib.redirect_stdout(io.StringIO()):
        picks = generate_bracket_picks(
            bracket,
            upset_aggression=upset_aggression,
            quadrant_order=quadrant_order,
        )

    return {
        "year": year,
        "upset_aggression": upset_aggression,
        "picks": picks,
    }


@app.get("/bracket/{year}/monte-carlo")
def get_monte_carlo(
    year: int,
    sims: int = Query(default=10000, ge=100, le=100000),
):
    bracket, _, _ = _load_bracket_for_year(year)
    config = ModelConfig(num_sims=sims)

    with contextlib.redirect_stdout(io.StringIO()):
        mc = run_monte_carlo(bracket, config=config)

    return {
        "year": year,
        "num_simulations": sims,
        "champion_probs": mc["champion_probs"],
        "final_four_probs": mc["final_four_probs"],
        "elite_eight_probs": mc["elite_eight_probs"],
        "sweet_sixteen_probs": mc["sweet_sixteen_probs"],
        "round_of_32_probs": mc["round_of_32_probs"],
    }
