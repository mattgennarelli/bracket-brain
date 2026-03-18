"""
Evan Miya data loader.

No public API; manually download CSVs from evanmiya.com and place them in data/.

Supported files (either or both):
  data/evanmiya_teams_YYYY.csv  — Team Ratings download
  data/evanmiya_players_YYYY.csv — Player Ratings download

Legacy format also accepted:
  data/evanmiya_YYYY.csv — older generic export

Output: data/evanmiya_YYYY.json with per-team:
  team, star_score (0-1), top_player, top_player_bpr,
  em_obpr, em_dbpr, em_bpr,          — team offensive/defensive/net BPR
  em_opponent_adjust,                 — strength-of-schedule Bayesian adjustment
  em_pace_adjust,                     — performance delta at faster vs slower pace
  em_off_rank, em_def_rank,           — national offense/defense rank
  em_tempo, em_tempo_rank,
  em_home_rank,                       — home/neutral court performance rank
  em_runs_per_game, em_runs_conceded, em_runs_margin,  — scoring-burst metrics
  em_top5_bpr,                        — sum of top-5 player BPRs (depth)
  em_star_concentration,              — top1_bpr / top5_bpr (1.0 = one-man band)
  em_poss_weighted_bpr,               — possession-weighted avg BPR
  em_depth_score,                     — 0-1 roster depth score
  em_size_adj_bpr,                    — BPR weighted by position (rewards frontcourt quality)
  em_interior_bpr                     — sum BPR of players at position >= 4.0 (top 8)
"""
import csv
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _normalize(name):
    if not name or not isinstance(name, str):
        return ""
    s = " ".join(name.strip().split())
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    return s


def _num(val, default=0.0):
    if val is None or val == "":
        return default
    try:
        return float(str(val).replace("%", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _find_col(headers, candidates):
    """Find first matching column name (case-insensitive)."""
    lower_headers = {h.strip().lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower_headers:
            return lower_headers[c.lower()]
    return None


def _load_team_ratings(path):
    """Parse evanmiya_teams_YYYY.csv → dict keyed by normalized team name."""
    teams = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        team_col = _find_col(headers, ["Team", "team", "tooltip_team", "School", "Name"])
        # BPR columns — exact names first, then legacy aliases
        o_col   = _find_col(headers, ["obpr", "O-Rate", "O Rate", "ORate", "Off Rating"])
        d_col   = _find_col(headers, ["dbpr", "D-Rate", "D Rate", "DRate", "Def Rating"])
        bpr_col = _find_col(headers, ["bpr",  "Relative Rating", "RelRating", "Net Rating"])
        # Adjustment columns
        opp_adj_col  = _find_col(headers, ["opponent_adjust", "Opponent Adjust", "opp_adjust"])
        pace_adj_col = _find_col(headers, ["pace_adjust", "Pace Adjust", "pace_adj"])
        off_rank_col = _find_col(headers, ["off_rank", "Off Rank", "OffRank"])
        def_rank_col = _find_col(headers, ["def_rank", "Def Rank", "DefRank"])
        tempo_col    = _find_col(headers, ["tempo", "True Tempo", "Tempo", "Pace"])
        tempo_rank_col = _find_col(headers, ["tempo_rank", "Tempo Rank"])
        home_rank_col  = _find_col(headers, ["home_rank", "Home Rank"])
        # Runs / scoring-burst columns
        runs_pg_col  = _find_col(headers, ["runs_per_game", "Runs Per Game", "runs_pg"])
        runs_con_col = _find_col(headers, ["runs_conceded_per_game", "Runs Conceded Per Game"])
        runs_mar_col = _find_col(headers, ["runs_margin", "Runs Margin"])
        # Legacy conf rating
        conf_rating_col = _find_col(headers, ["Conf Strength", "Conf Rank", "conf_rating", "conf_rank",
                                               "Conference Strength", "Upper Strength", "Conf"])
        for row in reader:
            team = _normalize(row.get(team_col, "") if team_col else "")
            if not team:
                continue
            entry = {"team": team}
            if o_col:   entry["em_obpr"] = _num(row.get(o_col))
            if d_col:   entry["em_dbpr"] = _num(row.get(d_col))
            if bpr_col: entry["em_bpr"]  = _num(row.get(bpr_col))
            if opp_adj_col:   entry["em_opponent_adjust"] = _num(row.get(opp_adj_col))
            if pace_adj_col:  entry["em_pace_adjust"]     = _num(row.get(pace_adj_col))
            if off_rank_col:  entry["em_off_rank"]        = _num(row.get(off_rank_col))
            if def_rank_col:  entry["em_def_rank"]        = _num(row.get(def_rank_col))
            if tempo_col:     entry["em_tempo"]           = _num(row.get(tempo_col))
            if tempo_rank_col: entry["em_tempo_rank"]     = _num(row.get(tempo_rank_col))
            if home_rank_col:  entry["em_home_rank"]      = _num(row.get(home_rank_col))
            if runs_pg_col:   entry["em_runs_per_game"]   = _num(row.get(runs_pg_col))
            if runs_con_col:  entry["em_runs_conceded"]   = _num(row.get(runs_con_col))
            if runs_mar_col:  entry["em_runs_margin"]     = _num(row.get(runs_mar_col))
            if conf_rating_col:
                val = row.get(conf_rating_col, "").strip()
                if val:
                    try:
                        entry["conf_rating"] = float(val)
                    except ValueError:
                        v = str(val).lower()
                        if "upper" in v or "top" in v or "strong" in v:
                            entry["conf_rating"] = 1.0
                        elif "middle" in v or "mid" in v:
                            entry["conf_rating"] = 2.0
                        elif "lower" in v or "weak" in v:
                            entry["conf_rating"] = 3.0
                        else:
                            entry["conf_rating"] = 2.0
            teams[team.lower()] = entry
    return teams


def _load_player_ratings(path):
    """Parse evanmiya_players_YYYY.csv → dict keyed by normalized team name.

    Computes per-team depth features:
      top_player / top_player_bpr — best player and their BPR
      em_top5_bpr        — sum of top-5 BPRs (total roster quality)
      em_star_concentration — top1 / top5_bpr (high = one-man-band, more upset-prone)
      em_poss_weighted_bpr  — possession-weighted avg BPR (who actually plays)
      em_depth_score     — 0-1 composite: high = deep, balanced roster
      em_size_adj_bpr    — BPR weighted by position (frontcourt quality signal)
      em_interior_bpr    — sum BPR of pos >= 4.0 players in top 8
    """
    # Collect all players per team
    team_players = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        player_col = _find_col(headers, ["Player", "player", "Name", "name"])
        team_col   = _find_col(headers, ["Team", "team", "School"])
        bpr_col    = _find_col(headers, ["BPR", "bpr", "Bayesian Performance Rating"])
        obpr_col   = _find_col(headers, ["OBPR", "obpr"])
        dbpr_col   = _find_col(headers, ["DBPR", "dbpr"])
        poss_col   = _find_col(headers, ["poss", "Off Poss", "Poss", "Minutes", "Min", "MP"])
        pos_col    = _find_col(headers, ["position", "Position", "Pos"])
        # Team-level efficiency estimates on same scale as Torvik adj_o/adj_d
        team_off_col = _find_col(headers, ["adj_team_off_eff", "Adj Team Off Eff"])
        team_def_col = _find_col(headers, ["adj_team_def_eff", "Adj Team Def Eff"])
        for row in reader:
            team   = _normalize(row.get(team_col, "") if team_col else "")
            player = row.get(player_col, "").strip() if player_col else ""
            if not team or not player:
                continue
            bpr  = _num(row.get(bpr_col))  if bpr_col  else 0.0
            poss = _num(row.get(poss_col), 1) if poss_col else 1.0
            pos  = _num(row.get(pos_col), 3.0) if pos_col else 3.0
            key  = team.lower()
            if key not in team_players:
                team_players[key] = {"team": team, "players": []}
            entry = {
                "name": player,
                "bpr":  bpr,
                "obpr": _num(row.get(obpr_col)) if obpr_col else 0.0,
                "dbpr": _num(row.get(dbpr_col)) if dbpr_col else 0.0,
                "poss": max(poss, 1),
                "pos":  max(1.0, min(5.0, pos)),
            }
            # Team efficiency context (same value for all players on a team; store once)
            if team_off_col and "em_adj_o" not in team_players[key]:
                val = _num(row.get(team_off_col), 0.0)
                if val > 50:  # sanity check — must be a real efficiency number
                    team_players[key]["em_adj_o"] = val
            if team_def_col and "em_adj_d" not in team_players[key]:
                val = _num(row.get(team_def_col), 0.0)
                if val > 50:
                    team_players[key]["em_adj_d"] = val
            team_players[key]["players"].append(entry)

    result = {}
    for key, data in team_players.items():
        players = sorted(data["players"], key=lambda p: -p["bpr"])
        top1 = players[0] if players else None
        if not top1:
            continue

        top5 = players[:5]
        top5_bpr_sum = sum(p["bpr"] for p in top5)

        # Star concentration: fraction of top-5 BPR held by top player
        star_conc = (top1["bpr"] / top5_bpr_sum) if top5_bpr_sum > 0 else 1.0
        star_conc = max(0.0, min(1.0, star_conc))

        # Possession-weighted average BPR across all players
        total_poss = sum(p["poss"] for p in players)
        poss_weighted = sum(p["bpr"] * p["poss"] for p in players) / total_poss if total_poss > 0 else 0.0

        # Depth score: supporting cast quality (players 2-5, excludes star).
        # Star player is already captured by star_score; depth measures the core
        # supporting cast. Data analysis shows players 2-5 carry real independent
        # signal (65% close-game predictor, 75% override when disagreeing with
        # efficiency), while players 6-10 add only noise.
        # Typical supporting_bpr range: ~5 (weak) to ~25 (elite). Normalize to 0-1.
        supporting = players[1:5]
        supporting_bpr_sum = sum(p["bpr"] for p in supporting)
        depth_quality = max(0.0, min(1.0, (supporting_bpr_sum - 5) / 20))
        depth_balance = 1.0 - star_conc
        depth_score   = round(0.6 * depth_quality + 0.4 * depth_balance, 3)

        # Interior strength metrics (top-8 by BPR).
        # Tournament basketball rewards frontcourt quality — rebounding, paint
        # scoring, and rim protection are harder to scheme against with limited
        # prep time.  Controlled analysis (2010-2025, N=225 equal-talent games)
        # shows ~58% win rate for teams with higher size-adjusted BPR.
        top8 = players[:8]
        has_pos = any(p.get("pos", 3.0) != 3.0 for p in top8)
        if has_pos:
            size_adj_bpr = round(sum(p["bpr"] * (p["pos"] / 3.0) for p in top8), 3)
            interior_bpr = round(sum(p["bpr"] for p in top8 if p["pos"] >= 4.0), 3)
        else:
            size_adj_bpr = None
            interior_bpr = None

        result[key] = {
            "team":           data["team"],
            "top_player":     top1["name"],
            "top_player_bpr": round(top1["bpr"], 4),
            "em_top5_bpr":           round(top5_bpr_sum, 3),
            "em_star_concentration": round(star_conc, 3),
            "em_poss_weighted_bpr":  round(poss_weighted, 3),
            "em_depth_score":        depth_score,
        }
        if size_adj_bpr is not None:
            result[key]["em_size_adj_bpr"] = size_adj_bpr
        if interior_bpr is not None:
            result[key]["em_interior_bpr"] = interior_bpr
        # Pass through team efficiency estimates if captured
        if "em_adj_o" in data:
            result[key]["em_adj_o"] = round(data["em_adj_o"], 4)
        if "em_adj_d" in data:
            result[key]["em_adj_d"] = round(data["em_adj_d"], 4)
    return result


def _bpr_to_star_score(bpr):
    """Convert BPR to 0-1 star_score. Typical range: -3 to +10."""
    return max(0.0, min(1.0, (bpr + 3) / 13))


def load_evanmiya_season(year=2026):
    """Load Evan Miya data for a given year and write evanmiya_YYYY.json."""
    teams_csv = os.path.join(DATA_DIR, f"evanmiya_teams_{year}.csv")
    players_csv = os.path.join(DATA_DIR, f"evanmiya_players_{year}.csv")
    legacy_csv = os.path.join(DATA_DIR, f"evanmiya_{year}.csv")
    out_json = os.path.join(DATA_DIR, f"evanmiya_{year}.json")

    has_teams = os.path.isfile(teams_csv)
    has_players = os.path.isfile(players_csv)
    has_legacy = os.path.isfile(legacy_csv)

    if not has_teams and not has_players and not has_legacy:
        print("Evan Miya data is optional. To use:")
        print(f"  1. Go to evanmiya.com → Team Ratings → Download")
        print(f"     Save as: {teams_csv}")
        print(f"  2. Go to evanmiya.com → Player Ratings → Download")
        print(f"     Save as: {players_csv}")
        print("  Either or both files work.")
        return False

    result = {}

    if has_teams:
        team_data = _load_team_ratings(teams_csv)
        print(f"  Team ratings: {len(team_data)} teams from {teams_csv}")
        for key, entry in team_data.items():
            result[key] = entry

    if has_players:
        player_data = _load_player_ratings(players_csv)
        print(f"  Player ratings: {len(player_data)} teams from {players_csv}")
        for key, pdata in player_data.items():
            if key not in result:
                result[key] = {"team": pdata["team"]}
            result[key]["top_player"]     = pdata["top_player"]
            result[key]["top_player_bpr"] = pdata["top_player_bpr"]
            result[key]["star_score"]     = round(_bpr_to_star_score(pdata["top_player_bpr"]), 3)
            # Depth features, interior strength, EvanMiya efficiency estimates
            for depth_key in ("em_top5_bpr", "em_star_concentration",
                              "em_poss_weighted_bpr", "em_depth_score",
                              "em_size_adj_bpr", "em_interior_bpr",
                              "em_adj_o", "em_adj_d"):
                if depth_key in pdata:
                    result[key][depth_key] = pdata[depth_key]

    if has_legacy and not has_teams and not has_players:
        with open(legacy_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row in reader:
                team = _normalize(
                    row.get("team") or row.get("Team") or row.get("name") or ""
                )
                if not team:
                    continue
                bpr = _num(row.get("BPR") or row.get("star_bpr") or row.get("bpr_team") or
                           row.get("Top Player BPR") or row.get("OBPR") or row.get("DBPR"), 0)
                key = team.lower()
                result[key] = {
                    "team": team,
                    "star_score": round(_bpr_to_star_score(bpr), 3),
                    "top_player_bpr": bpr,
                }
        print(f"  Legacy CSV: {len(result)} teams from {legacy_csv}")

    out_list = list(result.values())
    for entry in out_list:
        if "star_score" not in entry:
            entry["star_score"] = 0.5
        entry["team"] = entry.get("team", "")

    with open(out_json, "w") as f:
        json.dump(out_list, f, indent=2)
    print(f"  Wrote {len(out_list)} teams to {out_json}")
    return True


def main():
    year = 2026
    if len(sys.argv) > 1:
        try:
            year = int(sys.argv[1])
        except ValueError:
            pass
    print(f"Loading Evan Miya data for {year}...")
    ok = load_evanmiya_season(year)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
