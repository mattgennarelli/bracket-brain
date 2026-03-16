"""
Fetch team stats from Bart Torvik T-Rank for the given season.
Saves to data/torvik_YYYY.csv and data/torvik_YYYY.json with canonical schema:
  team, adj_o, adj_d, adj_tempo, barthag
"""
import csv
import json
import os
import re
import sys

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

# Project root = parent of scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Column mapping: various names Torvik might use -> our canonical name
# Torvik CSV uses: Team, AdjOE, AdjDE, Adj T. (or AdjT), Barthag, TOR, ORB, FTR
COLUMN_ALIASES = {
    "team": ["team", "Team", "teamname", "name", "TEAM", "School"],
    "adj_o": ["adj_o", "AdjO", "AdjOE", "adjoe", "adj_oe", "off eff", "ORtg"],
    "adj_d": ["adj_d", "AdjD", "AdjDE", "adjde", "adj_de", "def eff", "DRtg"],
    "adj_tempo": ["adj_tempo", "AdjT", "adj_t", "tempo", "Tempo", "adjt", "Adj T.", "Adj T"],
    "barthag": ["barthag", "Barthag", "barthag%", "win%"],
    # Four factors / possession drivers
    "to_rate": ["tov%", "to%", "tor", "tov", "TO%", "TOV%", "torate", "TOR"],
    "orb_rate": ["orb%", "or%", "orb", "OR%", "ORb"],
    "ft_rate": ["ftr", "ft_rate", "FT Rate", "FT%", "ft%", "FTR"],
    # 3PT shooting
    "three_rate": ["3PA%", "3par", "3P Rate", "three_rate", "3FGRate"],
    "three_pct": ["3P%", "3FG%", "three_pct", "3pt%"],
    # Schedule strength / resume
    "sos": ["sos", "SOS", "adj_sos", "AdjSOS"],
    # Quality metrics (from raw Torvik CSVs)
    "wab": ["WAB", "wab"],
    "elite_sos": ["elite SOS", "elite_sos"],
    "qual_o": ["Qual O", "qual_o"],
    "qual_d": ["Qual D", "qual_d"],
    "qual_barthag": ["Qual Barthag", "qual_barthag"],
    "conf_adj_o": ["Con Adj OE", "conf_adj_o", "ConOE"],
    "conf_adj_d": ["Con Adj DE", "conf_adj_d", "ConDE"],
    "record": ["record", "Record", "rec"],
    "conf_win_pct": ["Conf Win%", "conf_win_pct", "Conf Win", "conf win%"],
}


def normalize_team_name(name):
    """Normalize team name for matching (strip, collapse spaces)."""
    if not name or not isinstance(name, str):
        return ""
    return " ".join(name.strip().split())


def find_column(row_or_headers, canonical):
    """Find first matching column name for canonical key (case-insensitive)."""
    aliases = COLUMN_ALIASES.get(canonical, [canonical])
    if isinstance(row_or_headers, dict):
        keys = list(row_or_headers.keys())
    else:
        keys = list(row_or_headers) if hasattr(row_or_headers, "__iter__") else []
    for k in keys:
        # Keys may be ints or other types; always coerce to string
        k_clean = str(k or "").strip().lower()
        for a in aliases:
            if a.lower() == k_clean:
                return k
    return None


def parse_value(val, default_float=None):
    """Parse a cell value to float when possible."""
    if val is None or val == "":
        return default_float
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace("%", "")
    try:
        return float(s)
    except ValueError:
        return default_float


def _parse_rows_to_canonical(data_rows):
    """Parse raw rows to canonical format. Returns list of dicts, or empty if parse fails."""
    if not data_rows:
        return []
    canonical_rows = []
    for row in data_rows:
        if isinstance(row, dict):
            pass
        elif isinstance(row, (list, tuple)) and data_rows:
            headers_row = data_rows[0] if row is data_rows[0] else (data_rows[0] if data_rows else [])
            if hasattr(headers_row, "__iter__") and not isinstance(headers_row, dict):
                row = dict(zip(headers_row, row))
            else:
                continue
        else:
            continue

        team_key = find_column(row, "team")
        team = normalize_team_name(row.get(team_key or "team", row.get("Team", "")))
        adj_o_key = find_column(row, "adj_o")
        adj_d_key = find_column(row, "adj_d")
        tempo_key = find_column(row, "adj_tempo")
        barthag_key = find_column(row, "barthag")
        to_key = find_column(row, "to_rate")
        orb_key = find_column(row, "orb_rate")
        ft_key = find_column(row, "ft_rate")
        sos_key = find_column(row, "sos")
        three_rate_key = find_column(row, "three_rate")
        three_pct_key = find_column(row, "three_pct")
        wab_key = find_column(row, "wab")
        elite_sos_key = find_column(row, "elite_sos")
        qual_o_key = find_column(row, "qual_o")
        qual_d_key = find_column(row, "qual_d")
        qual_barthag_key = find_column(row, "qual_barthag")
        conf_adj_o_key = find_column(row, "conf_adj_o")
        conf_adj_d_key = find_column(row, "conf_adj_d")
        record_key = find_column(row, "record")
        conf_win_pct_key = find_column(row, "conf_win_pct")

        adj_o = parse_value(row.get(adj_o_key or "adj_o"), 100.0)
        adj_d = parse_value(row.get(adj_d_key or "adj_d"), 100.0)
        adj_tempo = parse_value(row.get(tempo_key or "adj_tempo"), 67.5)
        barthag = parse_value(row.get(barthag_key or "barthag"), 0.5)
        to_rate = parse_value(row.get(to_key or "to_rate"), None)
        orb_rate = parse_value(row.get(orb_key or "orb_rate"), None)
        ft_rate = parse_value(row.get(ft_key or "ft_rate"), None)
        sos = parse_value(row.get(sos_key or "sos"), None)
        three_rate = parse_value(row.get(three_rate_key) if three_rate_key else None, None)
        three_pct = parse_value(row.get(three_pct_key) if three_pct_key else None, None)
        wab = parse_value(row.get(wab_key) if wab_key else None, None)
        elite_sos = parse_value(row.get(elite_sos_key) if elite_sos_key else None, None)
        qual_o = parse_value(row.get(qual_o_key) if qual_o_key else None, None)
        qual_d = parse_value(row.get(qual_d_key) if qual_d_key else None, None)
        qual_barthag = parse_value(row.get(qual_barthag_key) if qual_barthag_key else None, None)
        conf_adj_o = parse_value(row.get(conf_adj_o_key) if conf_adj_o_key else None, None)
        conf_adj_d = parse_value(row.get(conf_adj_d_key) if conf_adj_d_key else None, None)

        win_pct = None
        record_val = row.get(record_key or "record", "")
        if record_val:
            m = re.match(r"(\d+)-(\d+)", str(record_val).strip())
            if m:
                w, l = int(m.group(1)), int(m.group(2))
                if w + l > 0:
                    win_pct = w / (w + l)
        conf_win_pct = parse_value(row.get(conf_win_pct_key) if conf_win_pct_key else None, None)
        if conf_win_pct is not None and conf_win_pct > 1.0:
            conf_win_pct = conf_win_pct / 100.0

        if not team and adj_o is None and adj_d is None:
            continue
        if not team:
            team = f"Unknown_{len(canonical_rows)}"
        row_out = {
            "team": team,
            "adj_o": adj_o,
            "adj_d": adj_d,
            "adj_tempo": adj_tempo,
            "barthag": barthag,
        }
        if to_rate is not None:
            row_out["to_rate"] = to_rate
        if orb_rate is not None:
            row_out["orb_rate"] = orb_rate
        if ft_rate is not None:
            ft = ft_rate / 100.0 if ft_rate > 1.5 else ft_rate
            row_out["ft_pct"] = ft
        if sos is not None:
            row_out["sos"] = sos
        if three_rate is not None:
            row_out["three_rate"] = three_rate
        if three_pct is not None:
            row_out["three_pct"] = three_pct
        if wab is not None:
            row_out["wab"] = wab
        if elite_sos is not None:
            row_out["elite_sos"] = elite_sos
        if qual_o is not None:
            row_out["qual_o"] = qual_o
        if qual_d is not None:
            row_out["qual_d"] = qual_d
        if qual_barthag is not None:
            row_out["qual_barthag"] = qual_barthag
        if conf_adj_o is not None:
            row_out["conf_adj_o"] = conf_adj_o
        if conf_adj_d is not None:
            row_out["conf_adj_d"] = conf_adj_d
        if win_pct is not None:
            row_out["win_pct"] = win_pct
        if conf_win_pct is not None:
            row_out["conf_win_pct"] = conf_win_pct
        canonical_rows.append(row_out)
    return canonical_rows


def _load_from_local_csv(path):
    """Load and parse a local CSV file. Returns (canonical_rows, raw_rows).
    raw_rows is kept for diagnostics when canonical is empty.
    """
    if not os.path.isfile(path):
        print(f"  ERROR: File not found: {os.path.abspath(path)}")
        return [], []
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            data_rows = list(reader)
        if not data_rows:
            print(f"  ERROR: CSV has no data rows: {path}")
            return [], []
        canonical = _parse_rows_to_canonical(data_rows)
        if not canonical and data_rows:
            cols = list(data_rows[0].keys())
            print(f"  WARNING: Parsed 0 teams from {len(data_rows)} rows.")
            print(f"  CSV columns found: {cols[:20]}")
            sample = {k: data_rows[0][k] for k in cols[:6]}
            print(f"  First row sample: {sample}")
        return canonical, data_rows
    except Exception as e:
        print(f"  Failed to parse {path}: {e}")
        return [], []


# Column index mapping for team-tables_each.php gdata array
_ADV_COL_MAP = {
    0:  "team",
    1:  "adj_o",
    2:  "adj_d",
    3:  "barthag",
    4:  "record",       # "35-5" string
    5:  "wins",
    6:  "games",
    7:  "efg_pct",      # eFG% offensive
    8:  "efg_d",        # eFG% defensive
    9:  "ft_rate",      # FT Rate (FTA/FGA)
    10: "ft_rate_d",
    11: "to_rate",      # TOV% offensive (lower is better)
    12: "to_rate_d",    # TOV% defensive (higher is better — forces TOs)
    13: "orb_rate",     # O Reb%
    14: "opp_orb_rate",
    15: "raw_tempo",
    16: "two_pt_pct",   # 2P%
    17: "two_pt_pct_d",
    18: "three_pt_pct", # 3P%
    19: "three_pt_pct_d",
    20: "blk_rate",     # Blk%
    21: "blked_rate",
    22: "ast_rate",     # Ast%
    23: "opp_ast_rate",
    24: "three_pt_rate", # 3P Rate (3PA/FGA)
    25: "three_pt_rate_d",
    26: "adj_tempo",
    27: "avg_height",   # Avg Hgt. (inches) — empty in aggregate; compiled from rosters
    28: "eff_height",   # Eff. Hgt.
    35: "ft_pct",       # FT% (free throw shooting %)
    36: "opp_ft_pct",   # Opponent FT%
    37: "ppp_off",      # Points per possession offense
    38: "ppp_def",      # Points per possession defense
    39: "avg_experience",  # Average months of college experience
}


def _adv_data_looks_valid(adv, min_teams=50):
    """Return True if adv data has real Four Factors (not all zeros)."""
    if not adv or len(adv) < min_teams:
        return False
    sample = list(adv.values())[:min(min_teams, len(adv))]
    with_efg = sum(1 for r in sample if r.get("efg_pct") and float(r.get("efg_pct", 0)) > 5)
    with_blk = sum(1 for r in sample if r.get("blk_rate") and float(r.get("blk_rate", 0)) > 1)
    return with_efg >= len(sample) * 0.5 and with_blk >= len(sample) * 0.5


def _derive_ppg(row):
    """Derive ppg/opp_ppg from ppp_off, ppp_def, adj_tempo when missing."""
    tempo = row.get("adj_tempo")
    if tempo is None:
        return
    ppp_o = row.get("ppp_off")
    ppp_d = row.get("ppp_def")
    if ppp_o is not None and row.get("ppg") is None:
        row["ppg"] = round(ppp_o * tempo, 1)
    if ppp_d is not None and row.get("opp_ppg") is None:
        row["opp_ppg"] = round(ppp_d * tempo, 1)


def fetch_torvik_adv_season(year=2026):
    """Fetch advanced stats (Four Factors, height, experience) from team-tables_each.php.
    Uses POST trick to bypass JS browser verification.
    Returns dict of {normalized_team_name: stats_dict} or empty dict on failure.
    """
    url = (f"https://barttorvik.com/team-tables_each.php"
           f"?year={year}&top=0&conlimit=All&venue=All&type=All&yax=3")
    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Referer": "https://barttorvik.com/",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        r = requests.post(url, data={"js_test_submitted": "1"},
                          headers=headers, timeout=30)
        r.raise_for_status()
        content = r.text
    except Exception as e:
        print(f"  fetch_torvik_adv {year}: {e}")
        return {}

    if "Verifying" in content[:500]:
        print(f"  fetch_torvik_adv {year}: still blocked by JS check")
        return {}

    m = re.search(r"var gdata\s*=\s*(\[.*?\]);", content, re.DOTALL)
    if not m:
        print(f"  fetch_torvik_adv {year}: gdata not found in response")
        return {}

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        print(f"  fetch_torvik_adv {year}: JSON parse error: {e}")
        return {}

    results = {}
    for row in data:
        if not isinstance(row, list) or len(row) < 27:
            continue
        team_raw = row[0]
        if not isinstance(team_raw, str) or not team_raw.strip():
            continue
        team = normalize_team_name(team_raw)
        rec = {}
        for idx, field in _ADV_COL_MAP.items():
            if idx >= len(row):
                continue
            val = row[idx]
            if field == "record":
                # Parse win_pct from record string like "35-5"
                s = str(val).replace("\u2013", "-").replace("\u2014", "-")
                m2 = re.match(r"(\d+)-(\d+)", s)
                if m2:
                    w, l = int(m2.group(1)), int(m2.group(2))
                    if w + l > 0:
                        rec["win_pct"] = round(w / (w + l), 4)
                        rec["wins"] = w
                        rec["games"] = w + l
            elif field in ("team", "wins", "games"):
                continue
            elif isinstance(val, (int, float)) and val != "":
                rec[field] = float(val)
            elif isinstance(val, str) and val.strip():
                try:
                    rec[field] = float(val.strip())
                except ValueError:
                    pass

        # Normalize avg_experience: Torvik stores it in months
        # Convert to 0–1 scale: 0 = all freshmen (~12mo), 1 = all seniors (~48mo)
        if "avg_experience" in rec:
            months = rec["avg_experience"]
            rec["avg_experience"] = months  # keep raw months too
            rec["experience"] = max(0.0, min(1.0, (months - 12.0) / 36.0))

        if rec:
            results[team] = rec

    return results


def fetch_torvik_season(year=2026, from_csv=None):
    """Download Torvik team data for the season and save in canonical form.
    If from_csv is set, load from that path instead of downloading.
    """
    url_csv = f"https://barttorvik.com/{year}_team_results.csv"
    url_json = f"https://barttorvik.com/{year}_team_results.json"
    out_csv = os.path.join(DATA_DIR, f"torvik_{year}.csv")
    out_json = os.path.join(DATA_DIR, f"torvik_{year}.json")

    canonical_rows = []

    if from_csv:
        from_csv_abs = os.path.abspath(from_csv)
        if not os.path.isfile(from_csv_abs):
            print(f"ERROR: --from-csv file not found: {from_csv_abs}")
            return False
        print(f"Loading from local CSV: {from_csv} ({os.path.getsize(from_csv_abs)} bytes)")
        canonical_rows, _ = _load_from_local_csv(from_csv_abs)
        if not canonical_rows:
            return False
        # Protect: don't overwrite the source file with canonical output
        out_csv_abs = os.path.abspath(out_csv)
        if from_csv_abs == out_csv_abs:
            out_csv = os.path.join(DATA_DIR, f"torvik_{year}_canonical.csv")
            print(f"  Source == output path; writing canonical CSV to {out_csv} instead.")
    else:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/csv,application/json",
            "Referer": "https://barttorvik.com/",
        }

        data_rows = None
        for url, content_type in [(url_json, "json"), (url_csv, "csv")]:
            try:
                r = requests.get(url, headers=headers, timeout=30)
                r.raise_for_status()
                text = r.text.strip()
                if not text or text.lstrip().startswith("<") or "Verifying" in text[:500]:
                    continue
                if content_type == "json":
                    try:
                        raw = json.loads(text)
                        if isinstance(raw, list):
                            data_rows = raw
                        elif isinstance(raw, dict) and "teams" in raw:
                            data_rows = raw["teams"]
                        elif isinstance(raw, dict):
                            data_rows = list(raw.values()) if raw else []
                        else:
                            data_rows = []
                        break
                    except json.JSONDecodeError:
                        pass
                else:
                    reader = csv.DictReader(text.splitlines())
                    data_rows = list(reader)
                    break
            except Exception as e:
                print(f"  {url}: {e}")
                continue

        if data_rows:
            canonical_rows = _parse_rows_to_canonical(data_rows)

        if not canonical_rows:
            local_path = os.path.join(DATA_DIR, f"torvik_{year}.csv")
            if os.path.isfile(local_path):
                print("Download blocked (Cloudflare). Trying existing local file...")
                canonical_rows, _ = _load_from_local_csv(local_path)

        if not canonical_rows:
            raw_path = os.path.join(DATA_DIR, f"torvik_{year}_raw.csv")
            print("Could not fetch Torvik data (site may use bot protection).")
            print("MANUAL DOWNLOAD REQUIRED (pick one):")
            print(f"  Option A — Browser:")
            print(f"    1. Open {url_csv} in your browser")
            print(f"    2. Save as: {raw_path}")
            print(f"    3. Run: python scripts/fetch_torvik.py --from-csv {raw_path} {year}")
            print(f"  Option B — curl:")
            print(f'    curl -sL -o "{raw_path}" "{url_csv}" -H "User-Agent: Mozilla/5.0"')
            print(f"    python scripts/fetch_torvik.py --from-csv {raw_path} {year}")
            return False

    # Reject bad data: >50% Unknown or no real team names
    unknown_count = sum(1 for r in canonical_rows if str(r.get("team", "")).startswith("Unknown"))
    if unknown_count > len(canonical_rows) * 0.5:
        print("ERROR: Parsed data has mostly 'Unknown' teams — column mapping failed.")
        print("  Expected CSV columns: Team, AdjOE, AdjDE, Adj T. (or AdjT), Barthag")
        print("  Manual fix: Download CSV from barttorvik.com in browser, save as data/torvik_YYYY.csv,")
        print("  then: python scripts/fetch_torvik.py --from-csv data/torvik_YYYY.csv YYYY")
        return False

    # Fetch and merge advanced stats (Four Factors, experience, height) if not from local CSV
    if not from_csv:
        print(f"  Fetching advanced stats (Four Factors, experience)...")
        adv = fetch_torvik_adv_season(year)
        if adv and _adv_data_looks_valid(adv):
            merged = 0
            for row in canonical_rows:
                key = normalize_team_name(row.get("team", ""))
                if key in adv:
                    for field, val in adv[key].items():
                        if field not in row or row[field] is None:
                            row[field] = val
                    merged += 1
            print(f"  Merged advanced stats for {merged}/{len(canonical_rows)} teams")
        elif adv:
            print("  Advanced stats rejected (corrupt/zero data for this year)")
        else:
            print("  Advanced stats unavailable (will retry on next fetch)")

    # Derive ppg/opp_ppg from ppp_off, ppp_def, adj_tempo when missing
    for row in canonical_rows:
        _derive_ppg(row)

    with open(out_json, "w") as f:
        json.dump(canonical_rows, f, indent=2)
    print(f"Wrote {len(canonical_rows)} teams to {out_json}")

    with open(out_csv, "w", newline="") as f:
        optional = ["to_rate", "orb_rate", "ft_rate", "ft_pct", "sos", "three_rate", "three_pct",
                     "wab", "elite_sos", "qual_o", "qual_d", "qual_barthag",
                     "conf_adj_o", "conf_adj_d", "win_pct", "conf_win_pct",
                     "efg_pct", "efg_d", "ft_rate_d", "to_rate_d", "opp_orb_rate",
                     "two_pt_pct", "two_pt_pct_d", "three_pt_pct", "three_pt_pct_d",
                     "blk_rate", "ast_rate", "three_pt_rate", "three_pt_rate_d",
                     "avg_height", "eff_height", "avg_experience", "experience",
                     "ppg", "opp_ppg"]
        extra_fields = [f for f in optional if any(f in r for r in canonical_rows)]
        fieldnames = ["team", "adj_o", "adj_d", "adj_tempo", "barthag"] + extra_fields
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(canonical_rows)
    print(f"Wrote {out_csv}")
    return True


def fetch_adv_and_merge(year):
    """Fetch advanced stats and merge into existing torvik_YYYY.json in-place."""
    out_json = os.path.join(DATA_DIR, f"torvik_{year}.json")
    if not os.path.isfile(out_json):
        print(f"  {year}: {out_json} not found, skipping")
        return False
    with open(out_json) as f:
        canonical_rows = json.load(f)
    adv = fetch_torvik_adv_season(year)
    if not adv:
        print(f"  {year}: No advanced stats returned")
        return False
    if not _adv_data_looks_valid(adv):
        print(f"  {year}: Advanced stats rejected (corrupt/zero data), skipping merge")
        return False
    merged = 0
    for row in canonical_rows:
        key = normalize_team_name(row.get("team", ""))
        if key in adv:
            for field, val in adv[key].items():
                if field not in row or row[field] is None:
                    row[field] = val
            merged += 1
    for row in canonical_rows:
        _derive_ppg(row)
    with open(out_json, "w") as f:
        json.dump(canonical_rows, f, indent=2)
    print(f"  {year}: merged advanced stats for {merged}/{len(canonical_rows)} teams → {out_json}")
    return True


def main():
    year = 2026
    from_csv = None
    adv_only = "--adv-only" in sys.argv
    all_years = "--all" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--adv-only", "--all")]

    if "--from-csv" in args:
        i = args.index("--from-csv")
        if i + 1 < len(args):
            from_csv = args[i + 1]
            args = args[:i] + args[i + 2:]
    for a in args:
        try:
            year = int(a)
            break
        except ValueError:
            pass

    if all_years:
        years = [y for y in range(2010, 2027) if y != 2020]
        print(f"Fetching advanced stats for all years: {years}")
        for yr in years:
            print(f"\n--- {yr} ---")
            fetch_adv_and_merge(yr)
        return

    if adv_only:
        print(f"Fetching advanced stats for {year}...")
        ok = fetch_adv_and_merge(year)
        sys.exit(0 if ok else 1)

    print(f"Fetching Torvik data for {year}...")
    ok = fetch_torvik_season(year, from_csv=from_csv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
