"""
Fetch team stats from KenPom (optional; requires subscription and kenpompy).
Saves to data/kenpom_YYYY.csv with canonical schema: team, adj_o, adj_d, adj_tempo, luck.
Set KENPOM_EMAIL and KENPOM_PASSWORD in the environment.
"""
import csv
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def normalize_team_name(name):
    if not name or not isinstance(name, str):
        return ""
    return " ".join(name.strip().split())


def fetch_kenpom_season(year=2026):
    """Download KenPom efficiency (and luck if available) for the season."""
    try:
        from kenpompy.utils import login
        import kenpompy.summary as kp
    except ImportError:
        print("KenPom fetch is optional. Install: pip install kenpompy")
        print("Then set KENPOM_EMAIL and KENPOM_PASSWORD (KenPom subscription required).")
        return False

    email = os.environ.get("KENPOM_EMAIL")
    password = os.environ.get("KENPOM_PASSWORD")
    if not email or not password:
        print("Set KENPOM_EMAIL and KENPOM_PASSWORD to use KenPom data.")
        return False

    try:
        browser = login(email, password)
        eff = kp.get_efficiency(browser, year=year)
    except Exception as e:
        print(f"KenPom login or fetch failed: {e}")
        return False

    # Map common KenPom column names to canonical
    # kenpompy may return DataFrame with columns like Team, AdjO, AdjD, AdjT, Luck%
    canonical_rows = []
    for _, row in eff.iterrows():
        team = normalize_team_name(row.get("Team", row.get("team", "")))
        adj_o = _num(row.get("AdjO") or row.get("adj_o"), 100)
        adj_d = _num(row.get("AdjD") or row.get("adj_d"), 100)
        adj_tempo = _num(row.get("AdjT") or row.get("adj_tempo"), 67.5)
        luck = _num(row.get("Luck") or row.get("Luck%"), 0)
        if not team:
            continue
        canonical_rows.append({
            "team": team,
            "kp_adj_o": adj_o,
            "kp_adj_d": adj_d,
            "kp_adj_tempo": adj_tempo,
            "luck": luck,
        })

    out_csv = os.path.join(DATA_DIR, f"kenpom_{year}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["team", "kp_adj_o", "kp_adj_d", "kp_adj_tempo", "luck"])
        w.writeheader()
        w.writerows(canonical_rows)
    print(f"Wrote {len(canonical_rows)} teams to {out_csv}")
    return True


def _num(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def main():
    year = 2026
    if len(sys.argv) > 1:
        try:
            year = int(sys.argv[1])
        except ValueError:
            pass
    print(f"Fetching KenPom data for {year}...")
    ok = fetch_kenpom_season(year)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
