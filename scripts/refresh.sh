#!/usr/bin/env bash
# refresh.sh — Keep current-year data up to date.
# Run once per day during tournament season, or any time before generating picks.
#
# Usage:
#   bash scripts/refresh.sh          # refresh current year (2026)
#   bash scripts/refresh.sh 2025     # refresh a specific year

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

YEAR="${1:-2026}"
echo "=== Refreshing data for $YEAR ==="

# 1. Torvik (efficiency, four factors, experience)
echo ""
echo "--- Torvik ---"
python scripts/fetch_torvik.py "$YEAR"

# 2. EvanMiya (star ratings, depth, player BPR)
#    Requires manual CSV download from evanmiya.com → data/evanmiya_teams_YYYY.csv
echo ""
echo "--- EvanMiya ---"
python scripts/fetch_evanmiya.py "$YEAR" || echo "  (skipped — place evanmiya_teams_${YEAR}.csv in data/ to enable)"

# 3. Merge all sources into teams_merged_YYYY.json
echo ""
echo "--- Merging sources ---"
python scripts/fetch_data.py --no-fetch "$YEAR"

# 4. Validate: every bracket team has real stats
echo ""
echo "--- Validating bracket coverage ---"
python scripts/validate_data.py "$YEAR"

echo ""
echo "=== Done. Run: python run.py --bracket data/bracket_${YEAR}.json ==="
