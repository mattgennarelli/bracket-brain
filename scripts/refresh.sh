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
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "=== Refreshing data for $YEAR ==="

# 1. Torvik (efficiency, four factors, experience)
echo ""
echo "--- Torvik ---"
"$PYTHON_BIN" scripts/fetch_torvik.py "$YEAR"

# 2. EvanMiya (star ratings, depth, player BPR)
#    Requires manual CSV download from evanmiya.com → data/evanmiya_teams_YYYY.csv
echo ""
echo "--- EvanMiya ---"
"$PYTHON_BIN" scripts/fetch_evanmiya.py "$YEAR" || echo "  (skipped — place evanmiya_teams_${YEAR}.csv in data/ to enable)"

# 3. Merge all sources into teams_merged_YYYY.json
echo ""
echo "--- Merging sources ---"
"$PYTHON_BIN" scripts/fetch_data.py --no-fetch "$YEAR"

# 4. Validate: every bracket team has real stats
echo ""
echo "--- Validating bracket coverage ---"
"$PYTHON_BIN" scripts/validate_data.py "$YEAR"

# 5. Settle yesterday's pending picks (requires ODDS_API_KEY)
if [ -n "${ODDS_API_KEY:-}" ]; then
  echo ""
  echo "--- Settling yesterday's picks ---"
  "$PYTHON_BIN" scripts/settle_bets.py || echo "  (settle failed — check ODDS_API_KEY)"

  # 6. Save today's picks
  echo ""
  echo "--- Saving today's picks ---"
  "$PYTHON_BIN" scripts/save_bets.py || echo "  (no picks today or API error)"
else
  echo ""
  echo "--- Skipping picks (no ODDS_API_KEY set) ---"
fi

echo ""
echo "=== Done. Run: ${PYTHON_BIN} run.py --bracket data/bracket_${YEAR}.json ==="
