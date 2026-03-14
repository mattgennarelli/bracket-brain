#!/usr/bin/env bash
# Run full model evaluation: calibration + benchmark, then append summary to MODEL_IMPROVEMENT_LOG.
# Usage: ./scripts/run_full_eval.sh [--report-only]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DOCS="$ROOT/docs"
DATA="$ROOT/data"
TIMESTAMP=$(date +%Y-%m-%d_%H%M)

mkdir -p "$DOCS"

echo "=== Calibration ==="
if [[ "$1" == "--report-only" ]]; then
    python3 scripts/calibrate.py --report-only --save-report "$DOCS/cal_report_$TIMESTAMP.json" 2>&1 | tee "$DOCS/cal_output_$TIMESTAMP.txt"
else
    python3 scripts/calibrate.py --save-report "$DOCS/cal_report_$TIMESTAMP.json" 2>&1 | tee "$DOCS/cal_output_$TIMESTAMP.txt"
fi

echo ""
echo "=== Benchmark (bracket quality) ==="
python3 scripts/benchmark.py --bracket-quality 2>&1 | tee "$DOCS/benchmark_output_$TIMESTAMP.txt"

echo ""
echo "=== Appending to MODEL_IMPROVEMENT_LOG ==="
{
    echo ""
    echo "## Eval run $TIMESTAMP"
    echo ""
    echo "### Calibration"
    grep -E "Games scored|Accuracy|Brier score|Improvement" "$DOCS/cal_output_$TIMESTAMP.txt" 2>/dev/null || true
    echo ""
    echo "### Benchmark"
    grep -E "full model|champion|Brier" "$DOCS/benchmark_output_$TIMESTAMP.txt" 2>/dev/null | tail -20 || true
    echo ""
} >> "$DOCS/MODEL_IMPROVEMENT_LOG.md" 2>/dev/null || true

echo "Done. Report: $DOCS/cal_report_$TIMESTAMP.json"
echo "Log: $DOCS/MODEL_IMPROVEMENT_LOG.md"
