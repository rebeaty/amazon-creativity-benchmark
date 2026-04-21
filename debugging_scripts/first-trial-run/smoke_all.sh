#!/usr/bin/env bash
# Smoke-test all TRY_IT datasets with max_instances=1.
# Writes per-dataset status + errors to debugging_scripts/first-trial-run/smoke_results/
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source .env 2>/dev/null || true
source .venv/Scripts/activate 2>/dev/null || true
export PYTHONUTF8=1

MODEL="google/gemini-2.5-flash-lite"
SUITE="trial"
MAX_INSTANCES=1
TIMEOUT=180

RESULTS_DIR="$SCRIPT_DIR/smoke_results"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/_summary.tsv"
echo -e "status\tdataset\telapsed_s\terror_snippet" > "$SUMMARY"

# Read TRY_IT datasets from triage.json
DATASETS=()
while IFS= read -r ds; do
    DATASETS+=("$ds")
done < <(python -c "
import json
t = json.load(open('debugging_scripts/first-trial-run/triage.json'))
for d, info in t.items():
    if info['category'] == 'TRY_IT':
        print(d)
" | tr -d '\r' | sort)

TOTAL=${#DATASETS[@]}
echo "Smoke-testing $TOTAL TRY_IT datasets with max_instances=$MAX_INSTANCES"
echo "Results -> $SUMMARY"
echo ""

PASSED=0; FAILED=0
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    idx=$((i + 1))
    log="$RESULTS_DIR/${dataset}.log"
    start=$(date +%s)
    model_safe="${MODEL//\//_}"
    stats_path="benchmark_output/runs/$SUITE/${dataset}_model=${model_safe}/stats.json"

    # Remove old stats.json so we know if THIS run produced one
    rm -f "$stats_path" 2>/dev/null

    printf "[%2d/%d] %-32s " "$idx" "$TOTAL" "$dataset"

    if [ ! -f "eval_scripts/${dataset}.sh" ]; then
        echo "NO_SCRIPT"
        echo -e "NO_SCRIPT\t$dataset\t0\t(eval script missing)" >> "$SUMMARY"
        FAILED=$((FAILED + 1))
        continue
    fi

    timeout $TIMEOUT bash "eval_scripts/${dataset}.sh" "$MODEL" "$SUITE" "$MAX_INSTANCES" > "$log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))

    if [ -f "$stats_path" ]; then
        echo "PASS ($elapsed s)"
        echo -e "PASS\t$dataset\t$elapsed\t" >> "$SUMMARY"
        PASSED=$((PASSED + 1))
    else
        # Extract meaningful error snippet
        snippet=$(grep -aE "^(Error|Traceback|.*Error:|.*Exception)" "$log" | tail -2 | tr '\t' ' ' | tr '\n' '|' | cut -c1-200)
        if [ $rc -eq 124 ]; then
            echo "TIMEOUT"
            echo -e "TIMEOUT\t$dataset\t$elapsed\t(killed at ${TIMEOUT}s)" >> "$SUMMARY"
        else
            echo "FAIL ($elapsed s)"
            echo -e "FAIL\t$dataset\t$elapsed\t$snippet" >> "$SUMMARY"
        fi
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== DONE ==="
echo "Passed: $PASSED / $TOTAL"
echo "Failed: $FAILED / $TOTAL"
echo ""
echo "Summary: $SUMMARY"
