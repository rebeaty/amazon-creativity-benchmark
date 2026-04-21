#!/usr/bin/env bash
# Smoke-test the 12 previously-skipped multimodal/gdrive datasets.
# Uses max_instances=1 and a slightly longer timeout (300s) since these
# typically need to fetch images from HF or external URLs.
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
TIMEOUT=300

RESULTS_DIR="$SCRIPT_DIR/smoke_results"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/_summary_multimodal.tsv"
echo -e "status\tdataset\telapsed_s\terror_snippet" > "$SUMMARY"

DATASETS=(ii_bench irfl banner_request_400 ava creation_mmbench creative_pair esp_dataset muse_perception puzzleworld yesbut vgsg storyer)

TOTAL=${#DATASETS[@]}
echo "Smoke-testing $TOTAL multimodal datasets (max_instances=$MAX_INSTANCES, timeout=${TIMEOUT}s)"
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

    rm -f "$stats_path" 2>/dev/null
    printf "[%2d/%d] %-30s " "$idx" "$TOTAL" "$dataset"

    timeout $TIMEOUT bash "eval_scripts/${dataset}.sh" "$MODEL" "$SUITE" "$MAX_INSTANCES" > "$log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))

    if [ -f "$stats_path" ]; then
        echo "PASS (${elapsed}s)"
        echo -e "PASS\t$dataset\t$elapsed\t" >> "$SUMMARY"
        PASSED=$((PASSED + 1))
    else
        # Extract meaningful error
        snippet=$(grep -aE "^(Error|.*Error:|.*Exception|FileNotFoundError|ModuleNotFound|gated|HTTP Error|TIMEOUT)" "$log" | tail -2 | tr '\t' ' ' | tr '\n' '|' | cut -c1-200)
        if [ $rc -eq 124 ]; then
            echo "TIMEOUT"
            echo -e "TIMEOUT\t$dataset\t$elapsed\t(killed at ${TIMEOUT}s)" >> "$SUMMARY"
        else
            echo "FAIL (${elapsed}s)"
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
