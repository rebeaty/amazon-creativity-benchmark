#!/bin/bash
# =============================================================================
# Debug a SINGLE dataset — use this to test your setup before the full run
# Usage: ./debug_single.sh <dataset_name>
# =============================================================================

set -euo pipefail

DATASET="${1:?Usage: ./debug_single.sh <dataset_name>}"
LOG_FILE="debug_logs/${DATASET//\//_}_single.log"
mkdir -p debug_logs debug_results

echo "Debugging single dataset: $DATASET"
echo "Log: $LOG_FILE"
echo ""

claude -p "
You are debugging the HELM evaluation pipeline for dataset: \"$DATASET\".

Follow the debugging protocol from CLAUDE.md exactly. Work through all 8 checks:
1. Data availability (download if needed)
2. Prompt template sanity check (cross-ref with paper from registry_master.yaml)
3. Generation config check (temperature, max_tokens appropriateness)
4. Raw generation sanity check (run 2-3 instances)
5. Metric selection verification (registry_metrics.yaml)
6. Evaluation metric execution (debug errors)
7. Aggregation correctness
8. Results saving

You have up to 10 debugging attempts. On success, write debug_results/${DATASET//\//_}_PASSED.json.
On failure after 10 attempts, write debug_results/${DATASET//\//_}_FAILED.json with the last error.

Be thorough. Show your work at each step.
" \
    --max-turns 30 \
    --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "── Done ──"
if [[ -f "debug_results/${DATASET//\//_}_PASSED.json" ]]; then
    echo "Result: PASSED ✓"
elif [[ -f "debug_results/${DATASET//\//_}_FAILED.json" ]]; then
    echo "Result: FAILED ✗"
    cat "debug_results/${DATASET//\//_}_FAILED.json"
else
    echo "Result: No result file written — check the log."
fi