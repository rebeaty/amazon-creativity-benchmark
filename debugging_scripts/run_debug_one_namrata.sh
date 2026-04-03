#!/bin/bash
# =============================================================================
# Debug ONE dataset for Namrata
# Picks the next pending dataset, runs debugging, moves to done on success.
# Usage: ./run_debug_one_namrata.sh              (auto-picks next pending)
#        ./run_debug_one_namrata.sh <dataset>    (debug a specific dataset)
# =============================================================================

set -euo pipefail

DEBUGGER_NAME="Namrata"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HELPER="$SCRIPT_DIR/_debug_helper.py"

# ── Pick dataset ─────────────────────────────────────────────────────────────
if [[ -n "${1:-}" ]]; then
    DATASET="$1"
else
    DATASET=$(python3 "$HELPER" next "$DEBUGGER_NAME")
    if [[ "$DATASET" == "__EMPTY__" ]]; then
        echo "All datasets done for $DEBUGGER_NAME!"
        python3 "$HELPER" status "$DEBUGGER_NAME"
        exit 0
    fi
fi

SAFE_NAME="${DATASET//\//_}"
LOG_FILE="debug_logs/${SAFE_NAME}_${DEBUGGER_NAME}.log"
LOG_PATH="$(pwd)/$LOG_FILE"
PASSED_FILE="debug_results/${SAFE_NAME}_PASSED.json"
FAILED_FILE="debug_results/${SAFE_NAME}_FAILED.json"
START_TIME=$(date +%s)
mkdir -p debug_logs debug_results

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Debugger:  $DEBUGGER_NAME"
echo "Dataset:   $DATASET"
echo "Started:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log:       $LOG_PATH"
python3 "$HELPER" status "$DEBUGGER_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Run Claude Code ──────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Starting Claude Code debugging..."
CLAUDE_EXIT=0
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

You have up to 10 debugging attempts. On success, write debug_results/${SAFE_NAME}_PASSED.json.
On failure after 10 attempts, write debug_results/${SAFE_NAME}_FAILED.json with the last error.

Be thorough. Show your work at each step.
" \
    --max-turns 5 \
    --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
    2>&1 | tee "$LOG_FILE" || CLAUDE_EXIT=$?

# ── Compute duration ─────────────────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$(( END_TIME - START_TIME ))
MINUTES=$(( DURATION / 60 ))
SECONDS=$(( DURATION % 60 ))

# ── Report results ───────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RESULTS: $DATASET"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Duration:  ${MINUTES}m ${SECONDS}s"
echo "Finished:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log:       $LOG_PATH"

if [[ "$CLAUDE_EXIT" -ne 0 ]]; then
    echo "Claude exit code: $CLAUDE_EXIT"
fi

if [[ -f "$PASSED_FILE" ]]; then
    echo "Status:    PASSED ✓"
    python3 "$HELPER" done "$DEBUGGER_NAME" "$DATASET"
elif [[ -f "$FAILED_FILE" ]]; then
    echo "Status:    FAILED ✗"
    echo ""
    echo "── Failure details ──"
    cat "$FAILED_FILE"
    python3 "$HELPER" done "$DEBUGGER_NAME" "$DATASET"
else
    echo "Status:    NO RESULT (dataset stays in pending)"
    echo ""
    echo "── Last 30 lines of log ──"
    tail -30 "$LOG_FILE" 2>/dev/null || echo "(log empty or missing)"
    echo ""
    echo "Full log: $LOG_PATH"
fi

echo ""
python3 "$HELPER" status "$DEBUGGER_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
