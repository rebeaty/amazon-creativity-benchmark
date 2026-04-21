#!/bin/bash
# =============================================================================
# Re-run datasets that got TIMEOUT-skipped during Task A at 120s.
# These passed smoke at max_instances=1 but timed out at max=10 because
# Hugging Face Xet downloads / multimodal image fetches exceeded the
# orchestrator's 120s eval budget.
#
# Uses an extended EVAL_TIMEOUT (default 360s) and the same
# google/gemini-2.5-flash-lite model & trial_10inst suite.
#
# Usage:
#   ./rerun_false_skips.sh [DATASET ...]
#   ./rerun_false_skips.sh              # re-run the known false-skip list
#   ./rerun_false_skips.sh litbench     # single dataset
# =============================================================================

set -uo pipefail

MODEL="google/gemini-2.5-flash-lite"
SUITE="trial_10inst"
MAX_INSTANCES=10
EVAL_TIMEOUT="${EVAL_TIMEOUT:-360}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_LIST=(
    # Namrata
    litbench
    creation_mmbench
    # Sai
    conceptual_design
    # rajkumar
    arastories
    arena_hard_creative
    banner_request_400
    creatset
    cs4
    liveideabench
    speak_to_structure
    ss_gen
    tinystories
)

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_LIST[@]}")
fi

echo "============================================================"
echo "  False-skip re-run"
echo "  Model:         $MODEL"
echo "  Suite:         $SUITE"
echo "  Max instances: $MAX_INSTANCES"
echo "  Eval timeout:  ${EVAL_TIMEOUT}s"
echo "  Datasets:      ${DATASETS[*]}"
echo "============================================================"

model_safe="${MODEL//\//_}"
PASSED=0; FAILED=0
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "── $ds ──────────────────────────────────────────────────"
    if compgen -G "benchmark_output/runs/$SUITE/${ds}_*model=${model_safe}/stats.json" > /dev/null 2>&1 \
        || compgen -G "benchmark_output/runs/$SUITE/${ds}:*model=${model_safe}/stats.json" > /dev/null 2>&1; then
        echo "  [ALREADY DONE] stats.json exists"
        (( PASSED++ ))
        continue
    fi
    log="$REPO_ROOT/debugging_scripts/debug_logs/${ds}_rerun_$(date +%Y%m%d_%H%M%S).log"
    timeout "$EVAL_TIMEOUT" ./"eval_scripts/${ds}.sh" "$MODEL" "$SUITE" "$MAX_INSTANCES" > "$log" 2>&1
    rc=$?
    if compgen -G "benchmark_output/runs/$SUITE/${ds}_*model=${model_safe}/stats.json" > /dev/null 2>&1 \
        || compgen -G "benchmark_output/runs/$SUITE/${ds}:*model=${model_safe}/stats.json" > /dev/null 2>&1; then
        echo "  [SUCCESS] stats.json written — see $log"
        (( PASSED++ ))
    else
        echo "  [FAILED] rc=$rc — see $log"
        (( FAILED++ ))
    fi
done

echo ""
echo "============================================================"
echo "  Re-run summary: $PASSED passed, $FAILED failed of ${#DATASETS[@]}"
echo "============================================================"
