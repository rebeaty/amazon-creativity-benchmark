#!/usr/bin/env bash
# ============================================================================
# Run the sampled pair (brainteaser_sampled + cs4_sampled) across all 15
# existing Gemini/Gemma model suites at MAX_INSTANCES=200.
#
# Writes into each suite's existing dir alongside the prior 20-item runs
# (new dirs are named like "<unit>_subtask=<x>,model=<model>" and do not
# overwrite the onboarded benchmarks' output).
#
# Parallelism: CONCURRENCY suites at a time, each running its 2 eval
# scripts sequentially. Default 4 — adjust if you hit judge rate limits.
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source .env 2>/dev/null || true
source .venv/Scripts/activate 2>/dev/null || true
export PYTHONUTF8=1

MAX_INSTANCES="${MAX_INSTANCES:-200}"
CONCURRENCY="${CONCURRENCY:-4}"

# suite_name -> model_id  (order = completion order for readability)
declare -a PAIRS=(
    "gemini_lite:google/gemini-2.5-flash-lite"
    "gemini_flash:google/gemini-2.5-flash"
    "gemini_pro:google/gemini-2.5-pro"
    "gemini_2_flash:google/gemini-2.0-flash"
    "gemini_3_flash:google/gemini-3-flash-preview"
    "gemini_3_pro:google/gemini-3-pro-preview"
    "gemini_31_flash_lite:google/gemini-3.1-flash-lite-preview"
    "gemma_3_1b:google/gemma-3-1b-it"
    "gemma_3_4b:google/gemma-3-4b-it"
    "gemma_3_12b:google/gemma-3-12b-it"
    "gemma_3_27b:google/gemma-3-27b-it"
    "gemma_3n_e2b:google/gemma-3n-e2b-it"
    "gemma_3n_e4b:google/gemma-3n-e4b-it"
    "gemma_4_26b:google/gemma-4-26b-a4b-it"
    "gemma_4_31b:google/gemma-4-31b-it"
)

LOG_DIR="$SCRIPT_DIR/../benchmark_output/sampled_runs_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/_summary.tsv"
echo -e "suite\tunit\tstatus\telapsed_s" > "$SUMMARY"

model_safe_from() {
    echo "${1//\//_}"
}

stats_exists() {
    local suite="$1" unit_with_subtask="$2" model="$3"
    local model_safe
    model_safe="$(model_safe_from "$model")"
    # HELM writes "<unit>,model=<safe>" on Windows (: -> _)
    compgen -G "benchmark_output/runs/${suite}/${unit_with_subtask},model=${model_safe}/stats.json" > /dev/null 2>&1
}

run_one_pair() {
    local suite="$1" model="$2"
    local tag="${suite}-$(echo "$model" | sed 's@.*/@@')"
    local log="$LOG_DIR/${tag}.log"
    local start=$(date +%s)

    echo "[START] $tag" | tee -a "$LOG_DIR/_progress.log"

    # brainteaser_sampled
    if bash eval_scripts/brainteaser_sampled.sh "$model" "$suite" "$MAX_INSTANCES" >> "$log" 2>&1; then
        :
    fi
    for subtask in sentence_puzzle word_puzzle; do
        if stats_exists "$suite" "brainteaser_sampled_subtask=${subtask}" "$model"; then
            echo -e "${suite}\tbrainteaser_sampled_subtask=${subtask}\tPASS\t$(( $(date +%s) - start ))" >> "$SUMMARY"
        else
            echo -e "${suite}\tbrainteaser_sampled_subtask=${subtask}\tFAIL\t$(( $(date +%s) - start ))" >> "$SUMMARY"
        fi
    done

    # cs4_sampled
    if bash eval_scripts/cs4_sampled.sh "$model" "$suite" "$MAX_INSTANCES" >> "$log" 2>&1; then
        :
    fi
    for subtask in instruction story; do
        if stats_exists "$suite" "cs4_sampled_subtask=${subtask}" "$model"; then
            echo -e "${suite}\tcs4_sampled_subtask=${subtask}\tPASS\t$(( $(date +%s) - start ))" >> "$SUMMARY"
        else
            echo -e "${suite}\tcs4_sampled_subtask=${subtask}\tFAIL\t$(( $(date +%s) - start ))" >> "$SUMMARY"
        fi
    done

    echo "[DONE ] $tag ($(( $(date +%s) - start ))s)" | tee -a "$LOG_DIR/_progress.log"
}

export -f run_one_pair stats_exists model_safe_from
export LOG_DIR SUMMARY MAX_INSTANCES

# xargs with parallelism
printf "%s\n" "${PAIRS[@]}" | xargs -P "$CONCURRENCY" -I {} bash -c '
    pair="{}"
    suite="${pair%%:*}"
    model="${pair#*:}"
    run_one_pair "$suite" "$model"
'

echo ""
echo "=== SUMMARY ==="
cat "$SUMMARY" | column -t -s $'\t'
echo ""
echo "Per-suite logs in $LOG_DIR"
