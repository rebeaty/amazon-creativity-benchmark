#!/bin/bash
# =============================================================================
# HELM Pipeline Debugger - Orchestration Script
# Runs Claude Code in headless mode for each dataset in parallel
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MAX_PARALLEL_JOBS=5          # Number of concurrent Claude Code instances
MAX_TURNS=30                 # Max agentic turns per dataset (Claude Code flag)
DATASET_LIST="scenarios/subsampled_list.json"
LOG_DIR="debug_logs"
RESULTS_FILE="debug_results/summary.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Model to evaluate ───────────────────────────────────────────────────────
# Set the model you want HELM to evaluate. Examples:
#   EVAL_MODEL="claude-haiku-4-5-20251001"    # Anthropic API model
#   EVAL_MODEL="claude-sonnet-4-6"            # Anthropic API model
#   EVAL_MODEL="meta-llama/Llama-3-8B"        # HuggingFace model
#   EVAL_MODEL=""                              # Use whatever is already configured
EVAL_MODEL="${EVAL_MODEL:-claude-haiku-4-5-20251001}"

# ── Skip already-debugged datasets ──────────────────────────────────────────
# Pass --force to re-run ALL datasets regardless of prior results
FORCE_RERUN=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE_RERUN=true
    echo "Force mode: re-running ALL datasets (ignoring prior results)"
fi

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "debug_results"

# Extract dataset names from the JSON list
# Adjust the jq query based on your JSON structure (array of strings vs objects)
DATASETS=$(python3 -c "
import json
with open('$DATASET_LIST') as f:
    data = json.load(f)
# If it's a list of strings:
if isinstance(data, list) and isinstance(data[0], str):
    for d in data:
        print(d)
# If it's a list of dicts with a 'name' key:
elif isinstance(data, list) and isinstance(data[0], dict):
    for d in data:
        print(d.get('name', d.get('dataset', d.get('id', ''))))
else:
    print('ERROR: Unknown JSON structure', file=__import__('sys').stderr)
    exit(1)
")

TOTAL=$(echo "$DATASETS" | wc -l | tr -d ' ')
echo "============================================="
echo "HELM Pipeline Debugger"
echo "Datasets to debug: $TOTAL"
echo "Eval model: ${EVAL_MODEL:-<using existing config>}"
echo "Parallel jobs: $MAX_PARALLEL_JOBS"
echo "Max turns per dataset: $MAX_TURNS"
echo "Logs: $LOG_DIR/"
echo "Results: $RESULTS_FILE"
echo "============================================="
echo ""

# ── Per-dataset debugging function ───────────────────────────────────────────
debug_dataset() {
    local dataset="$1"
    local safe_name="${dataset//\//_}"
    local log_file="${LOG_DIR}/${safe_name}_${TIMESTAMP}.log"
    local passed_file="debug_results/${safe_name}_PASSED.json"
    local failed_file="debug_results/${safe_name}_FAILED.json"
    local status="UNKNOWN"
    local error_msg=""

    # ── Skip if already debugged ─────────────────────────────────────────
    if [[ "$FORCE_RERUN" == "false" ]]; then
        if [[ -f "$passed_file" ]]; then
            echo "[SKIP-PASSED] $dataset (already passed — use --force to re-run)"
            echo "{\"dataset\": \"$dataset\", \"status\": \"SKIPPED_PASSED\", \"log\": \"none\", \"timestamp\": \"$TIMESTAMP\"}" >> "$RESULTS_FILE"
            return 0
        fi
        if [[ -f "$failed_file" ]]; then
            echo "[SKIP-FAILED] $dataset (previously failed — use --force to re-run)"
            echo "{\"dataset\": \"$dataset\", \"status\": \"SKIPPED_FAILED\", \"log\": \"none\", \"timestamp\": \"$TIMESTAMP\"}" >> "$RESULTS_FILE"
            return 0
        fi
    else
        # Force mode: remove old results so Claude writes fresh ones
        rm -f "$passed_file" "$failed_file"
    fi

    echo "[START] $dataset"

    # ── Build the model instruction ──────────────────────────────────────
    local model_instruction=""
    if [[ -n "$EVAL_MODEL" ]]; then
        model_instruction="
MODEL TO EVALUATE: $EVAL_MODEL
- If this is an API model (e.g. claude-*, gpt-*, etc.), ensure the inference config
  in data/registry/registry_inference.yaml uses the API backend (not HuggingFace).
  Verify the API endpoint, model name, and that ANTHROPIC_API_KEY / OPENAI_API_KEY
  is available in the environment.
- If this is a HuggingFace model, ensure the config points to the correct model path
  and the model can be loaded locally.
"
    fi

    # The prompt sent to Claude Code in headless mode
    PROMPT=$(cat <<PROMPT_EOF
You are debugging the HELM evaluation pipeline for dataset: "$dataset"
$model_instruction
Follow the debugging protocol from CLAUDE.md exactly. Here is your task:

ATTEMPT LOOP (max 10 attempts):

For each attempt:
1. Check if evaluation data for "$dataset" is available locally. If not, download it using the appropriate method from eval_scripts/.
2. Find the scenario file in scenarios_new/ for "$dataset". Read the run spec from run_specs/.
3. Look up the paper link in data/registry/registry_master.yaml and verify the prompt template makes sense for this task.
4. Check the inference config in data/registry/registry_inference.yaml — verify temperature, max_tokens, and other settings are logical for this task type.
5. Run a small evaluation (e.g., 2-3 instances) using the evaluation scripts in eval_scripts/.
6. Check the metric in data/registry/registry_metrics.yaml is correct for this dataset.
7. Verify the evaluation metric runs without errors on the generated outputs.
8. Verify aggregation and result saving work correctly.

If ALL steps succeed: write a file debug_results/${dataset//\//_}_PASSED.json with:
{
  "dataset": "$dataset",
  "status": "PASSED",
  "attempts": <number>,
  "notes": "<any observations>"
}

If you hit an error, try to fix it (up to 10 attempts total). If you exhaust all attempts, write debug_results/${dataset//\//_}_FAILED.json with:
{
  "dataset": "$dataset",
  "status": "FAILED",
  "attempts": 10,
  "last_error": "<description of the last error>",
  "notes": "<what you tried>"
}

IMPORTANT: Only read the dataset list from scenarios/subsampled_list.json. Focus exclusively on "$dataset".
PROMPT_EOF
    )

    # Run Claude Code in headless mode
    if claude -p "$PROMPT" \
        --max-turns "$MAX_TURNS" \
        --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
        --output-format text \
        > "$log_file" 2>&1; then
        status="COMPLETED"
    else
        status="CLAUDE_ERROR"
        error_msg="Claude Code exited with non-zero status"
    fi

    # Check if a result file was created
    if [[ -f "$passed_file" ]]; then
        status="PASSED"
    elif [[ -f "$failed_file" ]]; then
        status="FAILED"
    fi

    # Append to summary
    echo "{\"dataset\": \"$dataset\", \"status\": \"$status\", \"log\": \"$log_file\", \"timestamp\": \"$TIMESTAMP\"}" >> "$RESULTS_FILE"
    echo "[${status}] $dataset → $log_file"
}

export -f debug_dataset
export LOG_DIR TIMESTAMP MAX_TURNS RESULTS_FILE FORCE_RERUN EVAL_MODEL

# ── Parallel Execution ───────────────────────────────────────────────────────
# Option A: GNU Parallel (preferred — install with: apt install parallel)
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel with $MAX_PARALLEL_JOBS jobs..."
    echo "$DATASETS" | parallel \
        --jobs "$MAX_PARALLEL_JOBS" \
        --bar \
        --timeout 1800 \
        --results "${LOG_DIR}/parallel_results/" \
        --joblog "${LOG_DIR}/joblog_${TIMESTAMP}.txt" \
        debug_dataset {}

# Option B: Bash background jobs with concurrency limit
else
    echo "GNU Parallel not found. Using bash background jobs..."
    CURRENT_JOBS=0
    while IFS= read -r dataset; do
        if (( CURRENT_JOBS >= MAX_PARALLEL_JOBS )); then
            wait -n
            ((CURRENT_JOBS--))
        fi
        debug_dataset "$dataset" &
        ((CURRENT_JOBS++))
    done <<< "$DATASETS"
    wait
fi

# ── Summary Report ───────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "DEBUGGING COMPLETE"
echo "============================================="

PASSED=$(grep -c '"PASSED"' "$RESULTS_FILE" 2>/dev/null || echo 0)
FAILED=$(grep -c '"FAILED"' "$RESULTS_FILE" 2>/dev/null || echo 0)
ERRORS=$(grep -c '"CLAUDE_ERROR"' "$RESULTS_FILE" 2>/dev/null || echo 0)
SKIPPED_P=$(grep -c '"SKIPPED_PASSED"' "$RESULTS_FILE" 2>/dev/null || echo 0)
SKIPPED_F=$(grep -c '"SKIPPED_FAILED"' "$RESULTS_FILE" 2>/dev/null || echo 0)
OTHER=$(grep -c '"COMPLETED"' "$RESULTS_FILE" 2>/dev/null || echo 0)

echo "PASSED:          $PASSED / $TOTAL"
echo "FAILED:          $FAILED / $TOTAL"
echo "ERRORS:          $ERRORS / $TOTAL"
echo "SKIPPED (pass):  $SKIPPED_P / $TOTAL"
echo "SKIPPED (fail):  $SKIPPED_F / $TOTAL"
echo "OTHER:           $OTHER / $TOTAL"
echo ""
echo "Full summary: $RESULTS_FILE"
echo "Individual logs: $LOG_DIR/"
echo ""

# List failed datasets for quick reference
if (( FAILED > 0 )); then
    echo "── Failed Datasets ────────────────────────"
    grep '"FAILED"' "$RESULTS_FILE" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"  ✗ {d['dataset']} → {d['log']}\")
"
fi