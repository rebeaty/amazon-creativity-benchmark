#!/bin/bash
# =============================================================================
# HELM Dataset Debugger — Automated Claude Code Loop
#
# Replicates the manual debugging workflow:
#   1. Run eval script for a dataset
#   2. Check if stats.json was produced (success) or capture the error
#   3. If error: feed it to Claude Code to fix, then retry (up to 10 times)
#   4. If data-access error: skip the dataset
#   5. If stats.json exists: move dataset from pending to done
#   6. Move to next dataset
#
# Usage:
#   ./run_debug.sh <Assignee>                       # Run all pending datasets
#   ./run_debug.sh <Assignee> <dataset>             # Debug a single dataset
#   ./run_debug.sh <Assignee> --dry-run             # Show what would run
#
# Examples:
#   ./run_debug.sh Vijeta
#   ./run_debug.sh Vijeta funqa
#   ./run_debug.sh Clin --dry-run
# =============================================================================

set -uo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
MODEL="google/gemini-2.5-flash-lite"
SUITE="trial"
MAX_INSTANCES=10
MAX_ATTEMPTS=5                   # Max fix-and-retry cycles per dataset
EVAL_TIMEOUT=120                 # Seconds before killing a stuck eval run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


HELPER="$SCRIPT_DIR/_debug_helper.py"
INSIGHTS="$SCRIPT_DIR/debugging_w_vijeta.md"
LOG_DIR="$REPO_ROOT/debug_logs"
mkdir -p "$LOG_DIR"

# ── Args ────────────────────────────────────────────────────────────────────
ASSIGNEE="${1:?Usage: $0 <Assignee> [dataset|--dry-run]}"
SINGLE_DATASET="${2:-}"

if [[ "$SINGLE_DATASET" == "--dry-run" ]]; then
    echo "Dry run — datasets pending for $ASSIGNEE:"
    python3 "$HELPER" status "$ASSIGNEE"
    python3 -c "
import json
with open('$SCRIPT_DIR/debug_assignments_pending.json') as f:
    pending = json.load(f)
for d in pending.get('$ASSIGNEE', []):
    print(f'  - {d}')
"
    exit 0
fi

# ── Data-access error patterns (skip, don't retry) ─────────────────────────
DATA_ACCESS_PATTERNS=(
    "DatasetNotFoundError"
    "gated dataset"
    "Dataset file not found"
    "Please download from"
    "HTTP Error 401"
    "HTTP Error 403"
    "Repository Not Found"
    "FileNotFoundError.*download"
    "TIMEOUT: eval script exceeded"
)

is_data_access_error() {
    local error_text="$1"
    for pattern in "${DATA_ACCESS_PATTERNS[@]}"; do
        if echo "$error_text" | grep -qiE "$pattern"; then
            return 0
        fi
    done
    return 1
}

# ── Stats.json check ───────────────────────────────────────────────────────
stats_json_exists() {
    local dataset="$1"
    # HELM output dir uses underscores for slashes in model name
    local model_safe="${MODEL//\//_}"
    # Run dirs may include extra params (e.g. subtask=...) between dataset name and model=
    compgen -G "benchmark_output/runs/$SUITE/${dataset}:*model=${model_safe}/stats.json" > /dev/null 2>&1
}

# ── Run one eval ────────────────────────────────────────────────────────────
run_eval() {
    local dataset="$1"
    local eval_script="eval_scripts/${dataset}.sh"

    if [[ ! -f "$eval_script" ]]; then
        echo "    [ERROR] No eval script found: $eval_script"
        return 1
    fi

    # Run and capture stderr+stdout; kill if it exceeds EVAL_TIMEOUT
    local output rc=0
    output=$(timeout "$EVAL_TIMEOUT" ./"$eval_script" "$MODEL" "$SUITE" "$MAX_INSTANCES" 2>&1) || rc=$?
    if [[ $rc -eq 124 ]]; then
        output="TIMEOUT: eval script exceeded ${EVAL_TIMEOUT}s — treating as data access error"
    fi
    echo "$output"
}

# ── Extract the meaningful error from HELM output ───────────────────────────
extract_error() {
    local output="$1"
    # Grab from the first "Traceback" or "Error" to the end
    echo "$output" | awk '
        /Traceback \(most recent call last\)/ { found=1 }
        /Error when running/ { found=1 }
        found { print }
    ' | tail -80
}

# ── Ask Claude Code to fix the error ────────────────────────────────────────
ask_claude_to_fix() {
    local dataset="$1"
    local error_text="$2"
    local attempt="$3"

    local prompt
    prompt=$(cat <<PROMPT_EOF
You are debugging the HELM evaluation pipeline for dataset: "$dataset"

CONTEXT: Read the debugging insights file at debugging_scripts/debugging_w_vijeta.md
for common patterns and fixes from prior sessions. Apply known fixes directly
without re-discovering them.

The eval command was:
  ./eval_scripts/${dataset}.sh "$MODEL" "$SUITE" $MAX_INSTANCES

It produced this error (attempt $attempt of $MAX_ATTEMPTS):

$error_text

Fix the error. Only modify files in scenarios_new/, run_specs/, metrics/, or
eval_scripts/ — do NOT modify HELM's installed package files.

Rules:
- If this is a scenario bug (wrong type, missing import, bad data handling), fix it.
- If this is a run_spec bug (wrong metric class, missing args), fix it.
- If you need to check something, read the relevant files first.
- Be surgical — only fix what's broken for this dataset.
- Do NOT explain what you will do. Just read the relevant files and make the fix.
PROMPT_EOF
    )

    claude -p "$prompt" \
        --max-turns 10 \
        --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
        --output-format text \
        2>&1
}

# ── Debug one dataset ───────────────────────────────────────────────────────
debug_one() {
    local dataset="$1"
    local log_file="$LOG_DIR/${dataset}_$(date +%Y%m%d_%H%M%S).log"
    local start_time
    start_time=$(date +%s)

    echo ""
    echo "============================================================"
    echo "  Dataset:  $dataset"
    echo "  Model:    $MODEL"
    echo "  Suite:    $SUITE"
    echo "  Log:      $log_file"
    echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    # Check if already done (stats.json exists from a prior run)
    if stats_json_exists "$dataset"; then
        echo "  [ALREADY DONE] stats.json exists — moving to done"
        python3 "$HELPER" done "$ASSIGNEE" "$dataset"
        return 0
    fi

    local attempt=0
    while (( attempt < MAX_ATTEMPTS )); do
        (( attempt++ ))
        echo ""
        echo "  ── Attempt $attempt / $MAX_ATTEMPTS ──"

        # Step 1: Run the eval
        echo "    [RUN] ./eval_scripts/${dataset}.sh $MODEL $SUITE $MAX_INSTANCES"
        local output
        output=$(run_eval "$dataset")

        # Log full output
        echo "=== Attempt $attempt ===" >> "$log_file"
        echo "$output" >> "$log_file"
        echo "" >> "$log_file"

        # Step 2: Check success
        if stats_json_exists "$dataset"; then
            local elapsed=$(( $(date +%s) - start_time ))
            echo "    [SUCCESS] stats.json written (${elapsed}s, $attempt attempt(s))"
            python3 "$HELPER" done "$ASSIGNEE" "$dataset"
            echo "    [MOVED] $dataset -> done"
            return 0
        fi

        # Step 3: Extract error
        local error_text
        error_text=$(extract_error "$output")

        if [[ -z "$error_text" ]]; then
            echo "    [WARN] No stats.json but no clear error found in output"
            echo "    Last 10 lines of output:"
            echo "$output" | tail -10 | sed 's/^/      /'
            # Still try to get Claude to look at it
            error_text=$(echo "$output" | tail -40)
        fi

        # Check if data-access error → skip
        if is_data_access_error "$error_text"; then
            echo "    [SKIP] Data access issue detected — skipping dataset"
            echo "    Error snippet:"
            echo "$error_text" | head -5 | sed 's/^/      /'
            echo ""
            echo "    Reason logged to: $log_file"
            return 2
        fi

        # Step 3b: Ask Claude to fix
        if (( attempt < MAX_ATTEMPTS )); then
            echo "    [FIX] Sending error to Claude Code..."
            local fix_output
            fix_output=$(ask_claude_to_fix "$dataset" "$error_text" "$attempt")
            echo "$fix_output" >> "$log_file"
            echo "    [FIX] Claude Code responded. Retrying..."
        fi
    done

    # Exhausted all attempts
    echo "    [FAILED] $dataset — exhausted $MAX_ATTEMPTS attempts"
    echo "    Last error in: $log_file"
    return 1
}

# ── Main loop ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  HELM Dataset Debugger"
echo "  Assignee: $ASSIGNEE"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
echo "============================================================"
python3 "$HELPER" status "$ASSIGNEE"

# Build dataset list
if [[ -n "$SINGLE_DATASET" ]]; then
    DATASETS=("$SINGLE_DATASET")
else
    mapfile -t DATASETS < <(python3 -c "
import json
with open('$SCRIPT_DIR/debug_assignments_pending.json') as f:
    pending = json.load(f)
for d in pending.get('$ASSIGNEE', []):
    print(d)
")
fi

TOTAL=${#DATASETS[@]}
PASSED=0
FAILED=0
SKIPPED=0

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    echo ""
    echo "[$((i+1))/$TOTAL] ─────────────────────────────────────────"

    debug_one "$dataset"
    result=$?

    case $result in
        0) (( PASSED++ )) ;;
        1) (( FAILED++ )) ;;
        2) (( SKIPPED++ )) ;;
    esac
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "  Total:   $TOTAL"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED (data access issues)"
echo ""
python3 "$HELPER" status "$ASSIGNEE"
echo "============================================================"
