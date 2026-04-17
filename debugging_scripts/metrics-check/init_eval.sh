#!/bin/bash
# =============================================================================
# HELM Dataset Debugger — Single Dataset Eval + Auto-Fix
#
# Runs the eval script for one dataset, and if it fails, feeds the error to
# Claude Code for a fix-and-retry loop (up to MAX_ATTEMPTS times).
#
# Usage:
#   ./init_eval.sh <Assignee> <dataset>
#
# Examples:
#   ./init_eval.sh vijeta funqa
#   ./init_eval.sh clin macgyver
# =============================================================================

set -uo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
MODEL="google/gemini-2.5-flash-lite"
SUITE="trial"
MAX_INSTANCES=10
MAX_ATTEMPTS=5                   # Max fix-and-retry cycles per dataset
EVAL_TIMEOUT=120                 # Seconds before killing a stuck eval run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

HELPER="$SCRIPT_DIR/_debug_helper.py"

# ── Args ────────────────────────────────────────────────────────────────────
ASSIGNEE="${1:?Usage: $0 <Assignee> <dataset>}"
DATASET="${2:?Usage: $0 <Assignee> <dataset>}"

# ── Per-assignee output directory ──────────────────────────────────────────
ASSIGNEE_DIR="$SCRIPT_DIR/$ASSIGNEE"
mkdir -p "$ASSIGNEE_DIR"
LOG_DIR="$ASSIGNEE_DIR"
LEARNINGS_FILE="$ASSIGNEE_DIR/eval_run_learnings.md"

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
    local model_safe="${MODEL//\//_}"
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

    local output rc=0
    local timeout_cmd
    timeout_cmd=$(command -v timeout 2>/dev/null || command -v gtimeout 2>/dev/null || true)
    if [[ -n "$timeout_cmd" ]]; then
        output=$("$timeout_cmd" "$EVAL_TIMEOUT" ./"$eval_script" "$MODEL" "$SUITE" "$MAX_INSTANCES" 2>&1) || rc=$?
        if [[ $rc -eq 124 ]]; then
            output="TIMEOUT: eval script exceeded ${EVAL_TIMEOUT}s — treating as data access error"
        fi
    else
        output=$(./"$eval_script" "$MODEL" "$SUITE" "$MAX_INSTANCES" 2>&1) || rc=$?
    fi
    echo "$output"
}

# ── Extract the meaningful error from HELM output ───────────────────────────
extract_error() {
    local output="$1"
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

The eval command was:
  ./eval_scripts/${dataset}.sh "$MODEL" "$SUITE" $MAX_INSTANCES

It produced this error (attempt $attempt of $MAX_ATTEMPTS):

$error_text

Fix the error. Only modify files in scenarios/, run_specs/, metrics/, or
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

# ── Write learnings to eval_run_learnings.md ────────────────────────────────
write_learnings() {
    local dataset="$1"
    local outcome="$2"      # SUCCESS, FAILED, or SKIPPED
    local attempts="$3"
    local log_file="$4"

    local log_tail
    log_tail=$(tail -200 "$log_file" 2>/dev/null || echo "(no log available)")

    local prompt
    prompt=$(cat <<LEARN_EOF
You are summarizing the debugging session for HELM dataset: "$dataset"
Outcome: $outcome after $attempts attempt(s).

Below is the tail of the debug log:

$log_tail

Your task: append a concise entry to the learnings file at:
  $LEARNINGS_FILE

Rules:
- If the file already exists, READ it first, then APPEND a new section at the end.
- If the file does not exist, create it with a top-level heading.
- Use this format for the new entry:

## <dataset> — <outcome>
- **Attempts**: <N>
- **Root cause**: one-line summary of what went wrong (or "N/A" if first-attempt success)
- **Fix applied**: one-line summary of what was changed (or "N/A")
- **Key learning**: what a future debugger should know about this dataset or error pattern

- Be concise. Each bullet should be one line.
- Do NOT remove or rewrite existing entries in the file.
LEARN_EOF
    )

    claude -p "$prompt" \
        --max-turns 5 \
        --allowedTools "Read,Edit,Write" \
        --output-format text \
        2>&1
}

# ── Debug the dataset ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  HELM Dataset Debugger (single dataset)"
echo "  Assignee: $ASSIGNEE"
echo "  Dataset:  $DATASET"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
echo "============================================================"
python3 "$HELPER" status "$ASSIGNEE"

log_file="$LOG_DIR/${DATASET}_$(date +%Y%m%d_%H%M%S).log"
start_time=$(date +%s)

echo ""
echo "============================================================"
echo "  Dataset:  $DATASET"
echo "  Log:      $log_file"
echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Check if already done (stats.json exists from a prior run)
if stats_json_exists "$DATASET"; then
    echo "  [ALREADY DONE] stats.json exists — moving to done"
    python3 "$HELPER" done "$ASSIGNEE" "$DATASET"
    echo "    [LEARN] Recording learnings..."
    write_learnings "$DATASET" "SUCCESS (already done)" "0" "$log_file" > /dev/null 2>&1
    exit 0
fi

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
    (( attempt++ ))
    echo ""
    echo "  ── Attempt $attempt / $MAX_ATTEMPTS ──"

    # Step 1: Run the eval
    echo "    [RUN] ./eval_scripts/${DATASET}.sh $MODEL $SUITE $MAX_INSTANCES"
    output=$(run_eval "$DATASET")

    # Log full output
    echo "=== Attempt $attempt ===" >> "$log_file"
    echo "$output" >> "$log_file"
    echo "" >> "$log_file"

    # Step 2: Check success
    if stats_json_exists "$DATASET"; then
        elapsed=$(( $(date +%s) - start_time ))
        echo "    [SUCCESS] stats.json written (${elapsed}s, $attempt attempt(s))"
        python3 "$HELPER" done "$ASSIGNEE" "$DATASET"
        echo "    [MOVED] $DATASET -> done"
        echo "    [LEARN] Recording learnings..."
        write_learnings "$DATASET" "SUCCESS" "$attempt" "$log_file" > /dev/null 2>&1
        exit 0
    fi

    # Step 3: Extract error
    error_text=$(extract_error "$output")

    if [[ -z "$error_text" ]]; then
        echo "    [WARN] No stats.json but no clear error found in output"
        echo "    Last 10 lines of output:"
        echo "$output" | tail -10 | sed 's/^/      /'
        error_text=$(echo "$output" | tail -40)
    fi

    # Check if data-access error → skip
    if is_data_access_error "$error_text"; then
        echo "    [SKIP] Data access issue detected — skipping dataset"
        echo "    Error snippet:"
        echo "$error_text" | head -5 | sed 's/^/      /'
        echo ""
        echo "    Reason logged to: $log_file"
        echo "    [LEARN] Recording learnings..."
        write_learnings "$DATASET" "SKIPPED (data access)" "$attempt" "$log_file" > /dev/null 2>&1
        exit 2
    fi

    # Step 3b: Ask Claude to fix
    if (( attempt < MAX_ATTEMPTS )); then
        echo "    [FIX] Sending error to Claude Code..."
        fix_output=$(ask_claude_to_fix "$DATASET" "$error_text" "$attempt")
        echo "$fix_output" >> "$log_file"
        echo "    [FIX] Claude Code responded. Retrying..."
    fi
done

# Exhausted all attempts
echo "    [FAILED] $DATASET — exhausted $MAX_ATTEMPTS attempts"
echo "    Last error in: $log_file"
echo "    [LEARN] Recording learnings..."
write_learnings "$DATASET" "FAILED" "$MAX_ATTEMPTS" "$log_file" > /dev/null 2>&1
exit 1
