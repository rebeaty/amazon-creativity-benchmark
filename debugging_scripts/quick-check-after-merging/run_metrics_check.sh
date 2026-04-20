#!/bin/bash
[ -z "$BASH_VERSION" ] && exec bash "$0" "$@"
# =============================================================================
# Metrics Check Orchestrator — quick-check-after-merging
#
# Re-runs trial evals and metrics checks for every dataset listed in
# list_good.json (the merged-and-sorted "done" list from the per-assignee
# metrics-check state). Uses SUITE=trial_after_merging_April_20 so nothing
# here touches the original trial run outputs.
#
# Per dataset:
#   1. If no stats.json exists under the new suite, run init_eval.sh to
#      produce one (with Claude-driven fix-and-retry on eval errors).
#   2. Run metrics_check.py to compare expected vs actual metrics.
#   3. If metrics are missing, ask Claude Code to diagnose and fix the
#      run_spec / scenario, re-run the eval, and retry — up to MAX_ATTEMPTS.
#
# Results (passed/failed lists) are persisted to this directory so the
# script can be stopped and resumed without redoing completed datasets.
#
# Usage:
#   ./run_metrics_check.sh                     # Run every dataset in list_good.json
#   ./run_metrics_check.sh <dataset>           # Run a single dataset
#   ./run_metrics_check.sh --dry-run           # Show what would run
# =============================================================================

set -uo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
SUITE="trial_after_merging_April_20"
MODEL="google/gemini-2.5-flash-lite"
MAX_ATTEMPTS=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

METRICS_CHECK="$SCRIPT_DIR/metrics_check.py"
INIT_EVAL="$SCRIPT_DIR/init_eval.sh"
INPUT_LIST="$SCRIPT_DIR/list_metric_match.json"
PASSED_FILE="$SCRIPT_DIR/passed.json"
FAILED_FILE="$SCRIPT_DIR/failed.json"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Args ────────────────────────────────────────────────────────────────────
ARG="${1:-}"

# ── Helpers for passed/failed persistence ───────────────────────────────────
init_state_file() {
    local f="$1"
    [[ -f "$f" ]] || echo "[]" > "$f"
}
init_state_file "$PASSED_FILE"
init_state_file "$FAILED_FILE"

is_in_state() {
    local f="$1" ds="$2"
    python3 -c "
import json, sys
with open('$f') as fh:
    data = json.load(fh)
sys.exit(0 if '$ds' in data else 1)
"
}

add_to_state() {
    local f="$1" ds="$2"
    python3 -c "
import json
with open('$f') as fh:
    data = json.load(fh)
if '$ds' not in data:
    data.append('$ds')
    data.sort()
with open('$f', 'w') as fh:
    json.dump(data, fh, indent=2)
    fh.write('\n')
"
}

remove_from_state() {
    local f="$1" ds="$2"
    python3 -c "
import json
with open('$f') as fh:
    data = json.load(fh)
data = [d for d in data if d != '$ds']
with open('$f', 'w') as fh:
    json.dump(data, fh, indent=2)
    fh.write('\n')
"
}

# ── Ask Claude Code to diagnose and fix missing metrics ────────────────────
ask_claude_to_fix() {
    local dataset="$1"
    local missing_json="$2"
    local m1_json="$3"
    local m2_json="$4"
    local attempt="$5"

    local tmpfile prompt
    tmpfile=$(mktemp /tmp/metrics_prompt.XXXXXX)
    cat > "$tmpfile" <<PROMPT_EOF
You are fixing missing metrics for HELM dataset: "$dataset"
(attempt $attempt of $MAX_ATTEMPTS, suite=$SUITE)

Use the metrics-diagnose-fix skill workflow.

## Metrics Check Result
- Expected metrics (m1): $m1_json
- Actual metrics (m2):   $m2_json
- Missing metrics:       $missing_json

## Instructions

1. READ the diagnosis inputs:
   - Read data/registry/registry_metrics.yaml for "$dataset" to understand each missing metric's type, helm_class, and config
   - Read run_specs/${dataset}_run_specs.py to see current MetricSpecs
   - Read the scenario file (check scenarios/ directory)

2. WRITE a diagnosis file to:
     debugging_scripts/quick-check-after-merging/diagnoses/${dataset}_diagnosis.md
   Include: expected vs actual metrics, root cause analysis, proposed fix.

3. FIX the code:
   - Edit run_specs/${dataset}_run_specs.py to add/correct MetricSpecs for the missing metrics
   - If needed, add AnnotatorSpecs for LLM-judge metrics
   - If needed, fix scenario file issues
   - Verify any HELM class you reference actually exists: python3 -c "from X import Y"

Rules:
- Only modify files in run_specs/, scenarios/, metrics/, eval_scripts/
- Do NOT modify HELM's installed package files
- Be surgical — only fix what's broken for the missing metrics
- Write/update a summary of your fixes and why at:
    debugging_scripts/quick-check-after-merging/fixes/${dataset}_fixes.md
PROMPT_EOF
    prompt=$(cat "$tmpfile")
    rm -f "$tmpfile"

    claude -p "$prompt" \
        --max-turns 15 \
        --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
        --output-format text \
        2>&1
}

# ── Process one dataset ────────────────────────────────────────────────────
process_one() {
    local dataset="$1"
    local log_file="$LOG_DIR/${dataset}_metrics_$(date +%Y%m%d_%H%M%S).log"
    local start_time
    start_time=$(date +%s)

    echo ""
    echo "============================================================"
    echo "  Dataset:  $dataset"
    echo "  Suite:    $SUITE"
    echo "  Log:      $log_file"
    echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    # Skip if already passed in a prior run
    if is_in_state "$PASSED_FILE" "$dataset"; then
        echo "    [SKIP] Already in passed.json — nothing to do"
        return 0
    fi
    # If it was previously marked failed, drop that entry; we're retrying.
    remove_from_state "$FAILED_FILE" "$dataset"

    local attempt=0
    while (( attempt < MAX_ATTEMPTS )); do
        (( attempt++ ))
        echo ""
        echo "  ── Attempt $attempt / $MAX_ATTEMPTS ──"

        # Step 1: Run the metrics check
        echo "    [TEST] Running metrics check (suite=$SUITE)..."
        local check_output
        check_output=$(python3 "$METRICS_CHECK" "$dataset" --suite "$SUITE" --model "$MODEL" 2>&1)
        local check_rc=$?

        echo "    $check_output" >> "$log_file"

        if [[ $check_rc -eq 0 ]]; then
            local elapsed=$(( $(date +%s) - start_time ))
            echo "    [PASS] All metrics present (${elapsed}s, $attempt attempt(s))"
            echo "    [METRICS] Expected vs present in stats.json:"
            echo "$check_output" | python3 -c "
import json, sys
r = json.load(sys.stdin)
m1, m2 = r.get('m1', []), r.get('m2', [])
m2_set = set(m2)
for m in m1:
    mark = 'x' if m in m2_set else ' '
    print(f'      [{mark}] {m}  (expected)')
for m in m2:
    if m not in set(m1):
        print(f'      [x] {m}  (extra, not in m1)')
"
            add_to_state "$PASSED_FILE" "$dataset"
            echo "    [RECORDED] $dataset -> passed.json"
            return 0
        fi

        if [[ $check_rc -eq 2 ]]; then
            local status
            status=$(echo "$check_output" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
            if [[ "$status" == "no_registry" ]]; then
                echo "    [SKIP] Dataset not found in registry_metrics.yaml"
                add_to_state "$FAILED_FILE" "$dataset"
                return 2
            elif [[ "$status" == "no_stats" ]]; then
                echo "    [NO STATS] No stats.json found under suite=$SUITE — running eval"
                bash "$INIT_EVAL" "$dataset" >> "$log_file" 2>&1
                echo "    [EVAL] Eval finished. Continuing to next attempt..."
                continue
            fi
        fi

        # check_rc == 1: metrics are missing
        local missing m1 m2
        missing=$(echo "$check_output" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['missing']))" 2>/dev/null)
        m1=$(echo "$check_output" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['m1']))" 2>/dev/null)
        m2=$(echo "$check_output" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['m2']))" 2>/dev/null)

        echo "    [FAIL] Missing metrics: $missing"

        if (( attempt < MAX_ATTEMPTS )); then
            echo "    [FIX] Sending to Claude Code for diagnosis + fix..."
            local fix_output
            fix_output=$(ask_claude_to_fix "$dataset" "$missing" "$m1" "$m2" "$attempt")
            echo "=== Claude Fix (attempt $attempt) ===" >> "$log_file"
            echo "$fix_output" >> "$log_file"
            echo "" >> "$log_file"
            echo "    [FIX] Claude responded. Re-running eval..."

            bash "$INIT_EVAL" "$dataset" >> "$log_file" 2>&1
            echo "    [EVAL] Eval finished. Looping back to metrics check..."
        fi
    done

    add_to_state "$FAILED_FILE" "$dataset"
    echo "    [FAILED] $dataset — exhausted $MAX_ATTEMPTS attempts"
    echo "    Last log: $log_file"
    return 1
}

# ── Build dataset list ─────────────────────────────────────────────────────
if [[ ! -f "$INPUT_LIST" ]]; then
    echo "ERROR: $INPUT_LIST not found. Generate it first."
    exit 1
fi

if [[ "$ARG" == "--dry-run" ]]; then
    echo "Dry run — datasets in list_good.json:"
    python3 -c "
import json
with open('$INPUT_LIST') as f:
    for d in json.load(f):
        print(f'  - {d}')
"
    exit 0
fi

if [[ -n "$ARG" ]]; then
    DATASETS=("$ARG")
else
    DATASETS=()
    while IFS= read -r d; do
        DATASETS+=("$d")
    done < <(python3 -c "
import json
with open('$INPUT_LIST') as f:
    for d in json.load(f):
        print(d)
")
fi

# ── Main ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Metrics Check Orchestrator (quick-check-after-merging)"
echo "  Suite:    $SUITE"
echo "  Input:    $INPUT_LIST (${#DATASETS[@]} datasets)"
echo "  Model:    $MODEL"
echo "============================================================"

TOTAL=${#DATASETS[@]}
PASSED=0
FAILED=0
SKIPPED=0

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    echo ""
    echo "[$((i+1))/$TOTAL] ─────────────────────────────────────────"

    process_one "$dataset"
    result=$?

    case $result in
        0) (( PASSED++ )) ;;
        1) (( FAILED++ )) ;;
        2) (( SKIPPED++ )) ;;
    esac
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "  Total:   $TOTAL"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED (no registry entry)"
echo ""
echo "  passed.json: $PASSED_FILE"
echo "  failed.json: $FAILED_FILE"
echo "============================================================"
