#!/bin/bash
[ -z "$BASH_VERSION" ] && exec bash "$0" "$@"
# =============================================================================
# Metrics Check Orchestrator
#
# For each assigned dataset, checks whether all expected metrics (from
# registry_metrics.yaml) are present in the trial run stats.json. If not,
# uses Claude Code to diagnose and fix the run_spec/scenario, re-runs the
# eval, and retries — up to MAX_ATTEMPTS times per dataset.
#
# Usage:
#   ./run_metrics_check.sh <Assignee>                    # Run all pending
#   ./run_metrics_check.sh <Assignee> <dataset>          # Single dataset
#   ./run_metrics_check.sh <Assignee> --dry-run          # Show what would run
#
# Examples:
#   ./run_metrics_check.sh vijeta
#   ./run_metrics_check.sh vijeta dat
#   ./run_metrics_check.sh clin --dry-run
# =============================================================================

set -uo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
MAX_ATTEMPTS=10
EVAL_TIMEOUT=180                  # Seconds for eval timeout (passed to init_eval.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

METRICS_CHECK="$SCRIPT_DIR/metrics_check.py"
HELPER="$SCRIPT_DIR/_debug_helper.py"
INIT_EVAL="$SCRIPT_DIR/init_eval.sh"
PENDING_FILE="$SCRIPT_DIR/debug_assignments_pending.json"

# ── Args ────────────────────────────────────────────────────────────────────
ASSIGNEE="${1:?Usage: $0 <Assignee> [dataset|--dry-run]}"
SINGLE_DATASET="${2:-}"

if [[ "$SINGLE_DATASET" == "--dry-run" ]]; then
    echo "Dry run — datasets pending for $ASSIGNEE:"
    python3 "$HELPER" status "$ASSIGNEE"
    python3 -c "
import json
with open('$PENDING_FILE') as f:
    pending = json.load(f)
for d in pending.get('$ASSIGNEE', []):
    print(f'  - {d}')
"
    exit 0
fi

# ── Per-assignee output directory ──────────────────────────────────────────
ASSIGNEE_DIR="$SCRIPT_DIR/$ASSIGNEE"
mkdir -p "$ASSIGNEE_DIR"
LOG_DIR="$ASSIGNEE_DIR"

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
(attempt $attempt of $MAX_ATTEMPTS)

Use the metrics-diagnose-fix skill workflow.

## Metrics Check Result
- Expected metrics (m1): $m1_json
- Actual metrics (m2):   $m2_json
- Missing metrics:       $missing_json

## Instructions

1. READ the diagnosis instructions from the metrics-diagnose-fix skill:
   - Read data/registry/registry_metrics.yaml for "$dataset" to understand each missing metric's type, helm_class, and config
   - Read run_specs/${dataset}_run_specs.py to see current MetricSpecs
   - Read the scenario file (check scenarios/ directory)

2. WRITE a diagnosis file to: debugging_scripts/metrics-check/$ASSIGNEE/${dataset}_diagnosis.md
   Include: expected vs actual metrics, root cause analysis, proposed fix.

3. FIX the code:
   - Edit run_specs/${dataset}_run_specs.py to add/correct MetricSpecs for the missing metrics
   - If needed, add AnnotatorSpecs for LLM-judge metrics
   - If needed, fix scenario file issues
   - Verify any HELM class you reference actually exists: python3 -c "from X import Y"

Rules:
- Only modify files in run_specs/, scenarios/, metrics/, eval_scripts/
- Do NOT modify HELM's installed package files
- Be surgical - only fix what's broken for missing metrics
- Update/write the summary file of fixes you made and why, to debugging_scripts/metrics-check/$ASSIGNEE/${dataset}_fixes.md
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
    echo "  Log:      $log_file"
    echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    local attempt=0
    while (( attempt < MAX_ATTEMPTS )); do
        (( attempt++ ))
        echo ""
        echo "  ── Attempt $attempt / $MAX_ATTEMPTS ──"

        # Step 1: Run the metrics check test
        echo "    [TEST] Running metrics check..."
        local check_output
        check_output=$(python3 "$METRICS_CHECK" "$dataset" 2>&1)
        local check_rc=$?

        echo "    $check_output" >> "$log_file"

        if [[ $check_rc -eq 0 ]]; then
            # All metrics present — PASS
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
            python3 "$HELPER" done "$ASSIGNEE" "$dataset"
            echo "    [MOVED] $dataset -> done"
            return 0
        fi

        if [[ $check_rc -eq 2 ]]; then
            # No registry entry or no stats.json
            local status
            status=$(echo "$check_output" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
            if [[ "$status" == "no_registry" ]]; then
                echo "    [SKIP] Dataset not found in registry_metrics.yaml"
                return 2
            elif [[ "$status" == "no_stats" ]]; then
                echo "    [NO STATS] No stats.json found — need to run eval first"
                echo "    [EVAL] Running init_eval.sh..."
                bash "$INIT_EVAL" "$ASSIGNEE" "$dataset" >> "$log_file" 2>&1
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

        # Step 2: Ask Claude to diagnose and fix
        if (( attempt < MAX_ATTEMPTS )); then
            echo "    [FIX] Sending to Claude Code for diagnosis + fix..."
            local fix_output
            fix_output=$(ask_claude_to_fix "$dataset" "$missing" "$m1" "$m2" "$attempt")
            echo "=== Claude Fix (attempt $attempt) ===" >> "$log_file"
            echo "$fix_output" >> "$log_file"
            echo "" >> "$log_file"
            echo "    [FIX] Claude responded. Re-running eval..."

            # Step 3: Re-run the eval to regenerate stats.json
            echo "    [EVAL] Running init_eval.sh..."
            bash "$INIT_EVAL" "$ASSIGNEE" "$dataset" >> "$log_file" 2>&1
            echo "    [EVAL] Eval finished. Looping back to metrics check..."
        fi
    done

    # Exhausted all attempts
    echo "    [FAILED] $dataset — exhausted $MAX_ATTEMPTS attempts"
    echo "    Last log: $log_file"
    return 1
}

# ── Main ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Metrics Check Orchestrator"
echo "  Assignee: $ASSIGNEE"
echo "============================================================"
python3 "$HELPER" status "$ASSIGNEE"

# Build dataset list
if [[ -n "$SINGLE_DATASET" ]]; then
    DATASETS=("$SINGLE_DATASET")
else
    DATASETS=()
    while IFS= read -r d; do
        DATASETS+=("$d")
    done < <(python3 -c "
import json
with open('$PENDING_FILE') as f:
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
python3 "$HELPER" status "$ASSIGNEE"
echo "============================================================"
