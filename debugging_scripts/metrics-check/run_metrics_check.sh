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
MODEL="google/gemini-2.5-flash-lite"
SUITE="trial"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

METRICS_CHECK="$SCRIPT_DIR/metrics_check.py"
HELPER="$SCRIPT_DIR/_debug_helper.py"
INIT_EVAL="$SCRIPT_DIR/init_eval.sh"
PENDING_FILE="$SCRIPT_DIR/debug_assignments_pending.json"

# ── Stats.json existence check (reads only from benchmark_output/runs/$SUITE) ─
stats_json_exists() {
    local dataset="$1"
    local model_safe="${MODEL//\//_}"
    compgen -G "benchmark_output/runs/$SUITE/${dataset}:*model=${model_safe}/stats.json" > /dev/null 2>&1
}

# ── Run init_eval.sh and classify the outcome ───────────────────────────────
# Returns:
#   0 — init_eval ran and stats.json is present
#   2 — init_eval reported data-access skip (exit 2)
#   3 — init_eval finished but stats.json is still missing (silent failure)
run_init_eval() {
    local dataset="$1"
    local log_file="$2"
    bash "$INIT_EVAL" "$ASSIGNEE" "$dataset" >> "$log_file" 2>&1
    local rc=$?
    # init_eval.sh may have moved the dataset to done; keep it pending
    # until the metrics check itself passes.
    python3 "$HELPER" pending "$ASSIGNEE" "$dataset" >> "$log_file" 2>&1 || true
    if [[ $rc -eq 2 ]]; then
        return 2
    fi
    if ! stats_json_exists "$dataset"; then
        return 3
    fi
    return 0
}

# ── Hash the working-tree diff of dirs Claude may edit ──────────────────────
# Used to detect whether a Claude fix call actually modified any source files.
code_diff_hash() {
    git diff run_specs/ scenarios/ metrics/ eval_scripts/ 2>/dev/null | sha256sum | awk '{print $1}'
}

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

    # Pre-check: if no stats.json exists for this dataset, run the eval
    # first so the metrics-check workflow has something to compare against.
    if ! stats_json_exists "$dataset"; then
        echo "  [PRE-CHECK] No stats.json in benchmark_output/runs/$SUITE/ — running init_eval.sh first..."
        run_init_eval "$dataset" "$log_file"
        local pre_rc=$?
        case $pre_rc in
            0)
                echo "  [PRE-CHECK] init_eval.sh finished. Starting metrics-check workflow..."
                ;;
            2)
                echo "  [SKIP] init_eval.sh reported a data-access issue — cannot proceed"
                return 2
                ;;
            3)
                echo "  [SKIP] init_eval.sh finished but produced no stats.json — cannot proceed"
                return 2
                ;;
        esac
    fi

    local attempt=0
    while (( attempt < MAX_ATTEMPTS )); do
        (( attempt++ ))
        echo ""
        echo "  ── Attempt $attempt / $MAX_ATTEMPTS ──"

        # Step 1: Run the metrics check test. Keep stderr out of $check_output
        # so it stays pure JSON — stderr noise (yaml warnings, etc.) would
        # otherwise corrupt downstream `json.load` parses.
        echo "    [TEST] Running metrics check..."
        local check_output
        check_output=$(python3 "$METRICS_CHECK" "$dataset" 2>>"$log_file")
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
                run_init_eval "$dataset" "$log_file"
                local eval_rc=$?
                case $eval_rc in
                    0)
                        echo "    [EVAL] Eval finished. Continuing to next attempt..."
                        continue
                        ;;
                    2)
                        echo "    [SKIP] init_eval.sh reported a data-access issue — skipping"
                        return 2
                        ;;
                    3)
                        echo "    [SKIP] init_eval.sh finished but produced no stats.json — skipping"
                        return 2
                        ;;
                esac
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
            local before_hash after_hash
            before_hash=$(code_diff_hash)

            echo "    [FIX] Sending to Claude Code for diagnosis + fix..."
            local fix_output
            fix_output=$(ask_claude_to_fix "$dataset" "$missing" "$m1" "$m2" "$attempt")
            echo "=== Claude Fix (attempt $attempt) ===" >> "$log_file"
            echo "$fix_output" >> "$log_file"
            echo "" >> "$log_file"

            after_hash=$(code_diff_hash)
            if [[ "$before_hash" == "$after_hash" ]]; then
                echo "    [ABORT] Claude made no code changes — re-running eval would be pointless"
                break
            fi
            echo "    [FIX] Claude modified source files. Re-running eval..."

            # Step 3: Re-run the eval to regenerate stats.json.
            echo "    [EVAL] Running init_eval.sh..."
            run_init_eval "$dataset" "$log_file"
            local eval_rc=$?
            case $eval_rc in
                0)
                    echo "    [EVAL] Eval finished. Looping back to metrics check..."
                    ;;
                2)
                    echo "    [SKIP] init_eval.sh reported a data-access issue post-fix — skipping"
                    return 2
                    ;;
                3)
                    echo "    [SKIP] init_eval.sh finished but produced no stats.json post-fix — skipping"
                    return 2
                    ;;
            esac
        fi
    done

    # Exhausted all attempts. init_eval.sh may have moved the dataset to done
    # during a re-run; put it back in pending so state reflects the real outcome.
    python3 "$HELPER" pending "$ASSIGNEE" "$dataset" >> "$log_file" 2>&1 || true
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
