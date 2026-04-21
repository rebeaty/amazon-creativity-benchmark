#!/bin/bash
# =============================================================================
# Task B runner — 3 datasets with num_outputs > 8 that Google's Gemini API
# can't serve (candidate_count capped at 8). Runs them under a non-Gemini
# OpenRouter model instead.
#
# HARD RULE: MODEL must NOT be google/gemini-*. Gemini traffic always goes
# through the direct Google API, never OpenRouter. See Roger's feedback note
# in ~/.claude/.../memory/feedback_gemini_via_google_api.md.
#
# Usage:
#   ./run_taskB.sh [MODEL] [MAX_INSTANCES]
#
# Defaults:
#   MODEL         = openai/gpt-5-mini
#   MAX_INSTANCES = 10
#
# Examples:
#   ./run_taskB.sh
#   ./run_taskB.sh anthropic/claude-haiku-4.5
#   ./run_taskB.sh x-ai/grok-4.1-fast 20
# =============================================================================

set -uo pipefail

MODEL="${1:-openai/gpt-5-mini}"
MAX_INSTANCES="${2:-10}"
SUITE="taskB_10inst"

if [[ "$MODEL" == google/gemini-* ]]; then
    echo "ERROR: refusing to route google/gemini-* through OpenRouter." >&2
    echo "       Gemini goes via direct Google API only. See" >&2
    echo "       memory/feedback_gemini_via_google_api.md." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if command -v cygpath >/dev/null 2>&1; then
    SCRIPT_DIR_NATIVE="$(cygpath -w "$SCRIPT_DIR" | sed 's/\\/\//g')"
else
    SCRIPT_DIR_NATIVE="$SCRIPT_DIR"
fi

LOG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/debug_logs"
mkdir -p "$LOG_DIR"

# Read the flat list of Task B datasets (assignee-independent — they all run
# under the same model). tr -d '\r' strips Windows CRLF.
DATASETS=()
while IFS= read -r line; do
    DATASETS+=("$line")
done < <(python3 -c "
import json
with open('$SCRIPT_DIR_NATIVE/debug_assignments_taskB.json') as f:
    task_b = json.load(f)
seen = set()
for assignee, names in task_b.items():
    if assignee.startswith('_'): continue
    for n in names:
        if n not in seen:
            seen.add(n)
            print(n)
" | tr -d '\r')

echo "============================================================"
echo "  Task B runner"
echo "  Model:         $MODEL"
echo "  Suite:         $SUITE"
echo "  Max instances: $MAX_INSTANCES"
echo "  Datasets:      ${DATASETS[*]}"
echo "============================================================"

PASSED=0
FAILED=0
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "── $ds ──────────────────────────────────────────────────"
    log_file="$LOG_DIR/${ds}_taskB_$(date +%Y%m%d_%H%M%S).log"
    model_safe="${MODEL//\//_}"
    if compgen -G "benchmark_output/runs/$SUITE/${ds}*model=${model_safe}/stats.json" > /dev/null 2>&1; then
        echo "  [ALREADY DONE] stats.json exists"
        (( PASSED++ ))
        continue
    fi
    ./"eval_scripts/${ds}.sh" "$MODEL" "$SUITE" "$MAX_INSTANCES" > "$log_file" 2>&1
    rc=$?
    if compgen -G "benchmark_output/runs/$SUITE/${ds}*model=${model_safe}/stats.json" > /dev/null 2>&1; then
        echo "  [SUCCESS] stats.json written — see $log_file"
        (( PASSED++ ))
    else
        echo "  [FAILED] rc=$rc — see $log_file"
        (( FAILED++ ))
    fi
done

echo ""
echo "============================================================"
echo "  Task B summary: $PASSED passed, $FAILED failed of ${#DATASETS[@]}"
echo "============================================================"
