#!/usr/bin/env bash
# ============================================================================
# Evaluate: CLEF JOKER 2025 Task 2
# Dataset ID: clef_joker_2025_task2
# Input: text -> Output: text
# Paper: https://ceur-ws.org/Vol-3978/
# Repo:  (not available)

# ============================================================================
#
# Usage:
#   ./clef_joker_2025_task2.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./clef_joker_2025_task2.sh openai/gpt-4o
#   ./clef_joker_2025_task2.sh openai/gpt-4o my-suite
#   ./clef_joker_2025_task2.sh openai/gpt-4o my-suite 50
#
# Arguments:
#   MODEL          Required. The model to evaluate (e.g., openai/gpt-4o).
#   SUITE          Optional. Name for this evaluation run (default: creativity-benchmark).
#   MAX_INSTANCES  Optional. Limit the number of test instances (useful for quick tests).
# ============================================================================

set -euo pipefail

# ── Arguments ───────────────────────────────────────────────────────────────
MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}"
SUITE="${2:-creativity-benchmark}"
MAX_INSTANCES="${3:-}"

# ── Run entries ─────────────────────────────────────────────────────────────
RUN_ENTRY="clef_joker_2025_task2:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  CLEF JOKER 2025 Task 2"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "================================================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""

helm_run "${CMD[@]}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
