#!/usr/bin/env bash
# ============================================================================
# Evaluate: CAP (Creativity Assessment Platform) — 5 tasks
# Dataset ID: cap
# Input: text -> Output: text
# Source: human/uva_pilot (UVA Study 3 validation)
#   Tasks: AUT, SCTT, Design, Metaphor, Story
# ============================================================================
#
# Usage:
#   ./cap.sh MODEL [SUITE] [MAX_INSTANCES]
# ============================================================================

set -euo pipefail

MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}"
SUITE="${2:-creativity-benchmark}"
MAX_INSTANCES="${3:-}"

RUN_ENTRIES=()
RUN_ENTRIES+=("cap_aut:model=${MODEL}")
RUN_ENTRIES+=("cap_sctt:model=${MODEL}")
RUN_ENTRIES+=("cap_design:model=${MODEL}")
RUN_ENTRIES+=("cap_metaphor:model=${MODEL}")
RUN_ENTRIES+=("cap_story:model=${MODEL}")

source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "${RUN_ENTRIES[@]}" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  CAP (AUT + SCTT + Design + Metaphor + Story)"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "================================================================"
echo ""

helm_run "${CMD[@]}"

echo ""
echo "Summarizing results..."
helm-summarize --suite "$SUITE"
echo ""
echo "Done. Per-task results in: benchmark_output/runs/$SUITE/cap_*_model=*/"
echo "Novelty metric is computed post-hoc — see scripts/score_cap_novelty.py"
