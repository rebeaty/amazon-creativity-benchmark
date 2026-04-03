#!/usr/bin/env bash
# ============================================================================
# Evaluate: Javanese-Sundanese Story Cloze
# Dataset ID: javanese_sundanese_story_cloze
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2502.12932
# Repo:  (not available)

# ============================================================================
#
# Usage:
#   ./javanese_sundanese_story_cloze.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./javanese_sundanese_story_cloze.sh openai/gpt-4o
#   ./javanese_sundanese_story_cloze.sh openai/gpt-4o my-suite
#   ./javanese_sundanese_story_cloze.sh openai/gpt-4o my-suite 50
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
RUN_ENTRY="javanese_sundanese_story_cloze:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  Javanese-Sundanese Story Cloze"
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
