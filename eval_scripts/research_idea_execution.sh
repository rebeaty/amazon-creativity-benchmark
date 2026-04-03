#!/usr/bin/env bash
# ============================================================================
# Evaluate: Research Idea Execution
# Dataset ID: research_idea_execution
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2506.20803
# Repo:  https://github.com/NoviScl/AI-Researcher

# ============================================================================
#
# Usage:
#   ./research_idea_execution.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./research_idea_execution.sh openai/gpt-4o
#   ./research_idea_execution.sh openai/gpt-4o my-suite
#   ./research_idea_execution.sh openai/gpt-4o my-suite 50
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
RUN_ENTRY="research_idea_execution:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
CMD=(helm-run --plugins run_specs.research_idea_execution_run_specs --run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  Research Idea Execution"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "================================================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
