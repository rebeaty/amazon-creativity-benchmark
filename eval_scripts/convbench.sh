#!/usr/bin/env bash
# ============================================================================
# Evaluate: ConvBench
# Dataset ID: convbench
# Input: image, text -> Output: text
# Paper: https://arxiv.org/abs/2403.20194
# Repo:  https://github.com/shirlyliu64/ConvBench

# ============================================================================
#
# Usage:
#   ./convbench.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./convbench.sh openai/gpt-4o
#   ./convbench.sh openai/gpt-4o my-suite
#   ./convbench.sh openai/gpt-4o my-suite 50
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
RUN_ENTRY="convbench:model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
CMD=(helm-run --plugins run_specs.convbench_run_specs --run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  ConvBench"
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
