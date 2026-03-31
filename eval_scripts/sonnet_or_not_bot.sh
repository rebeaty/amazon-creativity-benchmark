#!/usr/bin/env bash
# ============================================================================
# Evaluate: Sonnet or Not Bot
# Dataset ID: sonnet_or_not_bot
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2406.18906
# Repo:  https://github.com/maria-antoniak/poetry-eval

# ============================================================================
#
# Usage:
#   ./sonnet_or_not_bot.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./sonnet_or_not_bot.sh openai/gpt-4o
#   ./sonnet_or_not_bot.sh openai/gpt-4o my-suite
#   ./sonnet_or_not_bot.sh openai/gpt-4o my-suite 50
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
RUN_ENTRY="sonnet_or_not_bot,model=${MODEL}"

# ── Build and execute HELM command ──────────────────────────────────────────
CMD=(helm-run --run-entries "$RUN_ENTRY" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  Sonnet or Not Bot"
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
