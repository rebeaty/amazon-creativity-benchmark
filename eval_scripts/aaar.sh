#!/usr/bin/env bash
# ============================================================================
# Evaluate: AAAR-1.0 (Assessing AI's Potential to Assist Research)
# Dataset ID: aaar
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2410.22394
# Repo:  https://github.com/RenzeLou/AAAR-1.0
#   Subsets: experiment_design, paper_weakness
# ============================================================================
#
# Usage:
#   ./aaar.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./aaar.sh openai/gpt-4o
#   ./aaar.sh openai/gpt-4o my-suite
#   ./aaar.sh openai/gpt-4o my-suite 50
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
RUN_ENTRIES=()
RUN_ENTRIES+=("aaar_experiment_design,model=${MODEL}")
RUN_ENTRIES+=("aaar_paper_weakness,model=${MODEL}")

# ── Build and execute HELM command ──────────────────────────────────────────
CMD=(helm-run --run-entries "${RUN_ENTRIES[@]}" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  AAAR-1.0 (Assessing AI's Potential to Assist Research)"
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
