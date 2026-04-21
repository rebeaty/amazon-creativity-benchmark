#!/usr/bin/env bash
# ============================================================================
# Evaluate (SAMPLED MIRROR): CS4 — instruction + story subtasks, 200-item sampled
# Dataset ID: cs4_sampled
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2410.04197
# Repo:  https://github.com/anirudhlakkaraju/cs4_benchmark
#   Subtasks: instruction, story
# ============================================================================
#
# Usage:
#   ./cs4_sampled.sh MODEL [SUITE] [MAX_INSTANCES]
# ============================================================================

set -euo pipefail

MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}"
SUITE="${2:-creativity-benchmark}"
MAX_INSTANCES="${3:-}"

RUN_ENTRIES=()
RUN_ENTRIES+=("cs4_sampled_instruction:model=${MODEL}")
RUN_ENTRIES+=("cs4_sampled_story:model=${MODEL}")

source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "${RUN_ENTRIES[@]}" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  CS4 (sampled, instruction + story)"
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
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
