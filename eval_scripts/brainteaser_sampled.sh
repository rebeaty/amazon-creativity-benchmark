#!/usr/bin/env bash
# ============================================================================
# Evaluate (SAMPLED MIRROR): BrainTeaser — SP + WP subtasks, 200-item sampled
# Dataset ID: brainteaser_sampled
# Input: text -> Output: text
# Paper: https://arxiv.org/abs/2310.05057
# Repo:  https://github.com/1171-jpg/BrainTeaser
#   Subtasks: sentence_puzzle, word_puzzle
# ============================================================================
#
# Usage:
#   ./brainteaser_sampled.sh MODEL [SUITE] [MAX_INSTANCES]
# ============================================================================

set -euo pipefail

MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}"
SUITE="${2:-creativity-benchmark}"
MAX_INSTANCES="${3:-}"

RUN_ENTRIES=()
RUN_ENTRIES+=("brainteaser_sampled_sentence_puzzle:model=${MODEL}")
RUN_ENTRIES+=("brainteaser_sampled_word_puzzle:model=${MODEL}")

source "$(dirname "$0")/_helm_run.sh"
CMD=(--run-entries "${RUN_ENTRIES[@]}" --suite "$SUITE")
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  BrainTeaser (sampled, SP + WP)"
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
