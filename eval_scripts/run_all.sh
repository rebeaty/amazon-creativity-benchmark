#!/usr/bin/env bash
# ============================================================================
# Run ALL datasets in the Amazon Creativity Benchmark
#
# Usage:
#   ./run_all.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Example:
#   ./run_all.sh openai/gpt-4o my-suite 10
# ============================================================================

set -euo pipefail

MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}"
SUITE="${2:-creativity-benchmark}"
MAX_INSTANCES="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PASSED=0
FAILED=0
FAILED_LIST=()

for script in "$SCRIPT_DIR"/*.sh; do
    name="$(basename "$script" .sh)"
    [ "$name" = "run_all" ] && continue

    echo ""
    echo "================================================================"
    echo "  Running: $name"
    echo "================================================================"

    if "$script" "$MODEL" "$SUITE" "$MAX_INSTANCES"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$name")
        echo "WARNING: $name failed, continuing..."
    fi
done

echo ""
echo "================================================================"
echo "  Summary: $PASSED passed, $FAILED failed"
if [ $FAILED -gt 0 ]; then
    echo "  Failed: ${FAILED_LIST[*]}"
fi
echo "================================================================"

# Final summarize
helm-summarize --suite "$SUITE"
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
