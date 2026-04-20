#!/usr/bin/env bash
# ============================================================================
# Orchestrator: First Full Trial Evaluation (parallel)
#
# Runs the per-dataset evaluation scripts in ./eval_scripts/ for every dataset
# listed in data/list_dataset_1st_trial.json. Datasets execute concurrently in
# batches of PARALLELISM. All runs share the suite "first_full_trial" and land
# in benchmark_output/runs/first_full_trial/.
#
# Usage:
#   ./run_all_parallel.sh MODEL [MAX_INSTANCES] [PARALLELISM]
#
# Examples:
#   ./run_all_parallel.sh google/gemini-2.5-flash-lite
#   ./run_all_parallel.sh google/gemini-2.5-flash-lite -1 4       # all instances, 4 at a time
#   ./run_all_parallel.sh google/gemini-2.5-flash-lite 10 8       # 10 instances, 8-way parallel
#
# Arguments:
#   MODEL          Required. OpenRouter-formatted model slug (e.g. vendor/model).
#   MAX_INSTANCES  Optional. Integer. -1 (default) means evaluate ALL instances.
#   PARALLELISM    Optional. How many dataset scripts to run concurrently. Default 4.
#
# Notes on model access:
#   - The evaluated MODEL is looked up on OpenRouter. If it is not listed, the
#     script exits with a warning BEFORE running any dataset.
#   - Judge models are hard-coded inside each dataset's run_spec; they are also
#     routed via OpenRouter (same OPENROUTER_API_KEY). We sanity-check a small
#     set of commonly used judge slugs up front.
#   - Dataset downloads are handled inside the HELM scenario Python code, so
#     missing data is fetched automatically on first run.
# ============================================================================

set -uo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_LIST="$PROJECT_ROOT/data/list_dataset_1st_trial.json"
LOG_DIR="$PROJECT_ROOT/benchmark_output/runs/first_full_trial/_orchestrator_logs"
STATUS_DIR="$LOG_DIR/_status"

# ── Arguments ──────────────────────────────────────────────────────────────
MODEL="${1:?Error: MODEL is required. Usage: $0 MODEL [MAX_INSTANCES] [PARALLELISM]}"
MAX_INSTANCES="${2:--1}"
PARALLELISM="${3:-4}"
SUITE="first_full_trial"

if ! [[ "$PARALLELISM" =~ ^[0-9]+$ ]] || [ "$PARALLELISM" -lt 1 ]; then
    echo "ERROR: PARALLELISM must be a positive integer (got '$PARALLELISM')." >&2
    exit 2
fi

# Convert MAX_INSTANCES=-1 to empty so the downstream scripts run on all instances.
if [ "$MAX_INSTANCES" = "-1" ]; then
    MAX_ARG=""
else
    MAX_ARG="$MAX_INSTANCES"
fi

# ── Load .env if present (for OPENROUTER_API_KEY) ──────────────────────────
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ── Validate OpenRouter credentials ────────────────────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set. All model access (including judge" >&2
    echo "       models) is routed through OpenRouter. Aborting." >&2
    exit 2
fi

# ── OpenRouter model availability check ────────────────────────────────────
# OpenRouter does NOT expose a per-slug GET (/api/v1/models/{slug} returns 404
# for every model). Fetch the full list once, then check membership locally.
OPENROUTER_MODELS_CACHE="$(mktemp -t openrouter_models.XXXXXX.json)"
trap 'rm -f "$OPENROUTER_MODELS_CACHE"' EXIT

_openrouter_list_http=$(curl -s -o "$OPENROUTER_MODELS_CACHE" -w "%{http_code}" \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    "https://openrouter.ai/api/v1/models")
if [ "$_openrouter_list_http" != "200" ]; then
    echo "ERROR: failed to fetch OpenRouter model list (HTTP $_openrouter_list_http)." >&2
    echo "       Check OPENROUTER_API_KEY and network access." >&2
    exit 2
fi

check_openrouter_model() {
    local slug="$1"
    python3 - "$slug" "$OPENROUTER_MODELS_CACHE" <<'PY'
import json, sys
slug, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
ids = {m.get("id") for m in data.get("data", [])}
sys.exit(0 if slug in ids else 1)
PY
}

echo "Verifying evaluation model is reachable on OpenRouter: $MODEL"
if ! check_openrouter_model "$MODEL"; then
    echo "WARNING: model '$MODEL' is not available on OpenRouter (HTTP != 200)." >&2
    echo "         Refusing to start trial. Use an OpenRouter-compatible slug" >&2
    echo "         like 'anthropic/claude-sonnet-4' or 'openai/gpt-4o'." >&2
    exit 3
fi
echo "  OK."

# Judge models observed in run_specs/ — sanity check the common ones.
JUDGE_SLUGS=(
    "openai/gpt-4-1106-preview"
    "anthropic/claude-sonnet-4"
    "google/gemini-2.5-flash-lite"
)
echo "Verifying judge models are reachable on OpenRouter:"
for slug in "${JUDGE_SLUGS[@]}"; do
    if check_openrouter_model "$slug"; then
        echo "  OK   $slug"
    else
        echo "WARNING: judge model '$slug' is not available on OpenRouter." >&2
        echo "         It is referenced by at least one dataset's run_spec." >&2
        echo "         Aborting before running the trial." >&2
        exit 4
    fi
done

# ── Load dataset list ──────────────────────────────────────────────────────
if [ ! -f "$DATASET_LIST" ]; then
    echo "ERROR: dataset list not found at $DATASET_LIST" >&2
    exit 5
fi

# Parse JSON array of strings without requiring jq.
mapfile -t DATASETS < <(python3 -c "
import json
with open('$DATASET_LIST') as f:
    for name in json.load(f):
        print(name)
")

if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "ERROR: dataset list is empty." >&2
    exit 6
fi

mkdir -p "$LOG_DIR" "$STATUS_DIR"
# Clean stale status markers from a previous run.
rm -f "$STATUS_DIR"/*.status 2>/dev/null || true

# ── Single-dataset worker ──────────────────────────────────────────────────
# Runs one dataset script, writes a status marker when done. Safe to background.
run_one() {
    local idx="$1"
    local total="$2"
    local dataset="$3"
    local script="$SCRIPT_DIR/${dataset}.sh"
    local log="$LOG_DIR/${dataset}.log"
    local status="$STATUS_DIR/${dataset}.status"

    if [ ! -f "$script" ]; then
        printf 'SKIP\tno script at %s\n' "$script" >"$status"
        echo "[$idx/$total] SKIP $dataset (no script)"
        return 0
    fi

    echo "[$idx/$total] START $dataset"
    # Dataset data download is handled inside the HELM scenario's Python code
    # (get_instances -> ensure_file_downloaded / HuggingFace hub).
    if bash "$script" "$MODEL" "$SUITE" "$MAX_ARG" >"$log" 2>&1; then
        printf 'PASS\t0\n' >"$status"
        echo "[$idx/$total] PASS  $dataset"
    else
        local rc=$?
        printf 'FAIL\t%s\n' "$rc" >"$status"
        echo "[$idx/$total] FAIL  $dataset (rc=$rc, log: $log)" >&2
    fi
}

# ── Bounded parallel dispatch ──────────────────────────────────────────────
TOTAL=${#DATASETS[@]}

echo "================================================================"
echo "  First Full Trial"
echo "  Model:         $MODEL"
echo "  Suite:         $SUITE"
echo "  Max instances: ${MAX_ARG:-ALL}"
echo "  Parallelism:   $PARALLELISM"
echo "  Datasets:      $TOTAL"
echo "  Logs:          $LOG_DIR"
echo "================================================================"

running=0
i=0
for dataset in "${DATASETS[@]}"; do
    i=$((i + 1))

    # Throttle: if we already have PARALLELISM workers in flight, wait for one
    # to finish before launching the next.
    while [ "$running" -ge "$PARALLELISM" ]; do
        wait -n
        running=$((running - 1))
    done

    run_one "$i" "$TOTAL" "$dataset" &
    running=$((running + 1))
done

# Drain remaining workers.
while [ "$running" -gt 0 ]; do
    wait -n
    running=$((running - 1))
done

# ── Collect results from status markers ────────────────────────────────────
PASSED=()
FAILED=()
SKIPPED=()
for dataset in "${DATASETS[@]}"; do
    status="$STATUS_DIR/${dataset}.status"
    if [ ! -f "$status" ]; then
        FAILED+=("$dataset")
        continue
    fi
    state=$(cut -f1 "$status")
    case "$state" in
        PASS) PASSED+=("$dataset") ;;
        SKIP) SKIPPED+=("$dataset") ;;
        *)    FAILED+=("$dataset") ;;
    esac
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Trial complete"
echo "  Passed:  ${#PASSED[@]}"
echo "  Failed:  ${#FAILED[@]}"
echo "  Skipped: ${#SKIPPED[@]}"
echo "  Results: $PROJECT_ROOT/benchmark_output/runs/$SUITE/"
echo "================================================================"

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed datasets:"
    printf '  - %s\n' "${FAILED[@]}"
fi
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "Skipped datasets (missing script):"
    printf '  - %s\n' "${SKIPPED[@]}"
fi

# Exit non-zero if anything failed so CI/automation notices.
if [ "${#FAILED[@]}" -gt 0 ] || [ "${#SKIPPED[@]}" -gt 0 ]; then
    exit 1
fi
exit 0
