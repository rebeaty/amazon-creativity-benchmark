#!/usr/bin/env python3
"""Generate per-dataset bash evaluation scripts for the Amazon Creativity Benchmark.

Reads registry_master.yaml and the run_spec_function names to produce one
bash script per dataset in eval_scripts/.
"""

import json
import os
import re
import stat
import subprocess
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(ROOT, "data", "registry", "registry_master.yaml")
INFERENCE = os.path.join(ROOT, "data", "registry", "registry_inference.yaml")
DATASETS_JSON = os.path.join(ROOT, "scenarios", "list_of_all_datasets.json")
RUN_SPECS_DIR = os.path.join(ROOT, "run_specs")
OUT_DIR = os.path.join(ROOT, "eval_scripts")

os.makedirs(OUT_DIR, exist_ok=True)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_run_spec_names():
    """Extract @run_spec_function("name") from all run spec files."""
    names = []
    for fname in sorted(os.listdir(RUN_SPECS_DIR)):
        if not fname.endswith(".py"):
            continue
        with open(os.path.join(RUN_SPECS_DIR, fname)) as f:
            for line in f:
                m = re.search(r'@run_spec_function\("([^"]+)"\)', line)
                if m:
                    names.append(m.group(1))
    return names


def map_dataset_to_entries(datasets, all_entries):
    """Map each dataset id to its list of run_spec_function names."""
    mapping = {}
    for ds in datasets:
        # Exact match first
        exact = [e for e in all_entries if e == ds]
        if exact:
            mapping[ds] = exact
            continue
        # Prefix match (e.g., aaar -> aaar_experiment_design, aaar_paper_weakness)
        prefix = [e for e in all_entries if e.startswith(ds + "_")]
        if prefix:
            mapping[ds] = sorted(prefix)
            continue
        # Hyphen-normalized match (graphragbench-wrongone -> graphragbench_wrongone)
        norm = ds.replace("-", "_")
        norm_match = [e for e in all_entries if e == norm]
        if norm_match:
            mapping[ds] = norm_match
            continue
        # Fallback: assume dataset name IS the entry
        mapping[ds] = [ds]
    return mapping


def generate_script(ds_id, entries, master_info, inference_info):
    display = master_info.get("display_name", ds_id)
    paper = master_info.get("source_paper", "")
    repo = master_info.get("source_repo", "")
    modality_in = master_info.get("input_modality", ["text"])
    modality_out = master_info.get("output_modality", ["text"])

    # Build run entries line(s)
    if len(entries) == 1:
        run_entries_block = f'RUN_ENTRY="{entries[0]},model=${{MODEL}}"'
        helm_run_line = 'helm-run --run-entries "$RUN_ENTRY" --suite "$SUITE"'
    else:
        lines = []
        for i, e in enumerate(entries):
            lines.append(f'RUN_ENTRIES+=("{e},model=${{MODEL}}")')
        run_entries_block = "RUN_ENTRIES=()\n" + "\n".join(lines)
        helm_run_line = 'helm-run --run-entries "${RUN_ENTRIES[@]}" --suite "$SUITE"'

    # Determine subset labels for display
    subset_note = ""
    if len(entries) > 1:
        subset_labels = [e.replace(ds_id + "_", "") for e in entries]
        subset_note = f"#   Subsets: {', '.join(subset_labels)}"

    script = f'''#!/usr/bin/env bash
# ============================================================================
# Evaluate: {display}
# Dataset ID: {ds_id}
# Input: {", ".join(modality_in)} -> Output: {", ".join(modality_out)}
{"# Paper: " + paper if paper else "# Paper: (not available)"}
{"# Repo:  " + repo if repo else "# Repo:  (not available)"}
{subset_note}
# ============================================================================
#
# Usage:
#   ./{ds_id}.sh MODEL [SUITE] [MAX_INSTANCES]
#
# Examples:
#   ./{ds_id}.sh openai/gpt-4o
#   ./{ds_id}.sh openai/gpt-4o my-suite
#   ./{ds_id}.sh openai/gpt-4o my-suite 50
#
# Arguments:
#   MODEL          Required. The model to evaluate (e.g., openai/gpt-4o).
#   SUITE          Optional. Name for this evaluation run (default: creativity-benchmark).
#   MAX_INSTANCES  Optional. Limit the number of test instances (useful for quick tests).
# ============================================================================

set -euo pipefail

# ── Arguments ───────────────────────────────────────────────────────────────
MODEL="${{1:?Error: MODEL is required. Usage: $0 MODEL [SUITE] [MAX_INSTANCES]}}"
SUITE="${{2:-creativity-benchmark}}"
MAX_INSTANCES="${{3:-}}"

# ── Run entries ─────────────────────────────────────────────────────────────
{run_entries_block}

# ── Build and execute HELM command ──────────────────────────────────────────
CMD=({helm_run_line})
if [ -n "$MAX_INSTANCES" ]; then
    CMD+=(--max-eval-instances "$MAX_INSTANCES")
fi

echo "================================================================"
echo "  Dataset:  {display}"
echo "  Model:    $MODEL"
echo "  Suite:    $SUITE"
[ -n "$MAX_INSTANCES" ] && echo "  Max instances: $MAX_INSTANCES"
echo "================================================================"
echo ""
echo "Running: ${{CMD[*]}}"
echo ""

"${{CMD[@]}}"

# ── Summarize results ──────────────────────────────────────────────────────
echo ""
echo "Summarizing results..."
helm-summarize --suite "$SUITE"

echo ""
echo "Done! Results are in: benchmark_output/runs/$SUITE/"
'''
    return script


def main():
    with open(DATASETS_JSON) as f:
        datasets = json.load(f)

    master = load_yaml(MASTER)["datasets"]
    inference = load_yaml(INFERENCE)["datasets"]
    all_entries = get_run_spec_names()
    ds_to_entries = map_dataset_to_entries(datasets, all_entries)

    for ds_id in datasets:
        entries = ds_to_entries.get(ds_id, [ds_id])
        master_info = master.get(ds_id, {})
        inference_info = inference.get(ds_id, {})

        script_content = generate_script(ds_id, entries, master_info, inference_info)
        script_path = os.path.join(OUT_DIR, f"{ds_id}.sh")
        with open(script_path, "w") as f:
            f.write(script_content)
        # Make executable
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Generated {len(datasets)} evaluation scripts in {OUT_DIR}/")

    # Also generate a run-all script
    run_all = '''#!/usr/bin/env bash
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
'''
    run_all_path = os.path.join(OUT_DIR, "run_all.sh")
    with open(run_all_path, "w") as f:
        f.write(run_all)
    os.chmod(run_all_path, os.stat(run_all_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("Generated run_all.sh")


if __name__ == "__main__":
    main()
