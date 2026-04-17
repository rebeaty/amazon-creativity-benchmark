#!/usr/bin/env python3
"""
Metrics Check — Compare expected metrics (registry) vs actual metrics (stats.json).

Usage:
    python3 metrics_check.py <dataset>                      # JSON output
    python3 metrics_check.py <dataset> --format human       # Human-readable output
    python3 metrics_check.py <dataset> --suite trial --model google/gemini-2.5-flash-lite

Exit codes:
    0  — all expected metrics found (missing is empty)
    1  — some metrics are missing
    2  — dataset not found in registry or no stats.json found
"""

import argparse
import glob
import json
import os
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

REGISTRY_PATH = os.path.join(REPO_ROOT, "data", "registry", "registry_metrics.yaml")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs", "first-trial-run", "trial")
BENCHMARK_RUNS_DIR = os.path.join(REPO_ROOT, "benchmark_output", "runs")

# HELM infrastructure metrics — always present, never dataset-specific
HELM_INFRA_METRICS = {
    "num_references",
    "num_train_trials",
    "num_prompt_tokens",
    "num_completion_tokens",
    "num_output_tokens",
    "num_train_instances",
    "prompt_truncated",
    "finish_reason_length",
    "finish_reason_stop",
    "finish_reason_endoftext",
    "finish_reason_unknown",
    "inference_runtime",
    "batch_size",
    "logprob",
    "max_prob",
    "num_perplexity_tokens",
    "num_bytes",
    "training_co2_cost",
    "training_energy_cost",
}


def get_registry_metrics(dataset: str) -> list[str]:
    """Extract expected metric names from registry_metrics.yaml for a dataset."""
    with open(REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)

    datasets = registry.get("datasets", {})
    if dataset not in datasets:
        return []

    metrics = datasets[dataset].get("metrics", [])
    return [m["name"] for m in metrics if isinstance(m, dict) and "name" in m]


def find_stats_files(dataset: str, suite: str = "trial", model: str = "google/gemini-2.5-flash-lite") -> list[str]:
    """Find all stats.json files for a dataset (handles subtasks).

    Searches OUTPUTS_DIR first, then benchmark_output/runs/<suite>.
    """
    model_safe = model.replace("/", "_")
    files: list[str] = []

    search_dirs = [
        OUTPUTS_DIR,
        os.path.join(BENCHMARK_RUNS_DIR, suite),
    ]

    for base_dir in search_dirs:
        pattern = os.path.join(base_dir, f"{dataset}:*model={model_safe}", "stats.json")
        found = glob.glob(pattern)
        exact = os.path.join(base_dir, f"{dataset}:model={model_safe}", "stats.json")
        if os.path.exists(exact) and exact not in found:
            found.append(exact)
        files.extend(found)

    return sorted(set(files))


def get_stats_metrics(dataset: str, suite: str = "trial", model: str = "google/gemini-2.5-flash-lite") -> dict:
    """Extract actual metric names from stats.json files.

    Returns:
        dict with keys:
            "metrics": list of non-infra metric names (union across all subtask files)
            "files": list of stats.json paths found
            "per_file": dict mapping file path to its metric list
    """
    files = find_stats_files(dataset, suite, model)
    if not files:
        return {"metrics": [], "files": [], "per_file": {}}

    all_metrics = set()
    per_file = {}

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        file_metrics = set()
        for entry in data:
            name_obj = entry.get("name", {})
            metric_name = name_obj.get("name", "")
            split = name_obj.get("split", "")

            # Only consider test-split metrics, skip infra
            if split == "test" and metric_name not in HELM_INFRA_METRICS:
                file_metrics.add(metric_name)

        per_file[fpath] = sorted(file_metrics)
        all_metrics.update(file_metrics)

    return {
        "metrics": sorted(all_metrics),
        "files": files,
        "per_file": per_file,
    }


def compute_missing(dataset: str, suite: str = "trial", model: str = "google/gemini-2.5-flash-lite") -> dict:
    """Compare registry metrics (m1) vs stats metrics (m2), return the diff.

    Returns:
        dict with keys: dataset, m1, m2, missing, stats_files, status
        status is one of: "pass", "fail", "no_registry", "no_stats"
    """
    m1 = get_registry_metrics(dataset)
    if not m1:
        return {
            "dataset": dataset,
            "m1": [],
            "m2": [],
            "missing": [],
            "stats_files": [],
            "status": "no_registry",
        }

    stats_info = get_stats_metrics(dataset, suite, model)
    m2 = stats_info["metrics"]

    if not stats_info["files"]:
        return {
            "dataset": dataset,
            "m1": m1,
            "m2": [],
            "missing": m1,
            "stats_files": [],
            "status": "no_stats",
        }

    missing = sorted(set(m1) - set(m2))

    return {
        "dataset": dataset,
        "m1": m1,
        "m2": m2,
        "missing": missing,
        "stats_files": stats_info["files"],
        "status": "pass" if not missing else "fail",
    }


def main():
    parser = argparse.ArgumentParser(description="Check metrics coverage for a dataset")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--suite", default="trial", help="HELM suite name (default: trial)")
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite", help="Model identifier")
    parser.add_argument("--format", choices=["json", "human"], default="json", help="Output format")
    args = parser.parse_args()

    result = compute_missing(args.dataset, args.suite, args.model)

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Dataset: {result['dataset']}")
        print(f"Status:  {result['status']}")
        print(f"Expected (m1): {result['m1']}")
        print(f"Actual   (m2): {result['m2']}")
        if result["missing"]:
            print(f"Missing:       {result['missing']}")
        else:
            print("Missing:       (none — all metrics present)")
        if result["stats_files"]:
            print(f"Stats files:   {result['stats_files']}")

    # Exit code
    if result["status"] == "pass":
        sys.exit(0)
    elif result["status"] in ("no_registry", "no_stats"):
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
