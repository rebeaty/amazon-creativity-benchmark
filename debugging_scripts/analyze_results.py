#!/usr/bin/env python3
"""
Analyze debug results after a full run.
Usage: python3 analyze_results.py
"""

import json
import glob
import os
from collections import Counter

def main():
    results_dir = "debug_results"
    
    passed = []
    failed = []
    missing = []
    
    # Load all result files
    for path in sorted(glob.glob(f"{results_dir}/*_PASSED.json")):
        with open(path) as f:
            passed.append(json.load(f))
    
    for path in sorted(glob.glob(f"{results_dir}/*_FAILED.json")):
        with open(path) as f:
            failed.append(json.load(f))
    
    # Cross-reference with expected dataset list
    with open("scenarios/subsampled_list.json") as f:
        dataset_list = json.load(f)
    
    if isinstance(dataset_list[0], dict):
        all_datasets = {d.get("name", d.get("dataset", d.get("id", ""))) for d in dataset_list}
    else:
        all_datasets = set(dataset_list)
    
    processed = {d["dataset"] for d in passed} | {d["dataset"] for d in failed}
    missing = all_datasets - processed
    
    print("=" * 60)
    print("HELM DEBUGGING RESULTS ANALYSIS")
    print("=" * 60)
    print(f"Total datasets:    {len(all_datasets)}")
    print(f"  PASSED:          {len(passed)}")
    print(f"  FAILED:          {len(failed)}")
    print(f"  NOT PROCESSED:   {len(missing)}")
    print()
    
    if failed:
        print("── Failed Datasets ─────────────────────────────────────")
        error_types = Counter()
        for d in failed:
            err = d.get("last_error", "unknown")
            print(f"  ✗ {d['dataset']}")
            print(f"    Error: {err[:120]}")
            print(f"    Attempts: {d.get('attempts', '?')}")
            print()
            # Categorize errors
            err_lower = err.lower()
            if "metric" in err_lower:
                error_types["Metric errors"] += 1
            elif "download" in err_lower or "data" in err_lower or "file not found" in err_lower:
                error_types["Data availability"] += 1
            elif "template" in err_lower or "prompt" in err_lower:
                error_types["Prompt/template errors"] += 1
            elif "config" in err_lower or "inference" in err_lower:
                error_types["Config errors"] += 1
            else:
                error_types["Other"] += 1
        
        print("── Error Categories ────────────────────────────────────")
        for cat, count in error_types.most_common():
            print(f"  {cat}: {count}")
        print()
    
    if missing:
        print("── Not Processed ───────────────────────────────────────")
        for d in sorted(missing):
            print(f"  ? {d}")
        print()
    
    if passed:
        print("── Passed Datasets ─────────────────────────────────────")
        total_attempts = sum(d.get("attempts", 1) for d in passed)
        print(f"  Average attempts to pass: {total_attempts / len(passed):.1f}")
        multi_attempt = [d for d in passed if d.get("attempts", 1) > 1]
        if multi_attempt:
            print(f"  Required multiple attempts: {len(multi_attempt)}")
            for d in multi_attempt:
                print(f"    {d['dataset']}: {d['attempts']} attempts")
    
    # Write machine-readable summary
    summary = {
        "total": len(all_datasets),
        "passed": len(passed),
        "failed": len(failed),
        "missing": len(missing),
        "failed_datasets": [d["dataset"] for d in failed],
        "missing_datasets": sorted(missing),
    }
    with open(f"{results_dir}/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMachine-readable summary: {results_dir}/analysis_summary.json")

if __name__ == "__main__":
    main()