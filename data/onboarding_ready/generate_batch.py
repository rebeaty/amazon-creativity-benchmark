#!/usr/bin/env python3
"""Generate batch JSON from classified_benchmarks.csv"""

import csv
import json
from pathlib import Path

def main():
    csv_path = Path(__file__).parent / "classified_benchmarks.csv"
    output_path = Path(__file__).parent / "batch_all.json"
    
    benchmarks = []
    seen_names = set()
    
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("benchmark_name", "").strip()
            paper_id = row.get("paper_id", "").strip()
            
            if not name or not paper_id:
                continue
            
            # Skip duplicates
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            
            benchmarks.append({
                "name": name,
                "paper_id": paper_id
            })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    
    print(f"Generated {len(benchmarks)} unique benchmarks -> {output_path}")

if __name__ == "__main__":
    main()
