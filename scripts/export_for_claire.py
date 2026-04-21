#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export sampled-pair results to long-format TSV for Claire's stats pipeline.

Scans ``benchmark_output/runs/*/`` for directories matching the sampled
units (``brainteaser_sampled_subtask=*,model=*`` and ``cs4_sampled_subtask=*,model=*``),
pulls per-instance primary metric values out of ``per_instance_stats.json``,
and writes a long-format TSV:

    suite  model  evaluation_unit  item_id  metric_name  metric_value

One row per (suite, model, evaluation_unit, item, metric). Primary metrics
only — HELM infrastructure stats (num_tokens, logprob, etc.) are excluded.

Usage:
    python scripts/export_for_claire.py \\
        --output-dir claire_data_2026-04-21 \\
        [--runs-root benchmark_output/runs]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PRIMARY_METRICS = {
    "brainteaser_sampled": "exact_match",
    "cs4_sampled": "llm_judge_creativity",
}

# Suites to skip from the export. gemini_2_flash was 429-rate-limited on all
# 4 sampled units — no valid stats.
EXCLUDED_SUITES = {
    "gemini_2_flash",
    "trial",
    "trial_sampled",
    "trial_10inst",
}

DIRNAME_PATTERN = re.compile(
    r"^(?P<unit_base>brainteaser_sampled|cs4_sampled)_subtask=(?P<subtask>[^,]+),model=(?P<model_safe>.+)$"
)


def walk_runs(runs_root: Path):
    for suite_dir in sorted(runs_root.iterdir()):
        if not suite_dir.is_dir() or suite_dir.name.startswith("_"):
            continue
        if suite_dir.name in EXCLUDED_SUITES:
            continue
        for run_dir in sorted(suite_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = DIRNAME_PATTERN.match(run_dir.name)
            if not m:
                continue
            unit_base = m.group("unit_base")
            subtask = m.group("subtask")
            model_safe = m.group("model_safe")
            # Convert first _ back to / for a clean model id (google_gemini-... -> google/gemini-...)
            model_id = model_safe.replace("_", "/", 1)
            yield {
                "suite": suite_dir.name,
                "run_dir": run_dir,
                "unit_base": unit_base,
                "subtask": subtask,
                "model_id": model_id,
                "evaluation_unit": f"{unit_base}:subtask={subtask}",
            }


def extract_rows(run_meta: dict):
    """Yield long-format dict rows for a single run directory."""
    pis_path = run_meta["run_dir"] / "per_instance_stats.json"
    if not pis_path.exists():
        return
    pis = json.loads(pis_path.read_text(encoding="utf-8"))
    primary = PRIMARY_METRICS.get(run_meta["unit_base"])
    if not primary:
        return

    for entry in pis:
        item_id = entry.get("instance_id", "")
        for stat in entry.get("stats", []):
            name = stat.get("name", {}).get("name", "")
            split = stat.get("name", {}).get("split", "")
            if name != primary or split != "test":
                continue
            yield {
                "suite": run_meta["suite"],
                "model": run_meta["model_id"],
                "evaluation_unit": run_meta["evaluation_unit"],
                "item_id": item_id,
                "metric_name": name,
                "metric_value": stat.get("mean", ""),
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--runs-root", default="benchmark_output/runs")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Include suites that are missing some of the 4 units.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: discover per-suite unit coverage
    suite_units = {}
    metas = []
    for run_meta in walk_runs(runs_root):
        suite_units.setdefault(run_meta["suite"], set()).add(run_meta["evaluation_unit"])
        metas.append(run_meta)

    EXPECTED_UNITS = {
        "brainteaser_sampled:subtask=sentence_puzzle",
        "brainteaser_sampled:subtask=word_puzzle",
        "cs4_sampled:subtask=instruction",
        "cs4_sampled:subtask=story",
    }
    kept = {s for s, u in suite_units.items() if EXPECTED_UNITS <= u}
    dropped = {s: (EXPECTED_UNITS - u) for s, u in suite_units.items() if not EXPECTED_UNITS <= u}

    print(f"Suites with full 4-unit coverage: {len(kept)}")
    for s in sorted(kept):
        print(f"  keep: {s}")
    if dropped:
        print(f"Suites with missing units (rate-limited or failed):")
        for s, miss in sorted(dropped.items()):
            print(f"  drop: {s} — missing {sorted(miss)}")
        if not args.allow_partial:
            metas = [m for m in metas if m["suite"] in kept]

    rows = []
    for run_meta in metas:
        rows.extend(extract_rows(run_meta))

    if not rows:
        print("No rows found. Are the sampled runs present?")
        return

    tsv_path = out_dir / "sampled_pair_long.tsv"
    cols = ["suite", "model", "evaluation_unit", "item_id", "metric_name", "metric_value"]
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Summary: models × units matrix of item counts
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total rows: {len(rows)}\n\n")
        per_unit = {}
        for r in rows:
            k = (r["suite"], r["model"], r["evaluation_unit"])
            per_unit[k] = per_unit.get(k, 0) + 1
        f.write(f"{'suite':<24} {'model':<45} {'evaluation_unit':<50} n_items\n")
        for k, n in sorted(per_unit.items()):
            suite, model, unit = k
            f.write(f"{suite:<24} {model:<45} {unit:<50} {n}\n")

    print(f"Wrote {len(rows)} rows to {tsv_path}")
    print(f"Summary at {summary_path}")


if __name__ == "__main__":
    main()
