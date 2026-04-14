---
name: metrics-check-test
description: Test whether a dataset's trial run results contain all expected metrics from the registry. Compares registry_metrics.yaml (m1) against stats.json (m2) and returns missing metrics. Use when asked to check metrics coverage, verify trial run results, or test if a dataset passes the metrics check.
allowed-tools: Read, Bash, Grep, Glob
user-invocable: true
---

# Metrics Check Test

Compare expected metrics from the registry against actual metrics in trial run results.

## What This Skill Does

1. Extracts the expected metric list (m1) from `data/registry/registry_metrics.yaml`
2. Extracts the actual metric list (m2) from `outputs/first-trial-run/trial/<dataset>:*/stats.json`
3. Computes `missing = set(m1) - set(m2)`
4. Returns the result

## How To Run

Use the Python helper script:

```bash
python3 debugging_scripts/metrics-check/metrics_check.py <dataset> --format human
```

Or for JSON output (used by orchestrator):

```bash
python3 debugging_scripts/metrics-check/metrics_check.py <dataset>
```

## Exit Codes

- `0` — all expected metrics found (pass)
- `1` — some metrics are missing (fail)
- `2` — dataset not in registry or no stats.json found

## Output Format (JSON)

```json
{
  "dataset": "<name>",
  "m1": ["metric_a", "metric_b"],
  "m2": ["metric_a"],
  "missing": ["metric_b"],
  "stats_files": ["path/to/stats.json"],
  "status": "fail"
}
```

## Key Files

- **Registry**: `data/registry/registry_metrics.yaml` — source of truth for expected metrics per dataset
- **Stats output**: `outputs/first-trial-run/trial/<dataset>:*model=google_gemini-2.5-flash-lite/stats.json`
- **Helper script**: `debugging_scripts/metrics-check/metrics_check.py`

## HELM Infrastructure Metrics (filtered out)

These are always present in stats.json and are NOT dataset-specific evaluation metrics:
`num_references`, `num_train_trials`, `num_prompt_tokens`, `num_completion_tokens`,
`num_output_tokens`, `num_train_instances`, `prompt_truncated`, `finish_reason_*`,
`inference_runtime`, `batch_size`, `logprob`, `max_prob`, `num_perplexity_tokens`,
`num_bytes`, `training_co2_cost`, `training_energy_cost`

## Workflow Context

This skill is the "test" step in the metrics-check debugging loop:
1. **Test** (this skill) — are all metrics present?
2. If missing: diagnose + fix (metrics-diagnose-fix skill)
3. Re-run eval (`init_eval.sh`)
4. Loop back to step 1
