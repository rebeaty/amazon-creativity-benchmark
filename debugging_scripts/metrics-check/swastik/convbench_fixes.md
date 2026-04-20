# convbench — Fixes Summary

## Fix Applied (Attempt 2 — code change actually applied)

**File:** `run_specs/convbench_run_specs.py`

**Change:** Replaced wrong MetricSpec class with the registry-specified one.

- **Before:** `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]})`
- **After:** `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})`

**Why:** The run_spec had a TODO fallback using `BasicGenerationMetric` which produces text-similarity metrics (bleu, rouge, exact_match). The registry specifies `BasicMetric` (in_helm: true, formula_based) which produces `accuracy`. Switching to the correct class resolves the missing metric.
