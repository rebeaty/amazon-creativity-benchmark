# humordb — Fixes Summary

## Fix Applied

**File:** `run_specs/humordb_run_specs.py`

**Change:** Replaced `MultipleChoiceClassificationMetric` with `BasicMetric`.

**Before:**
```python
MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})
```

**After:**
```python
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})
```

## Why
The registry (`registry_metrics.yaml`) maps `humordb` to `accuracy` via `helm.benchmark.metrics.basic_metrics.BasicMetric`. The old class `MultipleChoiceClassificationMetric` was producing `classification_macro_f1` and `classification_micro_f1` instead. Replacing it with `BasicMetric` will produce the expected `accuracy` metric.

## Verification
- `python3 -m py_compile run_specs/humordb_run_specs.py` → Syntax OK
