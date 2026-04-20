# schnovel — Fixes Applied

## Summary
Replaced wrong MetricSpec class to produce the expected `accuracy` metric.

## Change
**File:** `run_specs/schnovel_run_specs.py`

**Before:**
```python
MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})
```

**After:**
```python
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})
```

## Why
`MultipleChoiceClassificationMetric` produces `classification_macro_f1` and `classification_micro_f1`, not `accuracy`. The registry (`registry_metrics.yaml`) specifies `helm.benchmark.metrics.basic_metrics.BasicMetric` as the correct class, which emits `accuracy` for MCQ tasks. Root cause: **Wrong class**.
