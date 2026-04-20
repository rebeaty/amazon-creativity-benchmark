# puntuguese — Fixes Summary

## File Modified
`run_specs/puntuguese_run_specs.py`

## What Changed
Replaced two duplicate `MultipleChoiceClassificationMetric` MetricSpecs with the correct custom metric classes:

**Before:**
```python
metric_specs = [
    MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={}),
    MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={}),
]
```

**After:**
```python
metric_specs = [
    MetricSpec(class_name="metrics.accuracy_metric.AccuracyMetric", args={}),
    MetricSpec(class_name="metrics.f1_metric.F1Metric", args={}),
]
```

## Why
- `MultipleChoiceClassificationMetric` emits `classification_macro_f1` and `classification_micro_f1`, not the required `accuracy` and `f1`.
- `AccuracyMetric` (custom) emits the `accuracy` stat via exact-match against the CORRECT_TAG reference.
- `F1Metric` (custom) emits the `f1` stat via token-level F1 against the CORRECT_TAG reference.
- The registry's listed `helm_class` values (`BasicMetric`, `compute_reference_metrics`) do not exist under those names in the installed HELM package; the custom classes in `metrics/` are the correct implementations.
