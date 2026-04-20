# Fixes: recombination_extraction

## File Modified
`run_specs/recombination_extraction_run_specs.py`

## Changes

### Before
```python
metric_specs = [
    MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": [...]}),
    MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": [...]}),  # duplicate
]
```

### After
```python
metric_specs = [
    MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": [...]}),
    MetricSpec(class_name="metrics.f1_metric.F1Metric", args={}),
    MetricSpec(class_name="metrics.classification_accuracy_metric.ClassificationAccuracyMetric", args={}),
]
```

## Why

- `f1` stat: produced by `metrics.f1_metric.F1Metric` (custom class in `metrics/`). `BasicGenerationMetric` produces `f1_score` not `f1`.
- `classification_accuracy` stat: produced by `metrics.classification_accuracy_metric.ClassificationAccuracyMetric`. No HELM built-in produces this stat name.
- Removed the duplicated `BasicGenerationMetric` entry (was listed twice).
- Both custom classes verified importable: `python3 -c "from metrics.f1_metric import F1Metric; from metrics.classification_accuracy_metric import ClassificationAccuracyMetric"` → OK
