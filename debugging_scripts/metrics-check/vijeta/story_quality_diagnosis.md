# Diagnosis: story_quality

## Expected vs Actual Metrics

| Metric | Expected | Actual |
|---|---|---|
| spearman_correlation | yes | **missing** |
| pearson_correlation | yes | yes (as pearson_correlation_pred, pearson_correlation_true) |

## Root Cause

`run_specs/story_quality_run_specs.py` only registers one `MetricSpec`:

```python
MetricSpec(class_name="metrics.correlation_metric.CorrelationMetric", args={"correlation_type": "pearson"}),
```

`CorrelationMetric` emits three stats per instance: `{type}_correlation_pred`, `{type}_correlation_true`, and `{type}_correlation` (signed error proxy). With only `correlation_type="pearson"` specified, the spearman variant is never instantiated, so `spearman_correlation` (and its sub-stats) never appear.

## Proposed Fix

Add a second `MetricSpec` for `spearman` to `metric_specs` in the run_spec:

```python
MetricSpec(class_name="metrics.correlation_metric.CorrelationMetric", args={"correlation_type": "spearman"}),
```
