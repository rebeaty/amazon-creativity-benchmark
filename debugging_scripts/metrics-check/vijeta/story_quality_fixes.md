# Fixes: story_quality

## Missing Metric: spearman_correlation

**File changed**: `run_specs/story_quality_run_specs.py`

**What was wrong**: Only one `MetricSpec` was present for `CorrelationMetric` with `correlation_type="pearson"`. The `spearman` variant was never instantiated, so `spearman_correlation` stats were never emitted.

**Fix**: Added a second `MetricSpec`:
```python
MetricSpec(class_name="metrics.correlation_metric.CorrelationMetric", args={"correlation_type": "spearman"}),
```

**Verified**: `from metrics.correlation_metric import CorrelationMetric; CorrelationMetric('spearman')` — OK.
