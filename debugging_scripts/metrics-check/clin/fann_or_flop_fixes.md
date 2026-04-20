# Fixes: fann_or_flop

## Missing Metric: `f1`

**Root cause**: The run spec only used `BasicGenerationMetric` which produces `f1_score`, not `f1`.

**Fix**: Added `MetricSpec(class_name="metrics.f1_metric.F1Metric", args={})` to `run_specs/fann_or_flop_run_specs.py`.

The custom `F1Metric` at `metrics/f1_metric.py` emits a stat named `f1` (token-level max F1 across gold references), matching the registry expectation.

**File changed**: `run_specs/fann_or_flop_run_specs.py` — added one MetricSpec line.
