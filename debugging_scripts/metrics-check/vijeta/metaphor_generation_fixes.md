# Fixes: metaphor_generation

## Problem
`f1` was missing from trial run output. The run_spec only included `BasicGenerationMetric` which emits `f1_score`, not `f1`.

## Fix
**File**: `run_specs/metaphor_generation_run_specs.py`  
**Change**: Added `MetricSpec(class_name="metrics.f1_metric.F1Metric", args={})` to `metric_specs`.

The custom `F1Metric` in `metrics/f1_metric.py` wraps `evaluate_reference_metrics.f1_score` and emits a stat named `f1`, matching the registry expectation.
