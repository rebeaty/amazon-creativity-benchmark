# Diagnosis: fann_or_flop

## Expected vs Actual Metrics

| Metric | Expected | Actual |
|--------|----------|--------|
| exact_match | ✅ | ✅ |
| f1 | ✅ | ❌ (f1_score present instead) |

## Root Cause

The run spec uses `BasicGenerationMetric` with `f1_score` in the names list. This produces a stat named `f1_score`, but the registry expects `f1`.

A custom `F1Metric` class exists at `metrics/f1_metric.py` that produces a stat named `f1` (token-level F1, max over gold references). This class needs to be added as a separate MetricSpec.

## Proposed Fix

Add a `MetricSpec` for `metrics.f1_metric.F1Metric` to `run_specs/fann_or_flop_run_specs.py`.
