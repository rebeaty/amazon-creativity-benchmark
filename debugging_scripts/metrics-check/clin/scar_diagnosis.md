# Diagnosis: scar

## Expected vs Actual Metrics

| | Metrics |
|---|---|
| Expected (m1) | `exact_match`, `f1` |
| Actual (m2) | `bleu_1`, `bleu_4`, `exact_match`, `f1_score`, `quasi_exact_match`, `rouge_l` |
| Missing | `f1` |

## Root Cause

`run_specs/scar_run_specs.py` uses `BasicGenerationMetric` with names
`["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]`.

`BasicGenerationMetric` produces a stat called `f1_score`, **not** `f1`.
The registry (`registry_metrics.yaml`) expects a stat named `f1` (token-level F1).

A custom metric class already exists at `metrics/f1_metric.py` (`F1Metric`) that
produces exactly the stat name `f1`.

## Proposed Fix

Add a `MetricSpec` for `metrics.f1_metric.F1Metric` to `metric_specs` in
`run_specs/scar_run_specs.py`. Keep the existing `BasicGenerationMetric` spec
(it still supplies `exact_match` which is also expected).
