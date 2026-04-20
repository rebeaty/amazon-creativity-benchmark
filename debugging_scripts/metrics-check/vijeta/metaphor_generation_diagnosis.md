# Diagnosis: metaphor_generation — Missing `f1` metric

## Expected vs Actual
- Expected (registry): `bleu_4`, `rouge_l`, `f1`
- Actual (trial run): `bleu_1`, `bleu_4`, `exact_match`, `f1_score`, `quasi_exact_match`, `rouge_l`
- Missing: `f1`

## Root Cause
The run_spec uses `BasicGenerationMetric` with the name list `["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]`.  
`BasicGenerationMetric` emits a stat called `f1_score` — **not** `f1`.

A custom `F1Metric` class already exists at `metrics/f1_metric.py` that wraps `evaluate_reference_metrics.f1_score` and emits a stat named `f1`, matching the registry expectation.

## Proposed Fix
Add a `MetricSpec` for `metrics.f1_metric.F1Metric` to `run_specs/metaphor_generation_run_specs.py`.
