# Diagnosis: recombination_extraction

## Expected vs Actual Metrics
- **Expected (registry)**: `f1`, `classification_accuracy`
- **Actual (trial run)**: `bleu_1`, `bleu_4`, `exact_match`, `f1_score`, `quasi_exact_match`, `rouge_l`
- **Missing**: `f1`, `classification_accuracy`

## Root Cause

The run spec uses only `BasicGenerationMetric` (twice, duplicated), which produces:
`exact_match`, `quasi_exact_match`, `f1_score`, `rouge_l`, `bleu_1`, `bleu_4`

These produce `f1_score` (not `f1`) and no `classification_accuracy`.

Two custom metric classes exist in `metrics/` that produce the expected stat names:
- `metrics/f1_metric.py` → `F1Metric` → stat: `f1`
- `metrics/classification_accuracy_metric.py` → `ClassificationAccuracyMetric` → stat: `classification_accuracy`

Neither is referenced in the run spec.

## Proposed Fix

1. Replace the duplicated `BasicGenerationMetric` entries with one copy.
2. Add `MetricSpec` for `metrics.f1_metric.F1Metric` (produces `f1`).
3. Add `MetricSpec` for `metrics.classification_accuracy_metric.ClassificationAccuracyMetric` (produces `classification_accuracy`).
