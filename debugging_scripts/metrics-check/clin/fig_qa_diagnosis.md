# Diagnosis: fig_qa

## Expected vs Actual Metrics
- **Expected (registry)**: `exact_match`
- **Actual (trial run)**: `classification_macro_f1`, `classification_micro_f1`
- **Missing**: `exact_match`

## Root Cause
`run_specs/fig_qa_run_specs.py` uses `MultipleChoiceClassificationMetric`, which produces `classification_macro_f1` and `classification_micro_f1` — not `exact_match`.

The registry (`registry_metrics.yaml`) maps `fig_qa` to `exact_match` (formula_based, via `compute_reference_metrics`). HELM computes this through `BasicGenerationMetric(names=["exact_match"])`.

## Proposed Fix
Replace:
```python
MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})
```
With:
```python
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match"]})
```

`BasicGenerationMetric` delegates to `compute_reference_metrics` and will output an `exact_match` stat, matching what the registry expects.
