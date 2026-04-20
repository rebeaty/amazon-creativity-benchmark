# metaphoric_analogies — Fixes Summary

## File Modified
`run_specs/metaphoric_analogies_run_specs.py`

## Change Made
Replaced `BasicGenerationMetric` (TODO fallback) with the registry-specified `compute_reference_metrics` class.

**Attempt 2 Before (reverted state):**
```python
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match"]})
```

**After:**
```python
MetricSpec(
    class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics",
    args={},
)
```

## Why
The `BasicGenerationMetric` produces `f1_score` (not `f1`). The registry expects `f1`, which is produced by `compute_reference_metrics`. This is a Pattern A fix (in_helm: true, use helm_class from registry directly). The single `compute_reference_metrics` MetricSpec produces both `exact_match` and `f1`, satisfying both registry requirements.
