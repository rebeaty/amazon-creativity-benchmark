# Fixes: fig_qa

## File Modified
`run_specs/fig_qa_run_specs.py`

## Change
Replaced `MultipleChoiceClassificationMetric` with `BasicGenerationMetric(names=["exact_match"])`.

**Before:**
```python
MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})
```

**After:**
```python
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match"]})
```

## Why
`MultipleChoiceClassificationMetric` outputs `classification_macro_f1` and `classification_micro_f1`, not `exact_match`. The registry requires `exact_match`. `BasicGenerationMetric` delegates to `compute_reference_metrics` which computes and emits the `exact_match` stat.
