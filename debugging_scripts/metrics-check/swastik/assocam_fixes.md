# assocam — Fixes Applied

## Attempt 1
Replaced `MultipleChoiceClassificationMetric` with `BasicGenerationMetric` + `BasicReferenceMetric` + `InstancesPerSplitMetric`. Result: `exact_match` and `num_instances` appeared but `rouge_l` still missing, and `classification_macro_f1`/`classification_micro_f1` persisted.

## Attempt 2

### File Modified
`run_specs/assocam_run_specs.py`

### Change
Replaced the three-MetricSpec list in all three run spec functions (`assocam_4T1`, `assocam_7T1`, `assocam_10T1`):

**Before:**
```python
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match"]})
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicReferenceMetric", args={})
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.InstancesPerSplitMetric", args={})
```

**After:**
```python
MetricSpec(
    class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics",
    args={},
)
```

### Why
The registry mandates `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` for both `exact_match` and `rouge_l`. This single MetricSpec emits both metrics together. The previous classes never computed ROUGE scores. Import verified: `from helm.benchmark.metrics.evaluate_reference_metrics import compute_reference_metrics` succeeds.

## Attempt 3 (current)

### File Modified
`run_specs/assocam_run_specs.py`

### Change
Changed the adapter method from `ADAPT_MULTIPLE_CHOICE_JOINT` to `ADAPT_GENERATION` in all three run spec functions (`assocam_4T1`, `assocam_7T1`, `assocam_10T1`).

**Before:**
```python
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
...
method=ADAPT_MULTIPLE_CHOICE_JOINT,
```

**After:**
```python
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
...
method=ADAPT_GENERATION,
```

The MetricSpec (`compute_reference_metrics`) is unchanged — it was already correct from Attempt 2.

### Why
`compute_reference_metrics` routes to different metric computation branches based on the adapter mode. Under `ADAPT_MULTIPLE_CHOICE_JOINT`, it computes classification metrics (`classification_macro_f1`, `classification_micro_f1`, `exact_match`) but skips ROUGE-L. Under `ADAPT_GENERATION`, it computes `exact_match` and `rouge_l` by comparing the model's generated text against the correct reference. Since the references are single letters (A–J), ROUGE-L is meaningful and matches the registry expectation.
