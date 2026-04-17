# humordb — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| accuracy | formula_based | true | `helm.benchmark.metrics.basic_metrics.BasicMetric` |

## Current Run Spec Metrics
- `MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})`

## Actual Stats.json Metrics (m2)
- `classification_macro_f1`
- `classification_micro_f1`

## Missing Metrics
- **accuracy**: Missing because `MultipleChoiceClassificationMetric` emits `classification_macro_f1` and `classification_micro_f1`, not `accuracy`. The registry requires `BasicMetric` which emits `accuracy`.

## Root Cause
**Wrong class** — the run_spec uses `MultipleChoiceClassificationMetric` instead of `BasicMetric`. Although the dataset uses `ADAPT_MULTIPLE_CHOICE_JOINT` adapter, the registry explicitly maps `humordb` to `BasicMetric` (`helm.benchmark.metrics.basic_metrics.BasicMetric`), which produces the `accuracy` metric. The wrong class was chosen, producing F1 scores instead of accuracy.

## Proposed Fix
In `run_specs/humordb_run_specs.py`, replace the `MetricSpec` class_name from:
```python
MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})
```
to:
```python
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})
```
