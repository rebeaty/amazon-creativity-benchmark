# meta4xnli — Metrics Diagnosis (Attempt 1)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
|--------|------|---------|------------|
| `accuracy` | formula_based | true | `helm.benchmark.metrics.basic_metrics.BasicMetric` |

## Run Spec Currently Has
- `MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})`

## Actual Output (m2)
- `classification_macro_f1`, `classification_micro_f1` — produced by `MultipleChoiceClassificationMetric`

## Missing
- `accuracy`

## Root Cause
**Wrong class (Pattern A).** The run spec uses `MultipleChoiceClassificationMetric`, which produces
`classification_macro_f1` and `classification_micro_f1`. The registry requires `BasicMetric`,
which produces `accuracy`.

## Proposed Fix
Replace MetricSpec `class_name` with `helm.benchmark.metrics.basic_metrics.BasicMetric`.
