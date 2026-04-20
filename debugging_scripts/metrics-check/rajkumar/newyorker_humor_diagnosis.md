# newyorker_humor — Metrics Diagnosis (Attempt 1)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
|--------|------|---------|------------|
| `accuracy` | formula_based | true | `helm.benchmark.metrics.basic_metrics.BasicMetric` |
| `rouge_l` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |

## Run Spec Currently Has
- `MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})` ×2

## Actual Output (m2)
- `classification_macro_f1`, `classification_micro_f1` — produced by `MultipleChoiceClassificationMetric`

## Missing
- `accuracy`, `rouge_l`

## Root Cause
**Wrong class (Pattern A) ×2.** Both MetricSpecs use `MultipleChoiceClassificationMetric`. The
registry requires `BasicMetric` for `accuracy` and `compute_reference_metrics` for `rouge_l`.

## Proposed Fix
Replace both `MultipleChoiceClassificationMetric` entries with:
1. `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})`
2. `MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={})`
