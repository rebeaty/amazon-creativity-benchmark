# brainteaser — Metrics Diagnosis (Attempt 1)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
|--------|------|---------|------------|
| `exact_match` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |

## Run Spec Currently Has
- `MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})`

## Actual Output (m2)
- `classification_macro_f1`, `classification_micro_f1` — produced by `MultipleChoiceClassificationMetric`

## Missing
- `exact_match`

## Root Cause
**Wrong class (Pattern A).** The run spec uses `MultipleChoiceClassificationMetric`, which produces
`classification_macro_f1` and `classification_micro_f1`. The registry requires
`compute_reference_metrics`, which produces `exact_match` (among others). The class was likely set
during initial run spec generation and was never corrected against the registry.

## Proposed Fix
Replace MetricSpec `class_name` with the registry-specified class:
`helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics`
