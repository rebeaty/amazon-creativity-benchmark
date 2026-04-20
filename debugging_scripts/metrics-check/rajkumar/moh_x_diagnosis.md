# moh_x — Metrics Diagnosis (Attempt 1)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
|--------|------|---------|------------|
| `accuracy` | formula_based | true | `helm.benchmark.metrics.basic_metrics.BasicMetric` |
| `f1` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |

## Run Spec Currently Has
- `MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})` ×2

## Actual Output (m2)
- `classification_macro_f1`, `classification_micro_f1` — produced by `MultipleChoiceClassificationMetric`

## Missing
- `accuracy`, `f1`

## Root Cause
**Wrong class (Pattern A) ×2.** Both MetricSpecs use `MultipleChoiceClassificationMetric`. The
registry requires `BasicMetric` for `accuracy` and `compute_reference_metrics` for `f1`.

## Registry Naming Discrepancy (⚠ tell Vijeta)
The registry specifies metric name `f1`, but `compute_reference_metrics` in HELM actually produces
a metric named `f1_score` (confirmed in Phase 1 stats for other datasets). This is a registry
naming error — `f1` should be `f1_score`. After the run spec fix, `accuracy` will pass but
`f1` will still show as missing in metrics_check (since HELM outputs `f1_score`).

## Proposed Fix
Replace both `MultipleChoiceClassificationMetric` entries with:
1. `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})`
2. `MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={})`

Registry must be corrected separately: `f1` → `f1_score`.
