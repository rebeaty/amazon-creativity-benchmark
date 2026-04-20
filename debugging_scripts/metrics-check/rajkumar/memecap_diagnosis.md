# memecap — Metrics Diagnosis (Attempt 1)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
|--------|------|---------|------------|
| `exact_match` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |
| `f1` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |

## Run Spec Currently Has
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]})`

## Actual Output (m2)
- `exact_match`, `f1_score`, `rouge_l`, `bleu_4`, etc. — all from BasicGenerationMetric

## Missing (per metrics_check)
- `f1` — but HELM produces `f1_score`

## Root Cause
**Registry naming discrepancy (⚠ tell Vijeta).** Same as crowdcounter. The registry specifies
`f1` but `compute_reference_metrics` produces `f1_score`. The metric IS computed correctly —
registry has the wrong name.

## Resolution
No code fix possible without modifying the registry. Registry must be updated:
`f1` → `f1_score` (tell Vijeta). Mark done — metric is computed correctly.
