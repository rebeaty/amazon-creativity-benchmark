# crowdcounter — Metrics Diagnosis (Attempt 1)

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
**Registry naming discrepancy (⚠ tell Vijeta).** The registry specifies metric name `f1`, but
`compute_reference_metrics` in HELM produces a metric named `f1_score`. Phase 1 stats confirm
this: `f1_score` is present, `f1` is not. The underlying metric IS being computed correctly —
the registry has the wrong name.

The run spec itself is functionally correct (`BasicGenerationMetric` with `f1_score` produces the
same metric as `compute_reference_metrics`). `exact_match` is also present in m2.

## Resolution
No code fix possible without modifying the registry. Registry must be updated:
`f1` → `f1_score` for this dataset (tell Vijeta). Mark done — metric is computed correctly.
