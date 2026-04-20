# sdat — Metrics Diagnosis (Attempt 1)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
|--------|------|---------|------------|
| `semantic_diversity_score` | model_based | false | null |

## Run Spec Currently Has
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match"]})`

## Actual Output (m2)
- `exact_match` and other basic metrics from BasicGenerationMetric

## Missing
- `semantic_diversity_score`

## Root Cause
**Pattern D — unimplementable.** `semantic_diversity_score` is a `model_based` metric with
`helm_class: null` (no HELM class). It requires an external model
(`ibm-granite/granite-embedding-278m-multilingual`) that cannot be wired as a standard MetricSpec.

## Proposed Fix
Cannot implement as MetricSpec. Keep existing fallback MetricSpec, add TODO comment.
Tell Vijeta: sdat requires custom model-based metric pipeline for `semantic_diversity_score`.
