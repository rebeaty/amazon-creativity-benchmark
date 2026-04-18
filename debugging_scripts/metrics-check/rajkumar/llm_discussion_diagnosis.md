# llm_discussion — Metrics Diagnosis (Attempt 1)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
|--------|------|---------|------------|
| `self_bleu` | formula_based | true | `helm.benchmark.metrics.disinformation_metrics.DisinformationMetric` |
| `distinct_1` | formula_based | false | `metrics.distinct_n_metric.DistinctNMetric` |
| `distinct_2` | formula_based | false | `metrics.distinct_n_metric.DistinctNMetric` |

## Run Spec Currently Has
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={...})`
- `MetricSpec(class_name="metrics.distinct_n_metric.DistinctNMetric", args={"n": 1})` ✓
- `MetricSpec(class_name="metrics.distinct_n_metric.DistinctNMetric", args={"n": 2})` ✓

## Actual Output (m2)
- `distinct_1`, `distinct_2` present (from DistinctNMetric) ✓
- No `self_bleu` (BasicGenerationMetric doesn't produce it)

## Missing
- `self_bleu`

## Root Cause
The `BasicGenerationMetric` doesn't produce `self_bleu`. The registry requires
`DisinformationMetric` which does. The DistinctNMetric entries are already correct.

## Proposed Fix
Replace `BasicGenerationMetric` with `DisinformationMetric`. Keep the two DistinctNMetric entries.
