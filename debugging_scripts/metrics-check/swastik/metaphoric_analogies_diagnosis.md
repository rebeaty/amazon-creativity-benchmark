# metaphoric_analogies — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| exact_match | formula_based | true | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics |
| f1 | formula_based | true | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics |

## Current Run Spec Metrics (attempt 2)
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match"]})`
- Note: attempt 1 fix was reverted; re-applying same fix

## Actual Stats.json Metrics (m2)
- bleu_1, bleu_4, exact_match, f1_score, quasi_exact_match, rouge_l

## Missing Metrics
- **f1**: Missing because `BasicGenerationMetric` produces `f1_score` (not `f1`). The registry-expected `f1` metric is the token-level F1 produced by `compute_reference_metrics`, which is a different value/name.

## Root Cause
The run_spec uses `BasicGenerationMetric` (a TODO fallback pattern) instead of the registry-specified `compute_reference_metrics` class. `BasicGenerationMetric` with `names=["f1_score", ...]` produces a metric named `f1_score`, while the registry expects `f1` — produced exclusively by `compute_reference_metrics`. Both `exact_match` and `f1` should come from a single `MetricSpec` using `compute_reference_metrics`.

## Proposed Fix
Replace the `BasicGenerationMetric` MetricSpec with:
```python
MetricSpec(
    class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics",
    args={},
)
```
This single MetricSpec produces both `exact_match` and `f1`, matching the registry exactly.
