# convbench — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| accuracy | formula_based | true | helm.benchmark.metrics.basic_metrics.BasicMetric |

## Current Run Spec Metrics
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]})`

## Actual Stats.json Metrics (m2)
- bleu_1, bleu_4, exact_match, f1_score, quasi_exact_match, rouge_l

## Missing Metrics
- **accuracy**: Missing because the run_spec uses `BasicGenerationMetric` (wrong class) which only produces text-similarity metrics. The registry requires `BasicMetric` which produces `accuracy`.

## Root Cause
The run_spec uses `BasicGenerationMetric` as a TODO fallback, producing exact_match/bleu/rouge metrics. The registry specifies `helm.benchmark.metrics.basic_metrics.BasicMetric` (Pattern A — in_helm: true) which produces `accuracy`. These are two different HELM metric classes; swapping to the correct one will produce the expected `accuracy` metric.

## Proposed Fix
Replace the `MetricSpec` in `run_specs/convbench_run_specs.py`:
- Remove: `BasicGenerationMetric` with `names` args
- Add: `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})`
