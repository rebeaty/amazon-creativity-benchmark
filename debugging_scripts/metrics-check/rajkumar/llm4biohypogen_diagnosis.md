# llm4biohypogen — Metrics Diagnosis (Attempt 1)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
|--------|------|---------|------------|
| `bleu_4` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |
| `rouge_l` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |
| `bert_score` | model_based | true | `helm.benchmark.metrics.summarization_metrics.SummarizationMetric` |

## Run Spec Currently Has
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={...})`

## Actual Output (m2)
- `bleu_4`, `rouge_l` present (from BasicGenerationMetric) ✓
- No `bert_score`

## Missing
- `bert_score`

## Root Cause
`BasicGenerationMetric` doesn't produce `bert_score`. The registry requires `SummarizationMetric`
from HELM which can produce `bert_score` when the `bert_score` Python package is installed.

## Environment Dependency
`bert_score` package is NOT installed locally (`ModuleNotFoundError: No module named 'bert_score'`).
Install via: `pip install "crfm-helm[summarization]"`. Server eval should have this installed.

## Proposed Fix
Replace `BasicGenerationMetric` with:
1. `MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={})` — for bleu_4, rouge_l
2. `MetricSpec(class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric", args={"task": "summarization", "language": "en", "normalize_by_length": False})` — for bert_score

Note: Cannot verify bert_score locally due to missing dependency; server eval required.
