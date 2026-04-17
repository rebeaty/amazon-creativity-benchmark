# ocw_connections — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| exact_match | formula_based | true | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics |
| rouge_1 | formula_based | true | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics |
| bert_score | model_based | true | helm.benchmark.metrics.summarization_metrics.SummarizationMetric |

## Current Run Spec Metrics (Attempt 3 state)
- `MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={})`
- `MetricSpec(class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric", args={})`

## Actual Stats.json Metrics (m2) — Attempt 2
- bleu_1, bleu_4, exact_match, f1_score, quasi_exact_match, rouge_1, rouge_2, rouge_l

## Missing Metrics

- **bert_score**: `SummarizationMetric` crashes at init time with `args={}` because the constructor requires a mandatory `task: str` argument (no default). Even if it ran, `bert_score` in `SummarizationMetric` is only computed when `device != 'cpu'` (GPU required). Neither condition is met. The custom class `metrics.bert_score_metric.BertScoreMetric` exists in the repo and is the correct implementation — it computes `bert_score` using `bert-base-uncased` without GPU dependency.

## Root Cause

The registry `helm_class` for `bert_score` points to HELM's `SummarizationMetric`, but that class: (1) requires a mandatory `task` constructor argument — calling with `args={}` raises `TypeError` at init, and (2) only computes BERTScore conditionally when a GPU device is specified. The repo provides a purpose-built custom class `metrics.bert_score_metric.BertScoreMetric` that correctly produces the `bert_score` stat using CPU. The run_spec must replace `SummarizationMetric` with this custom class.

## Proposed Fix

In `run_specs/ocw_connections_run_specs.py`, replace:
```python
MetricSpec(
    class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric",
    args={},
)
```
with:
```python
MetricSpec(
    class_name="metrics.bert_score_metric.BertScoreMetric",
    args={},
)
```
