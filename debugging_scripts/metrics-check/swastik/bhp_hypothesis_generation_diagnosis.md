# bhp_hypothesis_generation — Metrics Diagnosis (Attempt 3)

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| bleu_4 | formula_based | true | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics |
| rouge_l | formula_based | true | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics |
| bert_score | model_based | true | helm.benchmark.metrics.summarization_metrics.SummarizationMetric |

## Current Run Spec Metrics (after attempt 2)
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]})`
- `MetricSpec(class_name="metrics.sentence_bert_metric.SentenceBertMetric", args={})`

## Actual Stats.json Metrics (m2)
- bleu_1, bleu_4, exact_match, f1_score, quasi_exact_match, rouge_l (from BasicGenerationMetric)
- sentence_bert_f1, sentence_bert_precision, sentence_bert_recall (from SentenceBertMetric)

## Missing Metrics

### bert_score (still missing after attempt 2)
Attempt 2 added `SummarizationMetric`, but that class produces `BERTScore-P`, `BERTScore-R`, `BERTScore-F` (not `bert_score`) and requires a GPU — it sets `compute_bertscore = False` on CPU. Also, attempt 2 appears to have replaced `SummarizationMetric` with `SentenceBertMetric` in the final run_spec, which produces `sentence_bert_*` metrics instead.

The correct class is `metrics.bert_score_metric.BertScoreMetric` — a custom class in this repo that produces exactly `bert_score` using `bert-base-uncased` via the `bert_score` library. This class was confirmed to import successfully.

Note: the registry's `helm_class` for `bert_score` (`SummarizationMetric`) is incorrect — flagging for Vijeta.

## Root Cause
The registry `helm_class` for `bert_score` points to `SummarizationMetric`, which does NOT produce a stat named `bert_score`. The actual producing class `metrics.bert_score_metric.BertScoreMetric` exists in the repo but was never wired into the run_spec. Prior attempts used wrong classes.

## Proposed Fix
Replace `SentenceBertMetric` with `BertScoreMetric` in the metric_specs of `run_specs/bhp_hypothesis_generation_run_specs.py`:
```python
MetricSpec(class_name="metrics.bert_score_metric.BertScoreMetric", args={})
```
