# bhp_hypothesis_generation — Fixes Summary

## Missing Metric: bert_score (Attempt 3)

**Root cause:** Registry's `helm_class` for `bert_score` points to `SummarizationMetric`, which produces `BERTScore-P/R/F` (not `bert_score`) and requires GPU. Attempt 2 also tried `SentenceBertMetric`, which produces `sentence_bert_*` — still wrong. The correct class is the custom `metrics.bert_score_metric.BertScoreMetric` that produces exactly `bert_score`.

**Fix applied (attempt 3):** Replaced `SentenceBertMetric` with `BertScoreMetric` in `run_specs/bhp_hypothesis_generation_run_specs.py`:
```python
# Before:
MetricSpec(class_name="metrics.sentence_bert_metric.SentenceBertMetric", args={})

# After:
MetricSpec(class_name="metrics.bert_score_metric.BertScoreMetric", args={})
```

**Note:** Registry's `helm_class` for `bert_score` (`SummarizationMetric`) appears incorrect — flagged for Vijeta to review in `registry_metrics.yaml`.

**Verification:** `python3 -m py_compile` passes.
