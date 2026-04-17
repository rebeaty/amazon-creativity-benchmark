# scimon — Fixes Summary

## Fix Applied
**File:** `run_specs/scimon_run_specs.py`

Added a second `MetricSpec` for `bert_score`:
```python
MetricSpec(class_name="metrics.bert_score_metric.BertScoreMetric", args={})
```

## Why
The registry expects `bert_score` but the run_spec only had `BasicGenerationMetric` which doesn't produce it. The registry `helm_class` (`SummarizationMetric`) produces `BERTScore-P/R/F`, not `bert_score`. The custom class `metrics/bert_score_metric.py` (`BertScoreMetric`) produces exactly the `bert_score` stat and was already present in the repo.

## Verification
- `python3 -m py_compile run_specs/scimon_run_specs.py` → Syntax OK
- `metrics.bert_score_metric.BertScoreMetric` confirmed to emit `Stat(MetricName("bert_score"))`
