# Fixes: data_narrative

## Summary

**File changed**: `run_specs/data_narrative_run_specs.py`

**Change**: Added `SummarizationMetric` MetricSpec to produce `bert_score`.

```python
# Before
metric_specs = [
    MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
               args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]}),
]

# After
metric_specs = [
    MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
               args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]}),
    MetricSpec(class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric",
               args={"model_name": "bert-base-uncased"}),
]
```

**Why**: `bert_score` is produced by `SummarizationMetric`; it was simply missing from `metric_specs`. Verified the class exists via `python3 -c "from helm.benchmark.metrics.summarization_metrics import SummarizationMetric"`. Pattern matches existing datasets: `twistlist`, `v_flute`, `splat`.

**No other files changed.**
