# ocw_connections — Fixes Summary

## File Modified
`run_specs/ocw_connections_run_specs.py`

## Change Made
Replaced the TODO fallback `BasicGenerationMetric` with two registry-directed MetricSpecs:

**Before:**
```python
metric_specs = [
    MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]}),
]
```

**After:**
```python
metric_specs = [
    MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={}),
    MetricSpec(class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric", args={}),
]
```

## Why
- `rouge_1` was missing because `BasicGenerationMetric` only emits `rouge_l`, not `rouge_1`. The registry specifies `compute_reference_metrics` which produces both `exact_match` and `rouge_1`.
- `bert_score` was missing entirely. The registry specifies `SummarizationMetric` which produces `bert_score` (sentence-BERT based) alongside other summarization metrics.

## Verification
- `python3 -m py_compile run_specs/ocw_connections_run_specs.py` passes (Syntax OK).

## Attempt 2 Note
Attempt 1 fix was not persisted to the file (run_spec still had `BasicGenerationMetric` with `["exact_match", "rouge_1", "rouge_2", "rouge_l"]`). Re-applied same fix in attempt 2. `bert_score` still missing because `BasicGenerationMetric` does not produce it regardless of the names list.
