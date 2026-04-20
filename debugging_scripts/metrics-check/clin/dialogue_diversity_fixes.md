# Fixes: dialogue_diversity — `semantic_diversity`

## File Changed

`run_specs/dialogue_diversity_run_specs.py`

## What Was Fixed

Added `MetricSpec` for `SemanticDiversityMetric` to `metric_specs`:

```python
MetricSpec(
    class_name="metrics.semantic_diversity_metric.SemanticDiversityMetric",
    args={"model_name": "all-mpnet-base-v2", "task": "cwt"},
),
```

## Why

The `semantic_diversity` metric was expected per `registry_metrics.yaml` but was absent from the run spec. The class `SemanticDiversityMetric` already existed at `metrics/semantic_diversity_metric.py` — it just needed to be wired in.

- `model_name="all-mpnet-base-v2"`: matches the registry entry (`metric_model: "all-mpnet-base-v2 (SentenceTransformers)"`)
- `task="cwt"`: dialogue responses are short sentences, so sentence-level segmentation (`_cwt_response_to_sentences`) is appropriate (not "dat" which is for comma-separated word lists)
