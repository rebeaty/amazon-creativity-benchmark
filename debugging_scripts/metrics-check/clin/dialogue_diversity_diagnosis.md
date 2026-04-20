# Diagnosis: dialogue_diversity — Missing `semantic_diversity`

## Expected vs Actual Metrics

| Metric | Expected | Actual |
|---|---|---|
| distinct_1 | ✅ | ✅ |
| distinct_2 | ✅ | ✅ |
| semantic_diversity | ✅ | ❌ |
| coherence_score | ✅ | ✅ |

## Root Cause

`semantic_diversity` is missing from `metric_specs` in `run_specs/dialogue_diversity_run_specs.py`.

The metric class `SemanticDiversityMetric` already exists at `metrics/semantic_diversity_metric.py` and imports cleanly (`python3 -c "from metrics.semantic_diversity_metric import SemanticDiversityMetric"` → OK).

The run spec only includes:
- `metrics.distinct_n_metric.DistinctNMetric` (n=1)
- `metrics.distinct_n_metric.DistinctNMetric` (n=2)
- `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric` (coherence_score)

`SemanticDiversityMetric` was never added to `metric_specs`.

Registry entry confirms: `helm_class: null` (no built-in HELM class), uses `all-mpnet-base-v2` SentenceTransformer model.

## Proposed Fix

Add to `metric_specs` in `run_specs/dialogue_diversity_run_specs.py`:

```python
MetricSpec(
    class_name="metrics.semantic_diversity_metric.SemanticDiversityMetric",
    args={"model_name": "all-mpnet-base-v2", "task": "dat"},
),
```
