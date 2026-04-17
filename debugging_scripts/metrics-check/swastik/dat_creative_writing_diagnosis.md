# dat_creative_writing — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| semantic_diversity | model_based | false | null |
| llm_judge_creativity | llm_judge | false | null |

## Current Run Spec Metrics
- MetricSpec: `metrics.semantic_diversity_metric.SemanticDiversityMetric` with `args={"model_name": "all-mpnet-base-v2", "mode": "auto"}`
- MetricSpec: `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric` with `args={"metric_name": "llm_judge_creativity"}`
- AnnotatorSpec: `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator` with `judge_model_name="openai/gpt-4"`, `metric_name="llm_judge_creativity"`

## Actual Stats.json Metrics (m2)
- `llm_judge_creativity` (only)

## Missing Metrics
- **semantic_diversity**: The registry says `type: model_based`, `helm_class: null`, but a custom implementation exists at `metrics/semantic_diversity_metric.py` using SentenceTransformers. The previous diagnosis (attempt 1) incorrectly classified this as Pattern D (unimplementable). The `SemanticDiversityMetric` class was added to the run_spec in a later attempt. The stats.json is stale — it was produced by a trial run that predates the fix.

## Root Cause
The trial run that produced the current stats.json was executed before `SemanticDiversityMetric` was added to `metric_specs`. The run_spec is now correct (both metrics present), the custom class exists at `metrics/semantic_diversity_metric.py`, imports cleanly, and the `all-mpnet-base-v2` model is cached locally at `~/.cache/huggingface/hub`. The eval simply needs to be re-run to produce updated stats.json.

## Proposed Fix
No code changes needed — the run_spec is already correct. Re-running the eval will produce `semantic_diversity` in stats.json.
