---
# Diagnosis: data_narrative (Attempt 8)

## Expected vs Actual Metrics

| Metric | Expected | Actual |
|--------|----------|--------|
| bleu_4 | ✅ | ✅ |
| rouge_l | ✅ | ✅ |
| bert_score | ✅ | ❌ missing |

Extra metrics in actual (not in expected): `bleu_1`, `exact_match`, `f1_score`, `quasi_exact_match` — from `BasicGenerationMetric`, harmless.

## Root Cause (Multi-layer)

### Layer 1: SummarizationMetric cannot be used
The registry specifies `helm_class: "helm.benchmark.metrics.summarization_metrics.SummarizationMetric"` for `bert_score`. However, `SummarizationMetric` requires `summ_eval` which is **not installed**:
```
helm.common.optional_dependencies.OptionalDependencyNotInstalled: Optional dependency summ_eval is not installed.
```
Previous attempts (correctly) switched to a custom `metrics.bert_score_metric.BertScoreMetric`.

### Layer 2: Custom BertScoreMetric was already added — but eval never re-ran
The run spec (`run_specs/data_narrative_run_specs.py`) was already updated to include:
```python
MetricSpec(class_name="metrics.bert_score_metric.BertScoreMetric", args={"model_type": "bert-base-uncased", "device": "cpu"})
```
And `metrics/bert_score_metric.py` exists and imports correctly.

### Layer 3: Stale stats.json blocked re-evaluation
`init_eval.sh` checks: *if stats.json exists → skip, mark done, exit 0*. The stats.json from the **original run** (before BertScoreMetric was added) was never deleted, so `init_eval.sh` kept exiting early and the eval was never re-run with the fixed run spec.

## Fix Applied
1. **No code changes needed** — run_specs and bert_score_metric.py were already correct.
2. **Deleted the stale stats.json** at:
   `benchmark_output/runs/trial/data_narrative:model=google_gemini-2.5-flash-lite/stats.json`
   This forces `init_eval.sh` to re-run the eval using the current run spec with `BertScoreMetric`.

## Expected Outcome After Re-run
The eval will use the cached `scenario_state.json` (predictions already exist) and recompute metrics, producing a fresh `stats.json` containing `bert_score`.
