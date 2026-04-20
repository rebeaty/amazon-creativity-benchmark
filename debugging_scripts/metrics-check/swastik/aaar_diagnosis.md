# aaar — Metrics Diagnosis (Attempt 2 / orchestrator)

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| `sentence_bert_f1` | model_based | true | `helm.benchmark.metrics.summarization_metrics.SummarizationMetric` (registry) → actually uses custom `metrics.sentence_bert_metric.SentenceBertMetric` |
| `sentence_bert_precision` | model_based | true | same |
| `sentence_bert_recall` | model_based | true | same |
| `recall_gt_entail_score` | llm_judge | false | null |
| `precision_pred_entail_score` | llm_judge | false | null |

## Current Run Spec Metrics
Both `aaar_experiment_design` and `aaar_paper_weakness` have:
- `MetricSpec(BasicGenerationMetric, names=["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"])`
- `MetricSpec(GenericLLMJudgeMetric, metric_name="recall_gt_entail_score")` ✅
- `MetricSpec(GenericLLMJudgeMetric, metric_name="precision_pred_entail_score")` ✅
- Two `AnnotatorSpec(GenericLLMJudgeAnnotator, ...)` ✅
- **`SentenceBertMetric` is NOT present** ❌

## Actual Stats.json Metrics (m2)
`["bleu_1", "bleu_4", "exact_match", "f1_score", "precision_pred_entail_score", "quasi_exact_match", "recall_gt_entail_score", "rouge_l"]`

## Missing Metrics
- `sentence_bert_f1` — `SentenceBertMetric` MetricSpec absent from both run_spec functions
- `sentence_bert_precision` — same
- `sentence_bert_recall` — same

## Root Cause
Note: registry `helm_class` points to `SummarizationMetric`, but that class does NOT produce `sentence_bert_*` metrics (it produces BERTScore and rouge). The correct implementation is a custom class `metrics/sentence_bert_metric.py` (`SentenceBertMetric`) that was created in a prior fix attempt and is confirmed importable. However, the MetricSpec wiring it into the run_spec is missing in the current version of `run_specs/aaar_run_specs.py` — both subtask functions lack the MetricSpec entirely.

## Proposed Fix
Add to `metric_specs` in **both** `get_aaar_experiment_design_spec()` and `get_aaar_paper_weakness_spec()`:
```python
MetricSpec(class_name="metrics.sentence_bert_metric.SentenceBertMetric", args={"model_name": "all-mpnet-base-v2"})
```
No new imports needed.
