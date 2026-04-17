# scimon — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| bleu_4 | formula_based | true | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics |
| rouge_l | formula_based | true | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics |
| bert_score | model_based | true | helm.benchmark.metrics.summarization_metrics.SummarizationMetric |

## Current Run Spec Metrics
- `BasicGenerationMetric` with `names: ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]`
- `metrics.bert_score_metric.BertScoreMetric` with `args={}`
- `annotators=None`

## Actual Stats.json Metrics (m2)
- bleu_1, bleu_4, exact_match, f1_score, quasi_exact_match, rouge_l
- `bert_score` is MISSING despite the MetricSpec being present

## Missing Metrics
- **bert_score**: The `BertScoreMetric` MetricSpec exists but crashes at runtime with:
  `NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty()`
  This causes the whole eval run to fail and `bert_score` is never written to stats.json.

## Root Cause
Since transformers PR #36963, `init_empty_weights` is now **native to transformers** (copied from accelerate into `transformers.integrations.accelerate`) and models are **always** loaded on meta device regardless of whether accelerate is installed. The previous patch (`_modeling_utils.is_accelerate_available = lambda: False`) was targeting the wrong function — `is_accelerate_available` is imported as a local reference inside `modeling_utils.py` at module load time, so replacing the module attribute has no effect. Additionally, the meta device code path now comes from `transformers.integrations.accelerate.init_empty_weights` (which is also bound into `modeling_utils` as a local name at import time), not from accelerate. As a result, `bert_score` calls `model.to("cpu")` on a meta-device model, which raises `NotImplementedError`.

## Proposed Fix
Replace the ineffective `is_accelerate_available` patch with a no-op `init_empty_weights` context manager patched onto **both** `transformers.modeling_utils.init_empty_weights` and `transformers.integrations.accelerate.init_empty_weights`. This prevents the meta device initialization path so `bert_score` can safely call `model.to(device)`.
