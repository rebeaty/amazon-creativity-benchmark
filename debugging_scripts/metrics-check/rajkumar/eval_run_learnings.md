# Eval Run Learnings — rajkumar (Sessions 2026-04-17/18)

## Session Summary
- **Total datasets**: 32
- **PASS (18)**: arastories, arn, banner_request_400, brainteaser, chinese_homophonic_puns, creatset, crowd_vote, cs4, graphrag_bench, graphragbench-wrongone, humor_transfer, liveideabench, llm_discussion, ocw, speak_to_structure, ss_gen, thenextchapter, tinystories
- **FAIL — registry naming bugs (14)**: arena_hard_creative (win_rate), crowdcounter (f1), ii_bench (accuracy), irfl (accuracy), llm4biohypogen (bert_score), memecap (f1), meta4xnli (accuracy), moh_x (accuracy+f1), music_theory_bench (accuracy), newyorker_humor (accuracy), outline_to_story (f1), sdat (semantic_diversity_score), sonnet_or_not_bot (accuracy), yesbut (f1)
- All 32 datasets marked done in debug_assignments_done.json

---

## Infrastructure Fixes Applied (2026-04-18)

### Python 3.14 + dill `_batch_setitems` pickling bug
- Python 3.14's `pickle.py save_dict` calls `self._batch_setitems(obj.items(), obj)` with 2 positional args
- `datasets/utils/_dill.py` defined `_batch_setitems(self, items)` with only 1 arg → TypeError on every HF dataset load
- **Fix**: patched `.venv/lib/python3.14/site-packages/datasets/utils/_dill.py` line ~72 to add `obj=None` default:
  ```python
  def _batch_setitems(self, items, obj=None):
  ```
- Also added `--disable-cache` flag to avoid stale cache pickle errors

### HELM tokenizer override for Gemini API models
- HELM uses `google/gemma-2b` as proxy tokenizer for Gemini models — this is a gated HF model requiring auth
- **Fix**: created `prod_env/model_deployments.yaml` overriding the deployment config:
  ```yaml
  model_deployments:
    - name: google/gemini-2.5-flash-lite
      tokenizer_name: huggingface/gpt2
      ...
  ```
- Use `tokenizer_name: huggingface/gpt2` (NOT `gpt2` — must be fully namespaced)

### API keys in .env.local
- `source .env.local` to load `GOOGLE_API_KEY` and `ANTHROPIC_API_KEY` before running evals

---

## Critical HELM-Level Findings (⚠ Tell Vijeta)

### 1. `accuracy` metric name doesn't exist in HELM
- Registry uses `helm_class: "helm.benchmark.metrics.basic_metrics.BasicMetric"` for `accuracy`
- **`BasicMetric` class does NOT exist** in this HELM version
- **`accuracy` is NOT a valid metric name** in `compute_reference_metrics` or `BasicGenerationMetric`
- The correct HELM metric for MCQ accuracy is `exact_match` (produced by `BasicGenerationMetric(names=["exact_match"])`)
- **Datasets affected**: ii_bench, irfl, meta4xnli, sonnet_or_not_bot, music_theory_bench, moh_x, newyorker_humor
- **Fix needed in registry**: `accuracy` → `exact_match`, `BasicMetric` → `BasicGenerationMetric`

### 2. `f1` metric name doesn't exist in HELM
- Registry uses metric name `f1` but `BasicGenerationMetric`/`compute_reference_metrics` produces `f1_score`
- **Datasets affected**: crowdcounter, memecap, moh_x, outline_to_story, yesbut
- **Fix needed in registry**: `f1` → `f1_score`

### 3. `compute_reference_metrics` cannot be used as MetricSpec `class_name`
- `compute_reference_metrics` is a plain function, not a class
- Using it as `MetricSpec(class_name="...compute_reference_metrics", args={})` crashes at eval time
- **The correct approach**: use `BasicGenerationMetric(names=["exact_match", "f1_score", "rouge_l", "bleu_4", ...])`
- Registry should document `BasicGenerationMetric` as the class for all these metrics

### 4. `SummarizationMetric` and `DisinformationMetric` require optional packages
- `SummarizationMetric` requires `bert_score` package: `pip install "crfm-helm[summarization]"`
- `DisinformationMetric` requires `sacrebleu` package: `pip install "crfm-helm[metrics]"`
- Both packages missing from local venv; server must have them installed

---

## Dataset-Level Learnings

### Systemic fix: `scenarios_new.` → `scenarios.` in all run_specs
- All 313 run_spec files referenced `scenarios_new.*` after directory was renamed to `scenarios/`
- Fixed via bulk sed: `sed -i '' 's/scenarios_new\./scenarios./g' run_specs/*.py`

### arena_hard_creative
- JSONL parser bug: line 102 has unescaped literal newline causing cascade failure. Fixed by resetting buffer.
- Also had `scenarios_new.` path bug (fixed systemically above).

### brainteaser
- Had `MultipleChoiceClassificationMetric` (wrong); changed to `BasicGenerationMetric(names=["exact_match"])`.

### MCQ accuracy datasets (ii_bench, irfl, meta4xnli, sonnet_or_not_bot, music_theory_bench, newyorker_humor, moh_x)
- All had `MultipleChoiceClassificationMetric`; fixed to `BasicGenerationMetric(names=["exact_match"])`
- Registry naming must be fixed by Vijeta before metrics_check can pass

### f1/f1_score naming (crowdcounter, memecap, outline_to_story, yesbut)
- Already had correct metrics as `f1_score`; only registry naming is wrong

### llm_discussion
- Added `DisinformationMetric` for `self_bleu`; needs sacrebleu on server

### llm4biohypogen
- Added `SummarizationMetric` for `bert_score`; needs bert_score on server

### sdat
- `semantic_diversity_score` requires external embedding model — Pattern D, unimplementable

---

## Quick Reference: Correct MetricSpec Patterns

| Goal | Correct MetricSpec |
|------|-------------------|
| `exact_match`, `f1_score`, `rouge_l`, `bleu_4` | `BasicGenerationMetric(names=["exact_match", "f1_score", "rouge_l", "bleu_4"])` |
| MCQ "accuracy" | `BasicGenerationMetric(names=["exact_match"])` |
| `self_bleu` | `DisinformationMetric()` (needs sacrebleu) |
| `distinct_1` | `DistinctNMetric(n=1)` |
| `distinct_2` | `DistinctNMetric(n=2)` |
| `bert_score` | `SummarizationMetric(task=..., language=..., normalize_by_length=...)` (needs bert_score) |
| LLM judge metrics | `GenericLLMJudgeMetric` + `GenericLLMJudgeAnnotator` |
