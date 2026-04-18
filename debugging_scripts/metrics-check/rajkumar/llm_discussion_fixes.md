## llm_discussion — Fix Applied (Attempt 2, 2026-04-18)
- **Root cause**: `BasicGenerationMetric` doesn't produce `self_bleu`. Needed `DisinformationMetric(name="self_bleu")`.
- **Files changed**: `run_specs/llm_discussion_run_specs.py`
- **Change summary**: Replaced `BasicGenerationMetric` with `DisinformationMetric(name="self_bleu")`. Kept `DistinctNMetric(n=1)` and `DistinctNMetric(n=2)` unchanged.
- **Additional fixes found during eval**:
  - `DisinformationMetric.__init__` requires `name` arg (not `args={}`) — added `args={"name": "self_bleu"}`
  - HELM uses `google/gemma-2b` as proxy tokenizer for Gemini (gated HF model). Created `prod_env/model_deployments.yaml` override using `huggingface/gpt2` tokenizer instead.
- **Result**: metrics_check PASS — all 3 metrics (`self_bleu`, `distinct_1`, `distinct_2`) confirmed present.
