# aaar — Fixes Summary

## Attempt 6 (2026-04-16)

**Problem:** `sentence_bert_*` metrics still missing — stale stats.json from before the SentenceBertMetric fix was blocking re-evaluation.

**Root Cause:** `init_eval.sh` exits early (`[ALREADY DONE]`) when stats.json exists. The corrected run_spec (with `SentenceBertMetric`) was never executed because old stats.json files from prior runs prevented re-evaluation.

**Fix Applied:** Deleted stale stats.json files:
- `benchmark_output/runs/trial/aaar:subtask=experiment_design,model=google_gemini-2.5-flash-lite/stats.json`
- `benchmark_output/runs/trial/aaar:subtask=paper_weakness,model=google_gemini-2.5-flash-lite/stats.json`

No code changes needed — run_spec already correct after Attempt 5.

---

## Attempt 5 (2026-04-16)

**Fix Applied:** Created `metrics/sentence_bert_metric.py` with custom `SentenceBertMetric` (uses `sentence_transformers` `all-mpnet-base-v2`, CPU-compatible). Updated both run_spec functions to use `MetricSpec(class_name="metrics.sentence_bert_metric.SentenceBertMetric", args={"model_name": "all-mpnet-base-v2"})`.

**Why SummarizationMetric was wrong:** (1) requires mandatory `task` arg, crashes with `args={}`; (2) BERTScore disabled on CPU; (3) produces `BERTScore-P/R/F` keys, not `sentence_bert_*` names.

---

## Attempts 1–4 (2026-04-16)

Failed attempts using `helm.benchmark.metrics.summarization_metrics.SummarizationMetric` with `args={}` — crashed at runtime due to missing `task` arg.

## Verification (Attempt 6)
- `python3 -m py_compile run_specs/aaar_run_specs.py` → Syntax OK
- Stale stats.json files deleted — eval will re-run on next `init_eval.sh` invocation
