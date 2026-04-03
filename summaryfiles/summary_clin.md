# HELM Creativity Benchmark Debugging Session â€” Insights & Playbook

**Date:** 2026-04-03
**Model under test:** google/gemini-2.0-flash-lite (via Google API)
**Environment:** Google Colab (CPU-only, no GPU)
**HELM version:** crfm-helm 0.5.14

---

## Common Problems Encountered (Ranked by Frequency)

### 1. `compute_reference_metrics` Is a Function, Not a Class (69+ run specs)

**Symptom:**
```
TypeError: compute_reference_metrics() missing 4 required positional arguments: 'names', 'adapter_spec', 'request_state', and 'metric_service'
```

**Root cause:** `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` is a plain function, not a `Metric` subclass. HELM's `create_object()` tries to instantiate it as a class.

**Fix:** Replace with `helm.benchmark.metrics.basic_metrics.BasicReferenceMetric` (no args needed). This works for MCQ and tasks with CORRECT-tagged references.

**Caveat:** For open-ended generation tasks WITHOUT gold references, `BasicReferenceMetric` will `assert len(golds) > 0` and crash. Those tasks need an LLM judge instead.

**Applied via:** `sed -i` bulk replacement across all 69+ files.

---

### 2. `BasicMetric` Does Not Exist (36 run specs)

**Symptom:**
```
AttributeError: module 'helm.benchmark.metrics.basic_metrics' has no attribute 'BasicMetric'
```

**Root cause:** The class is called `BasicReferenceMetric` (for reference-based) or `BasicGenerationMetric` (for generation metrics like ROUGE). There is no `BasicMetric`.

**Fix:** Replace `BasicMetric` with `BasicReferenceMetric` (args={}).

**Applied via:** `sed -i` bulk replacement.

---

### 3. Eval Script Formatting Bugs (All 155+ scripts)

**Symptom:**
```
ValueError: Unknown run spec name: aaar_experiment_design,model=google/gemini-2.0-flash-lite
```

**Root cause (two issues):**
1. **Comma instead of colon:** Run entries used `name,model=X` but HELM's `parse_object_spec()` expects `name:key=value` with a colon separator.
2. **Missing `--plugins` flag:** Custom run spec functions are not in `helm.benchmark.run_specs`, so HELM can't discover them without `--plugins run_specs.DATASET_run_specs`.

**Fix:** Applied to all 155+ eval scripts:
- `s/,model=${MODEL}/\:model=${MODEL}/`
- Added `--plugins run_specs.DATASET_run_specs` to the `CMD=` line

---

### 4. Missing `__init__.py` and Package Registration

**Symptom:**
```
ModuleNotFoundError: No module named 'run_specs'
ModuleNotFoundError: No module named 'metrics'
```

**Root cause:** `pyproject.toml` only included `scenarios*` in the package find directive. `run_specs/`, `metrics/`, and `llm_judge/` directories were not installed as packages.

**Fix:**
1. Created `__init__.py` in `run_specs/`, `metrics/`
2. Updated `pyproject.toml`: `include = ["scenarios*", "run_specs*", "llm_judge*", "metrics*"]`
3. Ran `pip install -e .`

---

### 5. OpenAI LLM Judge Failures (All judge-based benchmarks)

**Symptom:** Annotations silently produce no scores; metrics default to 0.0.

**Root cause:** Many run specs use `openai/gpt-4`, `openai/gpt-4-turbo`, or `openai/gpt-4-1106-preview` as the judge model. Without an `OPENAI_API_KEY`, annotations fail silently.

**Fix:** Replace judge model with `google/gemini-2.0-flash-lite` in all annotator specs.

**Benchmarks fixed:** AAAR, AraStories, Arena Hard Creative, BHP Hypothesis Generation, CrowdCounter, Data Narrative, Dialogue Diversity, Future Ideas.

---

### 6. LLM Judge Metric Reading Wrong Key (GenericLLMJudgeMetric)

**Symptom:** Annotations exist with real scores but `stats.json` shows 0.0.

**Root cause:** `GenericLLMJudgeMetric.evaluate_generation()` read `annotations.get(metric_name)` but the actual structure is `annotations["generic_llm_judge"][metric_name]`.

**Fix (llm_judge/generic_llm_judge_metric.py):**
```python
# Before:
score = float(annotations.get(self.metric_name, 0))

# After:
judge_annotations = annotations.get("generic_llm_judge", {})
score = float(judge_annotations.get(self.metric_name, 0))
```

**This was a critical fix** â€” without it, ALL LLM judge scores are 0.

---

### 7. LLM Judge Annotator Reference Access Bug

**Symptom:**
```
AttributeError: 'dict' object has no attribute 'text'
```

**Root cause:** `generic_llm_judge_annotator.py` line 47 accesses `references[0].output.text` but `output` can be a dict when deserialized from cache.

**Fix (llm_judge/generic_llm_judge_annotator.py):**
```python
ref_output = request_state.instance.references[0].output
if hasattr(ref_output, 'text'):
    reference_text = ref_output.text
elif isinstance(ref_output, dict):
    reference_text = ref_output.get('text', '')
```

---

### 8. Gemini API: candidate_count Max 8

**Symptom:**
```
ClientError: 400 INVALID_ARGUMENT. candidate_count must be in the range [1, 8]
```

**Root cause:** Amuse Chord Generation had `num_outputs=30`. Gemini API caps `candidate_count` at 8.

**Fix:** Set `num_outputs=8` (Gemini max) or `num_outputs=1` if the prompt asks the model to generate multiple items in one response.

**Note:** This will keep retrying forever â€” HELM doesn't distinguish 400 errors from transient failures. Must kill the cell manually.

---

### 9. Missing Python Dependencies

| Dependency | Required by | Fix |
|---|---|---|
| `bert-score` | `SummarizationMetric` | **Cannot install** (broken metadata with pip>=24.1). Remove metric from run specs. |
| `sacrebleu` | `DisinformationMetric` | `pip install "crfm-helm[metrics]"` or remove metric. |

**Strategy:** Remove `SummarizationMetric` and `DisinformationMetric` from all run specs that use them. They're not the primary metrics for these creativity benchmarks anyway.

---

### 10. SentenceTransformer Meta Tensor Crash

**Symptom:**
```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

**Root cause:** `torch 2.10.0` + `accelerate 1.13.0` loads safetensors as meta tensors by default. `SentenceTransformer` then fails on `.to(device)`.

**Attempted fixes that DID NOT WORK:**
- `device="cpu"` parameter
- `model_kwargs={"device_map": None}`
- `torch.device("cpu")` context manager
- `model_kwargs={"low_cpu_mem_usage": False}`
- `os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"`
- Downgrading `sentence-transformers` to 3.3.1

**Status:** UNRESOLVED on CPU-only Colab with torch 2.10. This blocks `CreativityScoreMetric` (used by CDAT, SDAT, DAT). May need to switch to a GPU runtime or use a different embedding approach.

---

### 11. HuggingFace Rate Limits (429 Too Many Requests)

**Symptom:** `429 Client Error: Too Many Requests` during `snapshot_download()`.

**Affected datasets:** AAAR (16k+ files), ConvBench (2k+ images), and any HF dataset with many files.

**Mitigations:**
- Set `HF_TOKEN` for authenticated access (higher limits)
- Wait 5 minutes and retry (partial downloads resume from cache)
- Upgrade to HF Pro account
- Use smaller datasets for initial debugging

---

### 12. Gated HuggingFace Repos

**Symptom:** `403 Forbidden` / `GatedRepoError` for `google/gemma-2b` tokenizer.

**Root cause:** HELM uses `google/gemma-2b` as the tokenizer for unknown Gemini models. Gemma requires license acceptance at huggingface.co.

**Fix:** Accept the Gemma license on HuggingFace, or use `gemini-2.0-flash-lite` (which worked) instead of `gemini-2.5-flash-lite` (which triggered this).

---

### 13. Google API Key Not Found in Colab Subprocesses

**Symptom:**
```
ValueError: No API key was provided.
```

**Root cause:** `os.environ["GOOGLE_API_KEY"]` set in a notebook cell is inherited by `!` commands in the **same cell** only. If set in a different cell, the subprocess may not see it.

**Fix:** Always set env vars in the **same cell** as the `!` command:
```python
from google.colab import userdata
import os
os.environ["GOOGLE_API_KEY"] = userdata.get("GEM_KEY_LAB")
!cd /content/amazon-creativity-benchmark && ./eval_scripts/foo.sh ...
```

---

### 14. Scenario-Specific Bugs

| Dataset | Bug | Fix |
|---|---|---|
| **Arena Hard Creative** | Upstream JSONL has truncated line (line 102) | Added `try/except json.JSONDecodeError: continue` |
| **CSD100** | Passes PIL Image object instead of file path to `MediaObject` | Save PIL images to disk, pass file path |
| **KiVA** | `_download_and_extract_data` returns wrong path on cache hit | Fixed to always return `data_dir/single_image` |
| **Future Ideas** | Missing required `domain` arg in ScenarioSpec | Added `args={"domain": "computer"}` |
| **Amuse Chord** | `JSDMetric` missing `n` arg | Added `args={"n": 1}` |
| **Dialogue Diversity** | `DistinctNMetric` missing `n` arg | Added `args={"n": 2}` |

---

## Key Architectural Insights

### HELM Plugin System
- Custom run specs MUST be registered via `--plugins module.name` CLI flag
- The module must be importable (i.e., installed via pip or on PYTHONPATH)
- The `@run_spec_function("name")` decorator auto-registers at import time

### HELM Object Spec Parsing
- Format is `name:key=value,key=value` (colon separates name from args)
- NOT `name,key=value` (comma is treated as part of the name)

### HELM Metric Classes
- `BasicReferenceMetric` â€” for MCQ and tasks with CORRECT-tagged gold references (no args)
- `BasicGenerationMetric` â€” for generation tasks with gold references; requires `names` list (e.g., `["rouge_1", "rouge_2"]`)
- Custom metrics in `metrics/` â€” need `__init__.py` and package registration

### HELM Annotation Flow
- Annotations are stored under `annotations["annotator_name"][metric_name]`
- The annotator `name` class attribute (e.g., `"generic_llm_judge"`) is the key
- Metrics must look up the correct nested key

---

## Benchmarks Successfully Debugged (with Real Scores)

| Benchmark | Type | Metric | Score | Notes |
|---|---|---|---|---|
| **Arena Hard Creative** | Generation + Judge | win_rate | 4.30/5 | Full pipeline working |
| **AraStories** | Generation + Judge | variety | 3.30/5 | Only 1 of 5 judges ran (last one) |
| **BrainTeaser** | MCQ | exact_match | TBD | Pipeline working |
| **Dialogue Diversity** | Generation + Judge | coherence + distinct-2 | TBD | Pipeline working |
| **BHP Hypothesis Gen** | Generation + Judge | hypothesis_quality | TBD | Pipeline working |
| **CrowdCounter** | Generation + Judge | counterspeech_quality | TBD | Pipeline working |
| **Data Narrative** | Generation + Judge | narrative_quality | TBD | Pipeline working |

---

## Checklist for Debugging a New Dataset

1. **Check eval script:** `:` not `,` in run entry? `--plugins` present?
2. **Check run spec metrics:** No `compute_reference_metrics`? No `BasicMetric`? No `SummarizationMetric`?
3. **Check metric args:** `JSDMetric` needs `n`, `DistinctNMetric` needs `n`, `BasicGenerationMetric` needs `names`
4. **Check judge model:** `openai/*` â†’ `google/gemini-2.0-flash-lite`
5. **Check scenario args:** Any required constructor params missing from `ScenarioSpec.args`?
6. **Check data access:** HF rate limits? Gated repos? Large downloads?
7. **Check `num_outputs`:** Gemini max is 8 candidates per request
8. **Run with `max_instances 10`** first to catch errors quickly
9. **Set `GOOGLE_API_KEY` in same notebook cell** as the `!` command