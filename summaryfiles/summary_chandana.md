# Debugging Session Notes — Chandana x Claude
**Date:** 2026-04-03
**Model under evaluation:** `google/gemini-2.5-flash-lite`
**Suite:** `trial` (10 instances per dataset)
**Branch:** `debug/chandana`

---

## Environment Setup Issues (One-Time Fixes)

### 1. Run entry separator: comma vs colon (ALL 159 eval scripts)
**Error:** `ValueError: Unknown run spec name: aidanbench,model=google/gemini-2.5-flash-lite`

HELM's `parse_object_spec` uses `:` as the key-value separator, not `,`.

```bash
# WRONG (entire string treated as class_name, args={})
helm-run --run-entries "aidanbench,model=google/gemini-2.5-flash-lite"

# CORRECT
helm-run --run-entries "aidanbench:model=google/gemini-2.5-flash-lite"
```

**Fix:** Bulk sed across all 159 `eval_scripts/*.sh`:
```bash
for f in eval_scripts/*.sh; do sed -i '' 's|,model=${MODEL}|:model=${MODEL}|g' "$f"; done
```

### 2. Entry point registration (ALL run specs)
**Error:** `ValueError: Unknown run spec name: aidanbench`

HELM discovers run spec functions via Python entry points. The project had no entry points registered.

**Fix (3 steps):**
1. Added `[project.entry-points.helm]` to `pyproject.toml`:
   ```toml
   [project.entry-points.helm]
   creativity_run_specs = "run_specs"
   ```
2. Created `run_specs/__init__.py` that auto-imports all submodules:
   ```python
   import importlib, pkgutil
   import run_specs as _pkg
   for _, name, _ in pkgutil.iter_modules(_pkg.__path__, _pkg.__name__ + "."):
       importlib.import_module(name)
   ```
3. Added missing packages to `pyproject.toml`:
   ```toml
   include = ["scenarios*", "scenarios_new*", "run_specs*", "llm_judge*", "metrics*"]
   ```
4. Ran `pip install -e .` to register.

**Result:** All 161 run specs registered in one shot.

### 3. Gemini tokenizer gating (`google/gemma-2b`)
**Error:** `403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings`

HELM maps all Google models to the `google/gemma-2b` tokenizer, which is gated on HuggingFace.

**Fix:** Pre-downloaded the tokenizer with an explicit token so it caches locally:
```python
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('google/gemma-2b', token='hf_...')
```
After this, HELM loads it from cache without needing gated access.

### 4. Missing Google client dependency
**Error:** `Optional dependency google is not installed`

**Fix:** `pip install "crfm-helm[google]"`

### 5. Missing summarization dependency
**Error:** `No module named 'bert_score'`

**Fix:** `pip install "crfm-helm[summarization]"`

**Warning:** This downgraded PyTorch to 1.13.1. Had to restore with `pip install "torch>=2.1" torchvision`.

### 6. API keys not loaded
**Error:** Various auth failures for Gemini API.

`.env.local` contains `GEMINI_API_KEY` but is not auto-sourced. Must run:
```bash
source .env.local
```
before any eval script.

### 7. Missing `metrics/` package
**Error:** `ModuleNotFoundError: No module named 'metrics'`

The `metrics/` directory had no `__init__.py` and wasn't in `pyproject.toml` includes.

**Fix:** Created `metrics/__init__.py` and added `"metrics*"` to packages list.

---

## Systemic Run Spec Bugs (Affected Many Datasets)

### 1. `compute_reference_metrics` used as a class (~74 files)
**Error:** `TypeError: compute_reference_metrics() missing 4 required positional arguments`

`MetricSpec` pointed to a bare function instead of a `Metric` class.

**Fix:** Replaced with `get_exact_match_metric_specs()` from `helm.benchmark.metrics.common_metric_specs`. This returns `[BasicGenerationMetric, BasicReferenceMetric, InstancesPerSplitMetric]`.

**Pitfall encountered:** The bulk fix left orphan `]` brackets and missing imports in some files. Required a second cleanup pass.

### 2. `BasicMetric` doesn't exist (~37 files)
**Error:** `AttributeError: module 'helm.benchmark.metrics.basic_metrics' has no attribute 'BasicMetric'`

The correct classes are `BasicGenerationMetric` and `BasicReferenceMetric`.

**Fix:** Same as above — replaced with `get_exact_match_metric_specs()`.

**Pitfall encountered:** Some files had `BasicMetric` inside a `+ [...]` appended list (from the first fix). Required a third cleanup pass to remove the remaining 12 occurrences.

### 3. Missing `MetricSpec` import (~38 files)
**Error:** `NameError: name 'MetricSpec' is not defined`

When replacing `MetricSpec(class_name="...BasicMetric"...)` with `get_exact_match_metric_specs()`, the script also removed the `MetricSpec` import — but some files still had other `MetricSpec` usages (e.g., for LLM judge metrics, SummarizationMetric).

**Fix:** Re-added `from helm.benchmark.metrics.metric import MetricSpec` to all files that still use `MetricSpec(`.

### 4. Missing `get_exact_match_metric_specs` import (~25 files)
**Error:** `NameError: name 'get_exact_match_metric_specs' is not defined`

The bulk replacement script didn't add the import for all files.

**Fix:** Added import to all files using the function.

### 5. `DisinformationMetric` missing `name` arg (2 files)
**Error:** `TypeError: DisinformationMetric.__init__() missing 1 required positional argument: 'name'`

**Fix:** Added `args={"name": "self_bleu"}` (see aidanbench_run_specs.py for correct usage).
**Files:** `llm_discussion_run_specs.py`, `amuse_chord_generation_run_specs.py`

### 6. `DistinctNMetric` missing `n` arg (3 files)
**Error:** `TypeError: DistinctNMetric.__init__() missing 1 required positional argument: 'n'`

**Fix:** Added `args={"n": 2}`.
**Files:** `llm_discussion_run_specs.py`, `dialogue_diversity_run_specs.py`, `diverse_not_short_run_specs.py`

### 7. `SummarizationMetric` wrong args (9 files)
**Error:** `TypeError: SummarizationMetric.__init__() got an unexpected keyword argument 'model_name'`

The correct signature is `SummarizationMetric(task, language, device, bertscore_model, ...)`. The `model_name` kwarg doesn't exist.

**Fix:** Replaced `args={"model_name": "bert-base-uncased"}` with `args={"task": "summarization"}`.

**Note:** `SummarizationMetric` also requires gold references with `CORRECT_TAG`. For tasks without proper gold references, it's better to remove this metric entirely (as done for `ocw_connections`).

---

## Systemic Scenario Bugs

### 1. Missing `CORRECT_TAG` on references
**Error:** `assert len(golds) > 0` in `BasicReferenceMetric`

HELM filters references to only those tagged with `CORRECT_TAG`. If `tags=[]`, metrics crash.

**Fix pattern:**
```python
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Output, Reference
Reference(output=Output(text=answer), tags=[CORRECT_TAG])
```

**Files fixed:** `ocw_connections_scenario.py`, `pollux_creativity_scenario.py`, `deep_math_scenario.py`

### 2. `Reference(output=string)` instead of `Reference(output=Output(text=string))`
**Error:** `AttributeError: 'str' object has no attribute 'text'`

`Reference.output` must be an `Output` object, not a raw string.

**Fix:** Wrap in `Output(text=...)` and import `Output`.
**Files fixed:** `pollux_creativity_scenario.py`

### 3. Frozen dataclass mutation
**Error:** `dataclasses.FrozenInstanceError: cannot assign to field 'id'`

`Instance` is a frozen dataclass. Cannot assign `instance.id = ...` after creation.

**Fix:** Pass `id` and `extra_data` as constructor arguments:
```python
Instance(input=..., references=..., split=..., id="...", extra_data={...})
```
**Files fixed:** `pollux_creativity_scenario.py`

### 4. `ensure_file_downloaded` returns `None`
**Error:** `TypeError: expected str, bytes or os.PathLike object, not NoneType`

In this HELM version, `ensure_file_downloaded` has no return value. Must use the `target_path` directly.

**Correct pattern:**
```python
target = os.path.join(output_path, "filename.md")
ensure_file_downloaded(source_url=url, target_path=target)
with open(target, "r") as f:  # use target, not return value
```
**Files fixed:** `deep_math_scenario.py`

### 5. Non-ASCII URLs
**Error:** `UnicodeEncodeError: 'ascii' codec can't encode characters`

GitHub raw URLs with Chinese filenames must be percent-encoded.

**Fix:**
```python
# WRONG
"https://...main/data/78道证明题.md"
# CORRECT
"https://...main/data/78%E9%81%93%E8%AF%81%E6%98%8E%E9%A2%98.md"
```
**Files fixed:** `deep_math_scenario.py`

### 6. Wrong GitHub directory path
**Error:** `HTTP Error 404: Not Found`

The URL used `DeepMath-Creative-data/` but the actual repo path is `DeepMath-Creative/datasets/`.

**Fix:** Verified with GitHub API and corrected the path.
**Files fixed:** `deep_math_scenario.py`

### 7. `None` values in dataset fields
**Error:** `TypeError: replace() argument 2 must be str, not None`

Some dataset rows have `None` for optional fields like `title` or `abstract`.

**Fix:** Use `.get()` with `or ""` fallback:
```python
title = item.get('title') or ""
```
**Files fixed:** `grapheval_review_advisor_scenario.py`

---

## Gemini API Constraints

### `candidate_count` max is 8
**Error:** `400 INVALID_ARGUMENT: candidate_count must be in the range [1, 8]`

AidanBench requested `num_outputs=30` for self-BLEU diversity scoring. Gemini API caps at 8.

**Fix:** Reduced `num_outputs` to 8 in `aidanbench_run_specs.py`. This changes evaluation semantics but is necessary for the API.

---

## Datasets Marked BLOCKED (Data Access Issues)

| Dataset | Reason |
|---|---|
| `llm_srbench` | Gated HuggingFace dataset `nnheui/llm-srbench` — requires approved access |
| `memecap` | Requires manual image download from Kaggle/Google Drive |

---

## Datasets PASSED

| Dataset | Notes |
|---|---|
| `aidanbench` | Reduced num_outputs from 30 to 8 for Gemini |
| `arena_hard_v01` | Clean pass |
| `assocam` (3 subtasks) | Fixed compute_reference_metrics |
| `balderdash` | Fixed BasicMetric + missing import |
| `conceptual_design` | Fixed missing MetricSpec import |
| `deep_math` | Fixed URL encoding + path + ensure_file_downloaded + CORRECT_TAG |
| `grapheval_review_advisor` | Fixed None field values |
| `lcc_metaphor` | Fixed remaining BasicMetric reference |
| `llm_discussion` | Fixed DisinformationMetric + DistinctNMetric args |
| `ocw_connections` | Fixed CORRECT_TAG + removed inappropriate SummarizationMetric |
| `pollux_creativity` | Fixed frozen dataclass + Output wrapping + CORRECT_TAG |

---

## Key Lessons for Future Debugging

1. **Always check `.pyc` cache** after fixing run specs. Stale bytecode can mask fixes. Run:
   ```bash
   find run_specs/__pycache__ -name "*.pyc" -delete
   ```

2. **Bulk fixes need multiple passes.** The first sed/script pass often introduces secondary issues (orphan brackets, missing imports, etc.). Always verify with `ast.parse()` after.

3. **Verify metric constructors before using them.** Check the `__init__` signature:
   ```python
   import inspect
   inspect.signature(MetricClass.__init__)
   ```

4. **HELM's `Reference` requires `Output` objects, not strings.** And references need `CORRECT_TAG` for any metric that computes against gold.

5. **`ensure_file_downloaded` has no return value** in this HELM version. Always construct your own path.

6. **Gemini API limits:** `candidate_count` max 8, and all Google models use gated `gemma-2b` tokenizer.

7. **Source `.env.local` before every eval run** — it's not auto-loaded.

---

## Quick Debugging Checklist (Per Dataset)

Before running a dataset, check:

- [ ] Does the scenario import `CORRECT_TAG` and `Output`, and apply them to references?
- [ ] If `references=[]`, does the run spec use LLM judge instead of basic metrics?
- [ ] Does the scenario use `os.path.join(output_path, filename)` for downloads?
- [ ] Is `ensure_file_downloaded` return value being ignored (use your own path)?
- [ ] Does the metric class exist and have correct constructor args?
- [ ] For Gemini: is `num_outputs` <= 8?
- [ ] Are all URLs ASCII-safe (percent-encoded if needed)?
- [ ] Is `Instance` constructed with all fields in the constructor (frozen dataclass)?

---

## Running a Dataset

```bash
# Activate env and load keys
conda activate creativity-bench
source .env.local

# Run one dataset
./eval_scripts/DATASET_NAME.sh "google/gemini-2.5-flash-lite" "trial" 10
```
