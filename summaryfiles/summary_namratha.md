# Debugging Session Notes — Namrata × Claude
**Date:** 2026-04-03
**Model under evaluation:** `google/gemini-2.5-flash-lite`
**Suite:** `trial` (10 instances per dataset)
**HELM binary:** `/Users/nammu/miniconda3/envs/creativity-bench/bin/helm-run`

---

## Environment Facts (Mac-specific)

- Conda env: `creativity-bench` at `/Users/nammu/miniconda3/envs/creativity-bench/`
- `helm-run` is NOT on the system PATH — must call with full path or activate the conda env first
- API keys must be exported in the shell session before running; `.env.local` is not auto-sourced
- For Gemini: `GEMINI_API_KEY` (also accepted as `GOOGLE_API_KEY`)
- For LLM-judge runs: `OPENAI_API_KEY` also required (judge model is `openai/gpt-4o`)

---

## Systemic Root Causes (affected many datasets)

### 1. `assert len(golds) > 0` — Missing `CORRECT_TAG` on references

**The most common error in the entire session.** HELM's `compute_reference_metrics` (called by `BasicGenerationMetric` for `rouge_l`, `bleu_4`, `exact_match`) filters references to only those tagged `CORRECT_TAG`. If none are tagged, it asserts and crashes.

**Three sub-patterns:**

| Sub-pattern | Example | Fix |
|---|---|---|
| Reference created with `tags=[]` in an open-ended generation scenario | `futuregen`, `creai_cps`, `metaphor_generation`, etc. | Change `tags=[]` → `tags=[CORRECT_TAG]` and add import |
| Wrong tag used instead of `CORRECT_TAG` | `crowdcounter` used `tags=[cs_type]` | Replace with `tags=[CORRECT_TAG]` |
| Scenario intentionally has `references=[]` (no gold) but run spec uses `BasicGenerationMetric` | `balderdash`, `dat`, `noveltybench`, `sdat` | Switch run spec to LLM judge metric |

**Files fixed (added `CORRECT_TAG`):**
`futuregen`, `bhp_hypothesis_generation`, `creai_cps`, `data_narrative`, `llm4biohypogen`, `memecap`, `metaphor_generation`, `ocw_connections`, `ss_gen`, `story_generation_rocstories`, `twistlist`, `unfun_corpus`, `crowdcounter`

**Files switched to LLM judge:**
`balderdash`, `dat`, `noveltybench`, `sdat`

**How to scan for future occurrences:**
```bash
# Find scenarios with no CORRECT_TAG anywhere
for f in scenarios_new/*.py; do grep -q "CORRECT_TAG" "$f" || echo "NO TAG: $f"; done

# Find run specs using BasicGenerationMetric (need gold references)
grep -l "BasicGenerationMetric" run_specs/*.py
```

---

### 2. Wrong metric class / non-existent metric

Many run specs were generated with broken metric references:

| Wrong class | Correct replacement | Notes |
|---|---|---|
| `compute_reference_metrics` (function) | `BasicGenerationMetric` | Was being used as a class; it's a standalone function |
| `BasicMetric` | `BasicGenerationMetric` | `BasicMetric` does not exist in HELM |
| `CLIPScoreMetric(model_name=...)` | removed entirely | `model_name` kwarg not accepted |
| `CorrelationMetric()` | `CorrelationMetric(correlation_type="pearson")` | `correlation_type` is required |

**Affected ~110 run specs total — fixed via bulk sed/script.**

---

### 3. MCQ tasks using generation metrics

Datasets using `ADAPT_MULTIPLE_CHOICE_JOINT` were assigned `["rouge_l", "bleu_4"]`. These produce single-letter answers ("A", "B", etc.) — BLEU/ROUGE are meaningless and the string comparison is fragile.

**Fix:** Change metric names to `["exact_match"]` for all MCQ/classification tasks.

**Affected ~14 run specs** — identified by adapter method `ADAPT_MULTIPLE_CHOICE_JOINT`.

**`fig_qa` special case:** Uses MCQ adapter. The test split has hidden labels (`has_labels=False`), so all test instances had `references=[]`. Fix was to set `use_validation_as_test=True` in the scenario args so that the labeled validation split is used as the test set.

---

### 4. `ensure_file_downloaded` returns `None` in this HELM version

**Pattern that fails:**
```python
data_path = ensure_file_downloaded(source_url=url, target_path="file.csv")
df = pd.read_csv(data_path)  # data_path is None → ValueError
```

**Correct pattern:**
```python
data_path = os.path.join(output_path, "file.csv")
ensure_file_downloaded(source_url=url, target_path=data_path)
df = pd.read_csv(data_path)  # use your own path, not the return value
```

**Also:** Always use `os.path.join(output_path, filename)` for `target_path`. Relative paths resolve to the process working directory, not the scenario cache dir.

**Fixed in:** `futuregen`

---

### 5. Multimodal scenarios: images must be local file paths

HELM's `GoogleGenAIClient` (used for Gemini) cannot accept:
- PIL `Image` objects in `MediaObject`
- HTTP/HTTPS URLs as `MediaObject.location`

**Required pattern:**
```python
img_path = os.path.join(images_dir, "image.jpg")
img.convert("RGB").save(img_path, format="JPEG")  # save PIL to disk first
MediaObject(content_type="image/jpeg", location=img_path)  # local path only
```

**Also:** Text strings in `MultimediaObject` must be wrapped in `MediaObject`, not passed raw:
```python
# WRONG
MultimediaObject([img_media_obj, "some text"])

# CORRECT
MultimediaObject([
    MediaObject(content_type="image/jpeg", location=img_path),
    MediaObject(content_type="text/plain", text="some text")
])
```

**Fixed in:** `ava_scenario.py`, `creation_mmbench_scenario.py`, `esp_dataset_scenario.py`

---

### 6. Run entry separator: comma vs colon

All 158 eval scripts were generated with the wrong separator:
```bash
# WRONG (model not recognized as a key=value override)
helm-run --run-entries "fig_qa,model=google/gemini-2.5-flash-lite"

# CORRECT
helm-run --run-entries "fig_qa:model=google/gemini-2.5-flash-lite"
```

**Fixed via bulk sed across all `eval_scripts/*.sh`.**

---

### 7. Missing `--max-eval-instances` default

All eval scripts had `MAX_INSTANCES=""` (empty), which caused `helm-run` to error:
```
error: the following arguments are required: -m/--max-eval-instances
```

**Fix:** Changed default to `MAX_INSTANCES="${3:-10}"` in all 158 scripts.

---

## Dataset-Specific Fixes

### `fig_qa`
- **Problem:** Test split has hidden labels → no `CORRECT_TAG` → `assert len(golds) > 0`
- **Fix:** Pass `use_validation_as_test=True` in run spec `ScenarioSpec.args`

### `futuregen`
- **Problem 1:** `ensure_file_downloaded` returns `None` → `pd.read_csv(None)` crashes
- **Problem 2:** References had `tags=[]`, not `CORRECT_TAG`
- **Fix:** Construct `data_path = os.path.join(output_path, filename)` manually; add `CORRECT_TAG`

### `fuxibench`
- **Status:** No bugs found. Data loads cleanly, prompts look correct (Chinese poetry format verified). Fails only due to missing `GEMINI_API_KEY` at inference time.

### `esp_dataset`
- **Problem 1:** GitHub URL used `main` branch but repo only has `master`
- **Problem 2:** COCO image URLs passed directly to `MediaObject` (must be local files)
- **Problem 3:** References missing `CORRECT_TAG`
- **Fix:** Changed URL branch, download images with `urllib.request.urlretrieve`, add `CORRECT_TAG`

### `creation_mmbench`
- **Problem:** PIL `Image` objects passed directly to `MediaObject` instead of file paths
- **Fix:** Save each PIL image to `{images_dir}/{row_idx}_{idx}.jpg`, pass path to `MediaObject`

### `ava` (AVA aesthetic ratings)
- **Problem:** HuggingFace parquet was broken/truncated; also PIL images passed to `MediaObject`
- **Fix:** Rewrote to read images from zip archive by image ID, save locally

### `balderdash`, `dat`, `noveltybench`, `sdat`
- **Problem:** Scenarios produce `references=[]` (no gold) but run specs used `BasicGenerationMetric`
- **Fix:** Switched all 4 run specs to `GenericLLMJudgeMetric` + `GenericLLMJudgeAnnotator` with appropriate rubrics

### `creai_cps`
- **Problem:** Data path was `scenarios_new/data.json` but file lives at `scenarios/creai_cps/data.json`
- **Fix:** Updated path in scenario

### `cm3d`
- **Problem:** `kaggle` CLI not on PATH when invoked from Python subprocess
- **Fix:** Use `shutil.which("kaggle")` to resolve the full path before calling

---

## Infrastructure Fixes (One-Time)

### Entry point registration
Run specs were not being discovered by HELM because no entry point was registered. **Fix:**
1. Added `[project.entry-points.helm]` block to `pyproject.toml`
2. Created `run_specs/__init__.py` that imports all 158 run spec modules to trigger `@run_spec_function` registration
3. Ran `pip install -e .` to register the entry point

### Gemini tokenizer gating
HELM's default Gemini deployment config tried to load `google/gemma-2b` (gated HuggingFace model) for tokenization. **Fix:** Created `prod_env/model_deployments.yaml` overriding the tokenizer to `huggingface/gpt2` (public, always available).

### Missing `google` dependency
```
ModuleNotFoundError: No module named 'google'
```
**Fix:** `pip install "crfm-helm[google]"`

### Missing `metrics` package
`metrics/` directory existed but wasn't listed in `pyproject.toml` packages. **Fix:** Added `"metrics"` to the `packages` include list.

---

## Datasets Marked FAILED (Data Unavailable)

| Dataset | Reason |
|---|---|
| `clef_joker_2025_task2` | Gated HuggingFace dataset — requires approved access |
| `cm3d` | Invalid Kaggle dataset ID in scenario config |
| `creative_pair` | Dataset not yet publicly released |
| `d_humor` | Gated HuggingFace dataset — requires approved access + `huggingface-cli login` |

---

## LLM Judge Pattern (for reference)

Datasets with no gold references (pure generation tasks) should use this pattern in their run spec:

```python
from helm.benchmark.annotation.annotator import AnnotatorSpec

_RUBRIC = """\
Evaluate the quality of the generated text.
Score 1: ... Score 5: ...
"""

metric_specs = [
    MetricSpec(
        class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
        args={"metric_name": "llm_judge_quality"}
    ),
]

annotators = [
    AnnotatorSpec(
        class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
        args={
            "judge_model_name": "openai/gpt-4o",
            "judge_temperature": 0.0,
            "judge_max_new_tokens": 512,
            "metric_name": "llm_judge_quality",
            "rubric": _RUBRIC,
        },
    ),
]

return RunSpec(..., annotators=annotators)
```

Requires `OPENAI_API_KEY` to be set at eval time.

---

## Quick Debugging Checklist (Per Dataset)

Before running a dataset, check:

- [ ] Does the scenario import `CORRECT_TAG` and apply it to the gold reference?
- [ ] If `references=[]`, does the run spec use LLM judge instead of `BasicGenerationMetric`?
- [ ] Does the scenario use `output_path` for all downloaded files (not relative paths)?
- [ ] For multimodal: are images saved as local files before being passed to `MediaObject`?
- [ ] Does the run spec use `ADAPT_GENERATION_MULTIMODAL` for multimodal inputs?
- [ ] For MCQ: does the run spec use `exact_match` not `rouge_l`/`bleu_4`?
- [ ] Is `ensure_file_downloaded` return value being ignored (use your own path variable)?

---

## Running a Dataset

```bash
# Activate env and load keys
conda activate creativity-bench
source .env.local  # or export GEMINI_API_KEY="..." manually

# Run one dataset
/Users/nammu/miniconda3/envs/creativity-bench/bin/helm-run \
  --run-entries "DATASET_NAME:model=google/gemini-2.5-flash-lite" \
  --suite trial \
  --max-eval-instances 10
```