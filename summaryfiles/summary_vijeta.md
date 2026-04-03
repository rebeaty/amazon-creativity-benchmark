# Debugging Session Insights

**Date:** 2026-04-03
**Model under test:** `google/gemini-2.5-flash-lite`
**Suite:** `trial` (max 10 instances)

---

## Infrastructure Fixes (apply to all datasets)

### 1. Custom run specs not discovered by HELM

**Problem:** HELM's `discover_run_spec_functions()` only imports modules inside its own `helm.benchmark.run_specs` package. Our run specs in `run_specs/` were never registered.

**Fix:** Created `run_specs/__init__.py` that auto-imports all `*_run_specs.py` modules (triggering `@run_spec_function` decorators). Created `eval_scripts/_helm_run.sh` — a shared helper that sets `PYTHONPATH` and invokes HELM via `python -c "import run_specs; ..."` so registrations happen in the same process. All 158 eval scripts were patched to source this helper.

**Key detail:** Must run `import run_specs` and `helm.benchmark.run.main()` in the **same Python process** — a separate `python -c` beforehand does nothing since registrations live in process memory.

### 2. Run entry delimiter: comma vs colon

**Problem:** All eval scripts used `dataset_name,model=X` but HELM's `parse_object_spec()` expects `dataset_name:model=X` (colon separates name from args, commas separate args from each other).

**Fix:** `sed` replaced the first `,model=` with `:model=` across all 159 eval scripts. Also fixed 2 scripts (`aaar.sh`, `assocam.sh`) that use `RUN_ENTRIES+=()` array syntax.

### 3. Model name must include provider prefix

**Problem:** Passing `gemini-2.5-flash-lite` fails — HELM requires `google/gemini-2.5-flash-lite` to match its model deployment registry.

**Fix:** User-side fix — always use full model name with provider prefix.

### 4. Missing `google-genai` dependency

**Problem:** `ModuleNotFoundError: No module named 'google'` — HELM's `GoogleGenAIClient` needs the `google-genai` package.

**Fix:** Added `"google-genai>=1.0"` to `pyproject.toml` dependencies. Install via `pip install google-genai`.

---

## Common Scenario Bugs (patterns to watch for)

### A. `output={"text": ...}` instead of `output=Output(text=...)`

**Frequency:** 8 scenario files
**Error:** `AttributeError: 'dict' object has no attribute 'text'` in metrics or annotator code.
**Files fixed:** `matdesign`, `dialogue_diversity`, `historical_analogy`, `llm4biohypogen`, `metaphor_generation`, `puntuguese`, `schnovel`, `twistlist`.
**Fix pattern:** Replace `output={"text": X}` with `output=Output(text=X)` and add `Output` to the import from `helm.benchmark.scenarios.scenario`.

### B. PIL image objects passed to `MediaObject(location=...)`

**Frequency:** At least 1 (cii_bench), likely more image datasets.
**Error:** `AttributeError: 'WebPImageFile' object has no attribute 'decode'`
**Fix pattern:** Save PIL image to disk first, pass the file path:
```python
image_path = os.path.join(images_dir, f"{idx}.jpg")
if not os.path.exists(image_path):
    pil_image.convert("RGB").save(image_path, "JPEG")
```

### C. Wrong metric class for MCQ tasks

**Frequency:** 33 run specs used `ADAPT_MULTIPLE_CHOICE_JOINT` with `BasicGenerationMetric`.
**Error:** `AttributeError: 'NoneType' object has no attribute 'lower'` — MCQ output mapping returns `None` for unrecognized outputs, and text metrics (`f1_score`, `rouge_l`) crash on `None`.
**Fix:** Replace with `MultipleChoiceClassificationMetric`:
```python
MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})
```

### D. Non-existent metric classes in MetricSpec

**Frequency:** 77 + 37 = 114 run specs.
**Error:** `TypeError: compute_reference_metrics() missing 4 required positional arguments` — `MetricSpec` expects a **class** (instantiated with `args`), not a function.
**Broken references:**
- `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` (a function, not a class)
- `helm.benchmark.metrics.basic_metrics.BasicMetric` (does not exist)

**Fix:** Both replaced with `BasicGenerationMetric` + default metric names:
```python
MetricSpec(
    class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
    args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]}
)
```

### E. Missing `subset` argument in ScenarioSpec

**Example:** `thenextchapter` — scenario `__init__` requires `subset` but run spec passes `args={}`.
**Fix:** Add the arg to the run spec function signature with a default, and pass it through:
```python
def get_spec(subset: str = "roc") -> RunSpec:
    scenario_spec = ScenarioSpec(..., args={"subset": subset})
```

### F. Missing required args in custom MetricSpec

**Example:** `vietnamese_poem` — `VietnamesePoemMetric.__init__()` requires `metric` arg.
**Fix:** Instantiate one MetricSpec per metric variant with the appropriate arg.

---

## Data Availability Issues (no code fix)

| Dataset | Issue | Resolution |
|---|---|---|
| `funqa` | Videos not downloaded (YouTube source, no auto-download) | Skip instances with missing videos |
| `v_flute` | Gated HuggingFace dataset, requires access request | Request access at HF page |
| `webnovelbench` | Dataset JSON must be manually downloaded from GitHub | Download and place in output dir |

---

## URL / Network Issues

### Malformed URLs in dataset (infochartqa)

**Patterns found:**
1. `hhttps://...` — extra `h` prefix
2. `preview.redd.it/...` — missing `https://` scheme entirely

**Fix:** Sanitize URLs before passing to `MediaObject`:
```python
while url.startswith("hhttp"):
    url = url[1:]
if not url.startswith(("http://", "https://")):
    url = "https://" + url
```

### Hanging downloads (yesbut_v2)

**Problem:** `urllib.request.urlopen()` without timeout hangs on unresponsive Google Drive URLs.
**Fix:** Add `timeout=30` and wrap download in try/except to skip failures.

### Wrong GitHub branch (thenextchapter)

**Problem:** URL used `/main/` but repo default branch is `master`.
**Fix:** Change URL to `/master/`.

---

## ScenarioSpec class name mismatches

Only 1 found: `analobench` — run spec referenced `AnaloBenchScenario` but actual class is `AnalobenchScenario` (lowercase 'b'). Full audit of all 161 ScenarioSpec references confirmed no other mismatches.

---

## Quick Reference: Correct HELM Metric Classes

| Task Type | Correct MetricSpec class | Args |
|---|---|---|
| Open-ended generation | `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric` | `{"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]}` |
| Multiple choice | `helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric` | `{}` |
| Classification | `helm.benchmark.metrics.classification_metrics.ClassificationMetric` | `{"labels": [...]}` (optional) |
| Summarization | `helm.benchmark.metrics.summarization_metrics.SummarizationMetric` | varies |
| LLM judge | `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric` | `{"metric_name": "..."}` |

---

## Datasets Successfully Run

alpaca_eval_2, analobench, arn, balancecc_prompt_generation, calligrapher, chinese_homophonic_puns, and others through the pending list.