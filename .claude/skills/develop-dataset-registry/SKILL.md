---
name: develop-dataset-registry
description: >
  Extract dataset properties from scenario files, metric notes, papers, and repositories
  to build three YAML registry files for a large-scale HELM creativity benchmark.
  Processes one dataset at a time, writing/updating files after each.
---

# Develop Dataset Registry

## Goal

Build three YAML registry files that fully describe 158 creativity datasets for evaluation
under the HELM framework. Each dataset needs: basic metadata, inference generation config,
and evaluation metric definitions. All three files are written incrementally — one dataset
at a time — so that partial progress is always saved.

---

## Fixed Paths

> **CRITICAL: Never invent, shorten, or assume any path. Always use exactly these.**

| Item | Exact Path |
|------|-----------|
| Project root | `/home/public/vdeshpan/amazon-creativity-benchmark` |
| Datasets list (JSON) | `{root}/scenarios/list_of_all_datasets.json` |
| Per-dataset scenario file | `{root}/scenarios_new/{dataset_name}_scenario.py` |
| Per-dataset metric notes | `{root}/metric_notes/{dataset_name}_eval_metrics_notes.md` |
| Per-dataset annotator notes | `{root}/metric_notes/{dataset_name}_annotator_notes.md` |
| **OUTPUT: Master registry** | `{root}/registry_master.yaml` |
| **OUTPUT: Inference config** | `{root}/registry_inference.yaml` |
| **OUTPUT: Metric config** | `{root}/registry_metrics.yaml` |
| **OUTPUT: Summary report** | `{root}/registry_summary.md` |
| **OUTPUT: Per-metric table** | `{root}/registry_per_metric_breakdown.md` |

Where `{root}` = `/home/public/vdeshpan/amazon-creativity-benchmark`

---

## Reference Links (for lookup only)

- HELM repository: https://github.com/stanford-crfm/helm
- HELM metrics directory: https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/metrics
- HELM issues: https://github.com/stanford-crfm/helm/issues

---

## Step 0: Load the Dataset List

Before anything else, read the JSON file at `{root}/scenarios/list_of_all_datasets.json`.
This file contains the authoritative list of all dataset names. Every dataset in this list
must appear in all three output YAML files. Do not skip any. Do not add any that are not
in this list.

Parse the JSON and iterate over datasets **one at a time**, performing Steps 1–4 for each
before moving to the next.

---

## Step 1: Information Extraction — Priority Lookup Sequence

For each dataset, you need to extract information for all three registries (see Fields
below). Use this **strict lookup order** — stop as soon as you find the information:

```
PRIORITY 1 → Scenario file:  {root}/scenarios_new/{dataset_name}_scenario.py
PRIORITY 2 → Metric notes:   {root}/metric_notes/{dataset_name}_eval_metrics_notes.md
PRIORITY 3 → Annotator notes: {root}/metric_notes/{dataset_name}_annotator_notes.md
PRIORITY 4 → Research paper (URL will be inside the scenario file — use web_fetch)
PRIORITY 5 → GitHub repository (URL will be inside the scenario file — use web_fetch)
PRIORITY 6 → Put null for that field
```

### Detailed lookup instructions

**Priority 1 — Scenario file** (ALWAYS start here):
- Open `{root}/scenarios_new/{dataset_name}_scenario.py`
- Look for: paper URL, repo URL, modality info (what type of data is loaded as input,
  what type is expected as output), whether reference targets exist (check if `references`
  list is populated in `get_instances`), any generation parameters mentioned in comments
  or docstrings.

**Priority 2 — Metric notes** (check next):
- Open `{root}/metric_notes/{dataset_name}_eval_metrics_notes.md`
- This file typically contains: list of metrics used, metric types, model names for
  model-based metrics, LLM judge prompts, and generation configs for judges.
- **This is your primary source for Category-3 (metrics) information.**

**Priority 3 — Annotator notes** (check next):
- Open `{root}/metric_notes/{dataset_name}_annotator_notes.md`
- May contain additional metric details, annotation guidelines, or LLM judge instructions.

**Priority 4 — Research paper** (only if still missing info):
- The scenario file will contain a URL to the paper (usually arxiv).
- Use `web_fetch` to read the paper page.
- Look for: generation parameters (temperature, max_tokens — often in "Experimental Setup"
  or "Implementation Details" sections), evaluation metrics (often in "Evaluation" or
  "Metrics" sections).
- **Do NOT hallucinate values. If the paper does not explicitly state a parameter, move
  to Priority 5.**

**Priority 5 — GitHub repository** (last resort):
- The scenario file may contain a GitHub URL.
- Use `web_fetch` to read the repo README or config files.
- Look for: generation parameters in config files, metric implementations, evaluation scripts.

**Priority 6 — Not found**:
- If information cannot be found after checking all 5 sources, set the field to `null`.
- For generation config specifically: set `_use_defaults: true` (see format below).

### Important rules for this step
- **File may not exist**: If a scenario file, metric notes, or annotator notes file does
  not exist for a dataset, skip that priority level and move to the next. Do NOT error out.
- **Do NOT guess or infer values**: If a source does not explicitly state a value, do not
  assume it. Move to the next priority level.
- **Document your source**: For every non-null value you extract, note WHERE you found it
  (e.g., "scenario file line 45", "metric_notes", "paper Section 4.2", "repo README").

---

## Step 2: Fields to Extract

### Category 1 — Master Registry (`registry_master.yaml`)

For each dataset, extract:

| Field | Type | Description | How to find |
|-------|------|-------------|-------------|
| `display_name` | string | Human-readable name | Scenario file: class docstring or `name` attribute |
| `source_paper` | string or null | URL to the research paper | Scenario file: look for arxiv/doi URLs in comments, docstrings, or string constants |
| `source_repo` | string or null | URL to the GitHub repository | Scenario file: look for github.com URLs |
| `input_modality` | enum | One of: `text`, `image`, `audio`, `video`, `multimodal` | Scenario file: inspect what `get_instances` loads as `Input`. If `Input(text=...)` → `text`. If multimedia/image content → `image`. If both text and image → `multimodal` |
| `output_modality` | enum | One of: `text`, `image`, `audio`, `code`, `structured_json` | Scenario file + paper: what is the model expected to generate? Usually `text` for most creativity tasks |
| `has_reference_target` | boolean | Are ground-truth reference outputs provided? | Scenario file: check if `references=[Reference(...)]` is populated with actual content in `get_instances`. If references list is empty or not provided → `false` |
| `scenario_file` | string | Relative path from project root | Always: `scenarios_new/{dataset_name}_scenario.py` |

### Category 2 — Inference Config (`registry_inference.yaml`)

For each dataset, extract:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float | 0.7 | Sampling temperature |
| `max_new_tokens` | int | 512 | Maximum tokens to generate |
| `num_return_sequences` | int | 1 | Number of completions per input |
| `top_p` | float or null | null | Nucleus sampling threshold |
| `top_k` | int or null | null | Top-k sampling |
| `stop_sequences` | list or null | null | Sequences that stop generation |
| `do_sample` | bool | true | Whether to use sampling |
| `source` | string | — | Where you found these values (for traceability) |

**If none of the priority sources specify generation parameters**, set:
```yaml
dataset_name:
  _use_defaults: true
  source: "not specified in paper or repo; using benchmark defaults"
```

### Category 3 — Metric Config (`registry_metrics.yaml`)

For each dataset, extract a list of metrics. Each metric has a `type` field that determines
which additional fields are required:

**Common fields (ALL metrics):**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Metric name (e.g., "bleu", "rouge_l", "creativity_score") |
| `in_helm` | boolean | Is this metric already implemented in HELM? |
| `type` | enum | One of: `formula_based`, `model_based`, `llm_judge` |

**Type definitions and their additional required fields:**

#### `formula_based`
Metrics computed purely by formula (no neural model needed).
Examples: BLEU, ROUGE, exact match, F1, Levenshtein distance, perplexity.

| Additional Field | Type | Description |
|-----------------|------|-------------|
| `helm_class` | string or null | Full HELM class path if `in_helm: true`, else `null`. Find by checking https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/metrics |

#### `model_based`
Metrics that require a pre-trained model (NOT the model being evaluated).
Examples: BERTScore, BLEURT, CLIPScore, FID, MAUVE.

| Additional Field | Type | Description |
|-----------------|------|-------------|
| `helm_class` | string or null | Full HELM class path if available in HELM |
| `metric_model` | string | The model needed to compute this metric (e.g., `"bert-base-uncased"`, `"openai/clip-vit-base-patch32"`) |

#### `llm_judge`
Metrics that use an LLM-as-a-Judge to score the generated response.

| Additional Field | Type | Description |
|-----------------|------|-------------|
| `judge_model_name` | string or null | Which LLM to use as judge (e.g., `"openai/gpt-4"`, `"openai/gpt-4o"`) |
| `judge_prompt` | string or null | The exact prompt template for the judge. Must include placeholders if specified in the source: `{input_text}`, `{generated_response}`, `{reference_target}` |
| `judge_temperature` | float or null | Temperature for judge generation |
| `judge_max_new_tokens` | int or null | Max tokens for judge response |

### How to determine if a metric is available in HELM

To check if a metric exists in HELM:
1. Go to https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/metrics
2. Search for the metric name in the file listing
3. If found, record the full class path (e.g., `helm.benchmark.metrics.basic_metrics.BasicMetric`)
4. If not found, set `in_helm: false` and `helm_class: null`

Common HELM metrics you will likely encounter:
- BLEU → `helm.benchmark.metrics.basic_metrics.BasicMetric`
- ROUGE → `helm.benchmark.metrics.basic_metrics.BasicMetric`
- F1 → `helm.benchmark.metrics.basic_metrics.BasicMetric`
- Exact match → `helm.benchmark.metrics.basic_metrics.BasicMetric`
- BERTScore → check `helm.benchmark.metrics.bertscore_metrics` (if it exists)
- Toxicity → `helm.benchmark.metrics.toxicity_metrics.ToxicityMetric`
- Bias → `helm.benchmark.metrics.bias_metrics.BiasMetric`

**When uncertain**: set `in_helm: false` — it is safer to mark something as not in HELM
(we can verify later) than to incorrectly claim it is available.

---

## Step 3: Write/Update the Registry Files

After extracting information for ONE dataset, immediately write or update all three YAML
files. This ensures partial progress is saved.

### Output File 1: `{root}/registry_master.yaml`

```yaml
# Master registry - basic information for each dataset
# Auto-generated by develop-dataset-registry skill

datasets:
  # -- Example entry --
  torrance_verbal:
    display_name: "Torrance Verbal Creativity Test"
    source_paper: "https://arxiv.org/abs/XXXX.XXXXX"
    source_repo: "https://github.com/xxx/xxx"
    input_modality: text            # one of: text, image, audio, video, multimodal
    output_modality: text           # one of: text, image, audio, code, structured_json
    has_reference_target: true
    scenario_file: "scenarios_new/torrance_verbal_scenario.py"

  # -- Another entry --
  visual_metaphor_gen:
    display_name: "Visual Metaphor Generation"
    source_paper: "https://arxiv.org/abs/YYYY.YYYYY"
    source_repo: null
    input_modality: image
    output_modality: text
    has_reference_target: false
    scenario_file: "scenarios_new/visual_metaphor_gen_scenario.py"
```

### Output File 2: `{root}/registry_inference.yaml`

```yaml
# Generation parameters for model inference
# Auto-generated by develop-dataset-registry skill

defaults:
  temperature: 0.7
  max_new_tokens: 512
  num_return_sequences: 1
  top_p: null
  top_k: null
  stop_sequences: null
  do_sample: true

datasets:
  # -- When specific values are found --
  torrance_verbal:
    temperature: 0.7
    max_new_tokens: 256
    num_return_sequences: 1
    top_p: null
    top_k: null
    stop_sequences: null
    do_sample: true
    source: "paper Section 4.2, Table 3"

  # -- When NO generation config is found anywhere --
  some_other_dataset:
    _use_defaults: true
    source: "not specified in paper or repo; using benchmark defaults"
```

### Output File 3: `{root}/registry_metrics.yaml`

```yaml
# Evaluation metrics for each dataset
# Auto-generated by develop-dataset-registry skill

datasets:
  torrance_verbal:
    metrics:
      # formula-based metric that IS in HELM
      - name: "bleu_4"
        in_helm: true
        type: "formula_based"
        helm_class: "helm.benchmark.metrics.basic_metrics.BasicMetric"

      # model-based metric that is NOT in HELM
      - name: "bert_score"
        in_helm: false
        type: "model_based"
        helm_class: null
        metric_model: "bert-base-uncased"

      # LLM-as-a-Judge metric
      - name: "creativity_judgment"
        in_helm: false
        type: "llm_judge"
        judge_model_name: "openai/gpt-4"
        judge_prompt: |
          Evaluate the creativity of the following response.

          Input: {input_text}
          Response: {generated_response}
          Reference: {reference_target}

          Score from 1-5 and explain.
        judge_temperature: 0.0
        judge_max_new_tokens: 256
```

### Writing rules

- **Append, don't overwrite**: When adding a new dataset, read the existing YAML file first,
  add the new dataset entry under `datasets:`, then write the full file back. Never overwrite
  previously written dataset entries.
- **Use consistent key ordering**: Within each dataset entry, always write keys in the order
  shown in the templates above.
- **Strings with special characters**: Wrap in double quotes. Use YAML block scalar `|` for
  multi-line strings (especially judge prompts).
- **null values**: Write literally as `null` (no quotes).

---

## Step 4: After All Datasets — Generate and Save Summary Tables

Once ALL datasets from the list have been processed and written to the three YAML files,
generate and **save to disk** the following two summary files:

1. `{root}/registry_summary.md` — overall summary with all tables below
2. `{root}/registry_per_metric_breakdown.md` — the per-metric breakdown table (also included in the summary, but saved separately for easy reference)

**Both files MUST be written to disk. Do not skip this step.**

```markdown
# Creativity Benchmark Registry Summary

## Overview
- Total datasets: {N}
- Datasets with reference targets: {count}
- Datasets using defaults for generation config: {count}

## Modality Breakdown

| Input Modality | Count |
|---------------|-------|
| text          | X     |
| image         | X     |
| ...           | ...   |

| Output Modality | Count |
|----------------|-------|
| text           | X     |
| image          | X     |
| ...            | ...   |

## Metric Type Breakdown

| Metric Type    | Unique Metrics | Total Usages Across Datasets |
|---------------|---------------|------------------------------|
| formula_based | X             | X                            |
| model_based   | X             | X                            |
| llm_judge     | X             | X                            |

## Per-Dataset Summary

| Dataset | Input | Output | Has Ref? | Gen Config Source | # Metrics | Metric Types |
|---------|-------|--------|----------|-------------------|-----------|--------------|
| dataset_1 | text | text | yes | paper | 3 | formula_based, llm_judge |
| dataset_2 | image | text | no | defaults | 2 | model_based, llm_judge |
| ... | ... | ... | ... | ... | ... | ... |

## Per Metric Breakdown

| Metric Name | Metric Type | Metric Class in HELM (null if not available) | Datasets Using This Metric |
|-------------|-------------|----------------------------------------------|----------------------------|
| bleu_4      | formula_based | helm.benchmark.metrics.basic_metrics.BasicMetric | dataset_1, dataset_3, dataset_7 |
| bert_score  | model_based | null | dataset_2, dataset_5 |
| creativity_judgment | llm_judge | null | dataset_1, dataset_2, dataset_4 |
| ... | ... | ... | ... |

## Fields Set to Null (Gaps)

| Dataset | Field | Reason |
|---------|-------|--------|
| dataset_X | source_repo | not found in any source |
| dataset_Y | judge_prompt | metric notes incomplete |
| ... | ... | ... |
```

---

## Processing Loop (Pseudocode)

```
datasets = load_json("{root}/scenarios/list_of_all_datasets.json")

for dataset_name in datasets:

    # --- EXTRACT ---
    info = {}

    # Priority 1: scenario file
    scenario_path = "{root}/scenarios_new/{dataset_name}_scenario.py"
    if file_exists(scenario_path):
        info.update(extract_from_scenario(scenario_path))

    # Priority 2: metric notes
    metric_notes_path = "{root}/metric_notes/{dataset_name}_eval_metrics_notes.md"
    if file_exists(metric_notes_path):
        info.update_missing_fields(extract_from_metric_notes(metric_notes_path))

    # Priority 3: annotator notes
    annotator_path = "{root}/metric_notes/{dataset_name}_annotator_notes.md"
    if file_exists(annotator_path):
        info.update_missing_fields(extract_from_annotator_notes(annotator_path))

    # Priority 4: paper (only for STILL-MISSING fields)
    if info.has_missing_fields() and info.paper_url is not null:
        info.update_missing_fields(extract_from_paper(info.paper_url))

    # Priority 5: repo (only for STILL-MISSING fields)
    if info.has_missing_fields() and info.repo_url is not null:
        info.update_missing_fields(extract_from_repo(info.repo_url))

    # Priority 6: set remaining missing fields to null
    info.fill_missing_with_null()

    # --- WRITE ---
    update_registry_master(dataset_name, info)
    update_registry_inference(dataset_name, info)
    update_registry_metrics(dataset_name, info)

# --- FINAL ---
generate_and_save_summary("{root}/registry_summary.md")
generate_and_save_per_metric_breakdown("{root}/registry_per_metric_breakdown.md")
```

**Key behavior**: `update_missing_fields` means: only fill in fields that are STILL null/missing.
Never overwrite a value found at a higher priority level with a value from a lower priority level.

---

## Edge Cases and Error Handling

1. **Scenario file does not exist for a dataset**: Log a warning, set all Category-1 fields
   to null except `scenario_file` (still set the expected path). Continue to metric notes.

2. **Metric notes file does not exist**: Skip silently, move to next priority level.

3. **Paper URL is not found in scenario file**: Skip Priority 4. Do not search the web
   for the paper independently.

4. **Paper/repo is behind a paywall or returns error**: Log the error, move to Priority 6.

5. **Metric name is ambiguous** (e.g., "accuracy" could be classification accuracy or
   semantic accuracy): Use the most specific name possible based on context. If truly
   ambiguous, use the name exactly as written in the source and add a comment.

6. **Multiple metrics with the same name but different configs**: List them as separate
   entries with disambiguating suffixes (e.g., `bleu_1`, `bleu_4`, or `rouge_1`, `rouge_l`).

7. **The metric notes mention a metric but give no details**: Create the entry with `name`
   and set all other fields to null. Set `type` to null if you cannot determine it.

8. **A dataset has NO metrics listed anywhere**: Create an entry with an empty metrics list:
   ```yaml
   dataset_name:
     metrics: []
   ```

9. **YAML special characters in judge prompts**: Always use the block scalar `|` for
   multi-line judge prompts to avoid YAML parsing issues with colons, brackets, etc.

10. **Dataset name contains hyphens or special characters**: Use the exact name from the
    JSON list as the YAML key. If it contains characters that need quoting in YAML, wrap
    the key in double quotes.

---

## Quality Checklist (verify before finishing)

After all datasets are processed, verify:

- [ ] Number of entries in `registry_master.yaml` == number of datasets in the JSON list
- [ ] Number of entries in `registry_inference.yaml` == number of datasets in the JSON list
- [ ] Number of entries in `registry_metrics.yaml` == number of datasets in the JSON list
- [ ] Every dataset in the JSON list appears in all three files (no missing entries)
- [ ] No dataset appears that is NOT in the JSON list (no extra entries)
- [ ] All YAML files are valid YAML (parseable without errors)
- [ ] `registry_summary.md` exists and Per-Dataset Summary row count matches dataset count
- [ ] `registry_per_metric_breakdown.md` exists and lists every unique metric across all datasets
- [ ] No field contains placeholder text like "TODO" or "FIXME" (use null instead)