---
name: helm-runspec-generator
description: >
  Generate HELM RunSpec files for the Amazon Creativity Benchmark.
  Use this skill whenever the user asks to generate, create, write, or update
  HELM run spec files, run specification functions, or evaluation configs for
  datasets in the creativity benchmark. Also trigger when the user says
  "generate run specs", "write run_specs", "create evaluation configs",
  "build HELM specs for my datasets", or references the run_specs/ output
  directory. This skill reads two registry YAML files and ~160 scenario files,
  then produces one Python run spec file per dataset.
---

# HELM RunSpec Generator

## Goal

Produce **one Python run spec file per dataset** (~160 files total) in:

```
/home/public/vdeshpan/amazon-creativity-benchmark/run_specs/
```

Each file contains one or more `@run_spec_function`-decorated functions that
wire together the dataset's Scenario, AdapterSpec, MetricSpecs, and (when
needed) AnnotatorSpecs for LLM-as-Judge evaluation.

---

## 1. Input Files — Locations and Schemas

### 1.1 Inference Registry

**Path:** `/home/public/vdeshpan/amazon-creativity-benchmark/data/registry/registry_inference.yaml`

```yaml
defaults:
  temperature: 0.7
  max_new_tokens: 512
  num_return_sequences: 1
  top_p: null
  top_k: null
  stop_sequences: null
  do_sample: true

datasets:
  <dataset_name>:
    # If _use_defaults: true, inherit all values from `defaults:`
    _use_defaults: true
    source: "..."
    # OR explicit per-dataset overrides:
    num_return_sequences: 30
    temperature: 0.7
    max_new_tokens: 512
    top_p: null
    top_k: null
    stop_sequences: null
    do_sample: true
    source: "..."
```

**Field mapping to HELM AdapterSpec:**

| Registry Field         | AdapterSpec Field        | Notes                                         |
|------------------------|--------------------------|-----------------------------------------------|
| `temperature`          | `temperature`            | Direct map                                    |
| `max_new_tokens`       | `max_tokens`             | Rename only                                   |
| `num_return_sequences` | `num_outputs`            | Rename only                                   |
| `top_p`                | `top_p`                  | Omit from AdapterSpec if null                 |
| `top_k`                | `top_k`                  | Omit from AdapterSpec if null                 |
| `stop_sequences`       | `stop_sequences`         | Use `["\n"]` as default if null               |
| `do_sample`            | *(no direct equivalent)* | If false, set temperature=0.0 in AdapterSpec  |

### 1.2 Metrics Registry

**Path:** `/home/public/vdeshpan/amazon-creativity-benchmark/data/registry/registry_metrics.yaml`

```yaml
datasets:
  <dataset_name>:
    metrics:
      - name: "metric_display_name"
        in_helm: true                    # Use built-in HELM metric class
        type: "model_based"              # or "lexical", "statistical", "llm_judge", etc.
        helm_class: "helm.benchmark.metrics.some_module.SomeMetric"
        metric_model: "all-mpnet-base-v2"  # optional, for model-based metrics
      - name: "another_metric"
        in_helm: false                   # Custom / LLM-as-Judge
        type: "llm_judge"
        judge_model_name: "openai/gpt-4-1106-preview"
        judge_prompt: null               # null = use generic prompt (see §4)
        judge_temperature: 0.0
        judge_max_new_tokens: 256
```

### 1.3 Master Registry (Paper Links + Context)

**Path:** `/home/public/vdeshpan/amazon-creativity-benchmark/data/registry/registry_master.yaml`

This file contains paper links and repository links for each dataset. Used
primarily during **rubric generation** (§4) to look up how the original paper
defined each LLM-judge metric. Parse this file to get `paper_link` and
`repo_link` per dataset.

### 1.4 Scenario Files

**Path pattern:** `/home/public/vdeshpan/amazon-creativity-benchmark/scenarios_new/<dataset_name>_scenario.py`

Each scenario file contains a class inheriting from `helm.benchmark.scenarios.scenario.Scenario`.
Parse these files to extract:

- **Class name** (e.g. `AaarScenario`) — needed for `ScenarioSpec.class_name`
- **`SUBTASKS`** list or `__init__` parameters — needed to know which subtask
  values to generate `@run_spec_function` entries for
- **`name` attribute** — the scenario's registered name
- **`tags` attribute** — reuse as RunSpec `groups`
- **Prompt structure** (from docstring or `get_instances`) — to infer
  `input_prefix`, `output_prefix`, `instructions` for AdapterSpec
- **Task type** — infer from tags, docstring, or reference structure:
  - If references have multiple options with CORRECT_TAG → **MCQA**
  - If single reference with CORRECT_TAG → **generation** (short-answer or open-ended)
  - If no CORRECT_TAG → **open-ended generation**

---

## 2. Output File Structure

For each dataset, produce:

```
/home/public/vdeshpan/amazon-creativity-benchmark/run_specs/<dataset_name>_run_specs.py
```

### Template

```python
"""HELM Run Specs for <dataset_name>."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
# Only if LLM-as-Judge metrics are present:
from helm.benchmark.annotation.annotator import AnnotatorSpec


@run_spec_function("<dataset_name>")  # or "<dataset_name>_<subtask>" for multi-subtask
def get_<dataset_name>_spec(<params>) -> RunSpec:

    # --- 1. ScenarioSpec ---
    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.<dataset_name>_scenario.<ClassName>",
        args={<constructor_args>},
    )

    # --- 2. AdapterSpec ---
    adapter_spec = AdapterSpec(
        method=<ADAPT_METHOD>,           # See §3
        instructions=<instructions>,     # From scenario docstring/prompt
        input_prefix=<input_prefix>,     # Inferred from scenario
        input_suffix="\n",
        output_prefix=<output_prefix>,   # Inferred from scenario
        output_suffix="\n",
        max_train_instances=<N>,         # Default 0 for zero-shot unless scenario has TRAIN_SPLIT
        num_outputs=<num_return_sequences>,
        max_tokens=<max_new_tokens>,
        temperature=<temperature>,
        stop_sequences=<stop_sequences>,
        # Include top_p, top_k only if non-null
    )

    # --- 3. MetricSpecs ---
    metric_specs = [
        <see §3.2 for construction rules>
    ]

    # --- 4. AnnotatorSpecs (only if LLM-as-Judge metrics exist) ---
    annotators = [
        <see §4 for LLM-as-Judge setup>
    ]

    # --- 5. Assemble RunSpec ---
    return RunSpec(
        name="<dataset_name>:<param>=<value>",  # include subtask if applicable
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "<dataset_name>"],
        annotators=annotators if annotators else None,
    )
```

### Multi-Subtask Datasets

If a scenario defines `SUBTASKS = [...]` or has a `subtask` parameter in
`__init__`, generate **multiple `@run_spec_function` entries in the same file**:

```python
@run_spec_function("<dataset_name>_<subtask1>")
def get_<dataset_name>_<subtask1>_spec() -> RunSpec:
    ...

@run_spec_function("<dataset_name>_<subtask2>")
def get_<dataset_name>_<subtask2>_spec() -> RunSpec:
    ...
```

Each subtask gets its own decorated function. The `ScenarioSpec.args` must
include `{"subtask": "<subtask_value>"}`.

---

## 3. Construction Rules

### 3.1 AdapterSpec — Method Selection

Infer the HELM adaptation method from the scenario's task structure:

| Signal in Scenario                                         | HELM Method                              | Import Constant                       |
|------------------------------------------------------------|------------------------------------------|---------------------------------------|
| Multiple `Reference` objects with `CORRECT_TAG` on one     | Multiple-choice (joint)                  | `ADAPT_MULTIPLE_CHOICE_JOINT`         |
| Multiple `Reference`, scored separately                    | Multiple-choice (separate)               | `ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL` |
| Single `Reference` with `CORRECT_TAG`, short expected output | Generation (short-answer)              | `ADAPT_GENERATION`                    |
| Single `Reference` or none, open-ended / long output       | Generation (open-ended)                  | `ADAPT_GENERATION`                    |
| Tags contain "classification"                              | Generation with constrained output       | `ADAPT_GENERATION`                    |

**Default:** `ADAPT_GENERATION` — the vast majority of creativity benchmark
datasets are open-ended generation tasks.

### 3.2 MetricSpecs — Construction from Registry

For each metric entry in `registry_metrics.yaml`:

**Case A: `in_helm: true`**
```python
MetricSpec(
    class_name=entry["helm_class"],
    args={
        # Pass metric_model if present
        **({"model_name": entry["metric_model"]} if entry.get("metric_model") else {}),
    },
)
```

**Important:** Deduplicate MetricSpecs. If multiple metric names point to the
same `helm_class` with the same args (e.g. `sentence_bert_f1`,
`sentence_bert_precision`, `sentence_bert_recall` all using
`SummarizationMetric`), emit only **one** MetricSpec for that class — the
single class computes all sub-metrics internally.

**Case B: `in_helm: false` and `type: "llm_judge"`**
→ Use the Generic LLM-as-Judge solution described in §4 below.

**Case C: `in_helm: false` and NOT `type: "llm_judge"`**
→ The custom metric class should already exist. Use:
```python
MetricSpec(
    class_name="<custom_module_path>.<CustomMetricClass>",
    args={},
)
```
Search for the class in the project. If not found, emit a `# TODO:` comment
and flag it in the generation summary.

### 3.3 AdapterSpec — Prompt Details

Read the scenario file to extract prompt information:

1. **Instructions:** Look for long prompt strings in `get_instances()` or
   class-level constants. If the scenario embeds the full prompt in the
   `Input.text` (as AAAR does), set `instructions=""` — the scenario handles
   prompting internally.
2. **input_prefix / output_prefix:** If the scenario constructs prompts with
   clear noun labels (e.g. `"Question: "`, `"Passage: "`), extract them.
   Otherwise use `""` for both.
3. **max_train_instances:** Set to `0` (zero-shot) unless the scenario
   explicitly uses `TRAIN_SPLIT` instances. If it does, default to `5`.

### 3.4 Inference Parameters

Merge defaults and per-dataset overrides from `registry_inference.yaml`:

```python
def resolve_inference_config(defaults: dict, dataset_entry: dict) -> dict:
    if dataset_entry.get("_use_defaults", False):
        config = {**defaults}
    else:
        config = {**defaults, **dataset_entry}
    # Remove non-inference keys
    config.pop("_use_defaults", None)
    config.pop("source", None)
    return config
```

---

## 4. LLM-as-Judge — Per-Metric Rubric-Aware Architecture

Read `references/llm_judge_setup.md` for the full implementation code.
Below is the design summary.

### 4.1 Core Design Principle

**One judge call per metric, with a dataset-specific rubric.**

The judge for "fluency" on Arabic story generation must know what fluency
means *for stories*, with a scoring rubric tailored to that context. The same
"fluency" metric on a code generation dataset would need a completely different
rubric. A generic "rate fluency 1-5" prompt produces unreliable scores.

### 4.2 The Rubric Registry

A new YAML file is generated as an intermediate artifact:

**Path:** `/home/public/vdeshpan/amazon-creativity-benchmark/data/registry/registry_rubrics.yaml`

```yaml
datasets:
  arastories:
    task_description: "Generate Arabic short stories given a prompt/theme"
    paper_link: "https://..."
    metrics:
      fluency:
        rubric: |
          Evaluate the FLUENCY of the generated Arabic story. Fluency measures
          how natural, grammatically correct, and readable the text is.
          
          Score 1: Incomprehensible or severely broken language
          Score 2: Frequent grammatical errors that impede understanding
          Score 3: Generally understandable but with noticeable awkwardness
          Score 4: Reads naturally with only minor issues
          Score 5: Perfectly fluent, native-quality writing
        score_min: 1
        score_max: 5
      coherence:
        rubric: |
          Evaluate the COHERENCE of the generated Arabic story. Coherence
          measures whether the story has logical flow, consistent characters,
          and a sensible narrative structure.
          
          Score 1: No logical connection between sentences or ideas
          Score 2: Major gaps in logic or contradictions
          Score 3: Mostly follows a thread but with some disconnects
          Score 4: Well-structured narrative with minor inconsistencies
          Score 5: Perfectly coherent story with clear narrative arc
        score_min: 1
        score_max: 5
      # ... one entry per LLM-judge metric
```

### 4.3 How Rubrics Are Generated

Follow this procedure to populate `registry_rubrics.yaml`:

**Step A — Gather context per dataset:**
1. Read the dataset's entry from `registry_master.yaml` to get `paper_link`
2. Read the scenario file's docstring and `tags` for task description
3. Collect the list of `in_helm: false, type: "llm_judge"` metric names from
   `registry_metrics.yaml`

**Step B — Check for existing rubric info:**
1. Look in the scenario file docstring for evaluation descriptions
2. Check if `metric_notes/<dataset_name>_eval_metrics_notes.md` exists at
   `/home/public/vdeshpan/amazon-creativity-benchmark/data/metric_notes/`
3. If paper-defined rubrics are found in either source, use them verbatim

**Step C — Generate rubrics for metrics without paper-defined rubrics:**
Using the dataset's task description and metric name, write a rubric that:
1. **Names the metric** and defines what it measures *in the context of this task*
2. **Provides a 1–5 scoring scale** with one sentence per level
3. **Anchors the extremes** (1 = worst, 5 = best) with task-specific examples

**Step D — Write `registry_rubrics.yaml` and flag for human review:**
After generating, add a header comment:
```yaml
# AUTO-GENERATED RUBRICS — Review before running evaluations.
# Rubrics marked [PAPER] were extracted from the paper/docs.
# Rubrics marked [GENERATED] were auto-generated and should be verified.
```

### 4.4 Architecture — One AnnotatorSpec Per Metric

Unlike the previous grouped design, each LLM-judge metric gets its **own
AnnotatorSpec**. This ensures one judge call per metric with a focused,
rubric-specific prompt.

```python
# In the generated run spec file — for EACH llm_judge metric:
annotators = [
    AnnotatorSpec(
        class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
        args={
            "judge_model_name": "openai/gpt-4-0125-preview",
            "judge_temperature": 0.0,
            "judge_max_new_tokens": 256,
            "metric_name": "fluency",           # single metric
            "rubric": _RUBRIC_FLUENCY,          # dataset-specific rubric string
        },
    ),
    AnnotatorSpec(
        class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
        args={
            "judge_model_name": "openai/gpt-4-0125-preview",
            "judge_temperature": 0.0,
            "judge_max_new_tokens": 256,
            "metric_name": "coherence",
            "rubric": _RUBRIC_COHERENCE,
        },
    ),
    # ... one per metric
]
```

The rubric strings are stored as **module-level constants** in each run spec
file, loaded from `registry_rubrics.yaml` at generation time:

```python
# ── Rubrics (from registry_rubrics.yaml) ────────────────────────────────

_RUBRIC_FLUENCY = """\
Evaluate the FLUENCY of the generated Arabic story. Fluency measures
how natural, grammatically correct, and readable the text is.

Score 1: Incomprehensible or severely broken language
Score 2: Frequent grammatical errors that impede understanding
Score 3: Generally understandable but with noticeable awkwardness
Score 4: Reads naturally with only minor issues
Score 5: Perfectly fluent, native-quality writing"""

_RUBRIC_COHERENCE = """\
Evaluate the COHERENCE of the generated Arabic story. ..."""
```

### 4.5 MetricSpecs for LLM-Judge

Each LLM-judge metric also gets its own MetricSpec:

```python
metric_specs.extend([
    MetricSpec(
        class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
        args={"metric_name": "fluency"},
    ),
    MetricSpec(
        class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
        args={"metric_name": "coherence"},
    ),
    # ... one per metric
])
```

### 4.6 Summary of the Per-Metric Design

| Old Design (grouped)                        | New Design (per-metric)                        |
|---------------------------------------------|------------------------------------------------|
| 1 AnnotatorSpec for all judge metrics       | 1 AnnotatorSpec per judge metric               |
| 1 judge call, asks for all scores at once   | 1 judge call per metric, focused rubric prompt |
| Generic "rate 1-5" prompt                   | Task+metric-specific rubric with score anchors |
| 1 MetricSpec reads all annotations          | 1 MetricSpec per metric reads its annotation   |
| Judge must parse/return JSON with N scores  | Judge returns a single integer score           |

---

## 5. Step-by-Step Execution Procedure

Follow these steps IN ORDER:

### Step 1: Read and Parse Registries

```python
import yaml

# Load all registry files
with open(".../registry_inference.yaml") as f:
    inference_registry = yaml.safe_load(f)
with open(".../registry_metrics.yaml") as f:
    metrics_registry = yaml.safe_load(f)
with open(".../registry_master.yaml") as f:
    master_registry = yaml.safe_load(f)

inference_defaults = inference_registry["defaults"]
all_datasets = set(inference_registry["datasets"].keys()) | set(metrics_registry["datasets"].keys())
```

### Step 2: For Each Dataset, Parse Its Scenario File

For each `<dataset_name>` in the union of both registries:

1. **Locate** `scenarios_new/<dataset_name>_scenario.py`
2. **Extract** by reading the file:
   - The Scenario class name (e.g. `class AaarScenario(Scenario):`)
   - The `name = "..."` attribute
   - The `SUBTASKS` list if present, or `__init__` parameters with subtask-like args
   - Whether the scenario embeds prompts in `Input.text` or expects HELM to construct them
   - Whether `TRAIN_SPLIT` is used (to set `max_train_instances`)
   - The task type (MCQA vs generation) from reference construction patterns
3. **If scenario file not found:** emit a `# TODO: scenario file missing` stub
   and continue.

### Step 3: Resolve Inference Config

For each dataset, merge `registry_inference.yaml` defaults with per-dataset
overrides as described in §3.4.

### Step 4: Build MetricSpecs

For each dataset's metrics in `registry_metrics.yaml`:

1. Collect all `in_helm: true` metrics → deduplicated MetricSpec list
2. Collect all `in_helm: false, type: llm_judge` metrics → **one AnnotatorSpec
   AND one MetricSpec PER metric** (not grouped — see §4.4)
3. Collect all `in_helm: false, type != llm_judge` metrics → individual MetricSpecs (search for class)
4. If a dataset has no entry in metrics registry → flag with `# TODO:` and use
   `BasicMetric` with `["exact_match"]` as fallback.

### Step 4b: Generate Rubric Registry (run ONCE before Step 6)

Before generating run spec files, produce `registry_rubrics.yaml` following
the procedure in §4.3. This only needs to happen once — all run spec files
read from the same rubric registry.

For each dataset that has LLM-judge metrics:

1. Read the scenario file docstring for task context
2. Check `data/metric_notes/<dataset_name>_eval_metrics_notes.md` for
   paper-defined rubrics
3. Read `registry_master.yaml` for paper/repo links (for context)
4. For each LLM-judge metric:
   - If a rubric is found in the notes/paper → use it, tag `[PAPER]`
   - If not → auto-generate a task-specific rubric, tag `[GENERATED]`
5. Write all rubrics to `registry_rubrics.yaml`

**Critical:** The rubric MUST be specific to the dataset's task. Example:
- "fluency" for `arastories` → "how natural and grammatically correct the
  Arabic story text is"
- "fluency" for `code_generation` → "how syntactically correct and
  idiomatically written the code is"

Same metric name, completely different rubrics.

### Step 5: Build AdapterSpec

Using resolved inference config + scenario analysis:

1. Set `method` per §3.1
2. Set `temperature`, `max_tokens`, `num_outputs` from inference config
3. Set `instructions`, `input_prefix`, `output_prefix` from scenario analysis
4. Set `max_train_instances` based on TRAIN_SPLIT presence
5. Set `stop_sequences` — use `["\n"]` if null in registry, UNLESS the task is
   open-ended generation (then use `[]` to avoid cutting off output)

### Step 6: Generate the Run Spec File

Write the Python file using the template from §2. For multi-subtask datasets,
generate one `@run_spec_function` per subtask.

### Step 7: Create the LLM-as-Judge Infrastructure

If ANY dataset uses LLM-as-Judge metrics, create the two generic classes
as described in `references/llm_judge_setup.md`. Only create these files
if they don't already exist.

**Important:** The annotator now handles **one metric at a time** with a
rubric-specific prompt. The classes in `references/llm_judge_setup.md` have
been updated to support this per-metric design.

Also verify that `registry_rubrics.yaml` was generated in Step 4b.

### Step 8: Generate a Summary Report

After generating all files, produce a summary at:
```
/home/public/vdeshpan/amazon-creativity-benchmark/run_specs/_generation_report.md
```

Contents:
- Total datasets processed
- Datasets with missing scenario files (TODOs)
- Datasets with missing metrics entries (TODOs)
- Datasets using LLM-as-Judge (count + list)
- Datasets with multiple subtasks (count + list)
- Rubrics: count of `[PAPER]` vs `[GENERATED]` rubrics
- Rubrics flagged for review (list)
- Any assumptions made or flags raised

---

## 6. Validation Checklist

Before finishing, verify for each generated file:

- [ ] File is valid Python (no syntax errors)
- [ ] Every `@run_spec_function` name is unique across ALL files
- [ ] `ScenarioSpec.class_name` matches the actual module path and class name
- [ ] `ScenarioSpec.args` matches the scenario's `__init__` signature
- [ ] MetricSpecs are deduplicated (no duplicate class+args combos)
- [ ] LLM-judge datasets have both AnnotatorSpec and the judge MetricSpec
- [ ] `RunSpec.name` includes subtask/param info for multi-subtask datasets
- [ ] All necessary imports are present
- [ ] `stop_sequences` is `[]` (not `["\n"]`) for open-ended generation tasks
- [ ] `num_outputs` > 1 only when `num_return_sequences` > 1 in inference config
- [ ] LLM-judge rubrics are task-specific (not generic "rate 1-5" prompts)
- [ ] Each LLM-judge metric has its own AnnotatorSpec (not grouped)
- [ ] `registry_rubrics.yaml` exists and covers all LLM-judge datasets
- [ ] Rubric strings in run spec files match `registry_rubrics.yaml`

---

## 7. Important Defaults and Assumptions

When information is missing, use these defaults and **always flag with a
comment** in the generated file:

| Missing Info              | Default                                          | Comment to Add                                |
|---------------------------|--------------------------------------------------|-----------------------------------------------|
| Scenario file not found   | Skip ScenarioSpec args                           | `# TODO: scenario file not found`             |
| Task type unclear         | `ADAPT_GENERATION`                               | `# ASSUMPTION: defaulting to generation`      |
| Instructions unclear      | `""`                                             | `# NOTE: scenario handles prompting internally` |
| Metrics not in registry   | `BasicMetric(["exact_match"])`                   | `# TODO: no metrics in registry, using fallback` |
| stop_sequences is null    | `[]` for open-ended, `["\n"]` for QA/MCQA        | *(no comment needed)*                         |
| max_train_instances       | `0` (zero-shot)                                  | `# ASSUMPTION: zero-shot, no TRAIN_SPLIT seen` |
| Custom metric class path  | Search project, else `# TODO`                    | `# TODO: custom metric class not found`       |

---

## 8. Reference Files

- **`references/llm_judge_setup.md`** — Full implementation code for
  `GenericLLMJudgeAnnotator` and `GenericLLMJudgeMetric`. Read this before
  generating any file that uses LLM-as-Judge metrics.
- **`references/example_aaar_run_specs.py`** — Gold-standard example of a
  completed run spec file for a multi-subtask dataset with LLM-judge metrics.

## 9. Generated Artifacts (outputs beyond run spec files)

| Artifact | Path | Purpose |
|----------|------|---------|
| Rubric registry | `.../data/registry/registry_rubrics.yaml` | All LLM-judge rubrics, tagged `[PAPER]` or `[GENERATED]` |
| LLM-judge annotator | `.../llm_judge/generic_llm_judge_annotator.py` | Reusable per-metric judge caller |
| LLM-judge metric | `.../llm_judge/generic_llm_judge_metric.py` | Reads judge annotations into HELM Stats |
| Generation report | `.../run_specs/_generation_report.md` | Summary of all decisions, flags, TODOs |