---
name: metrics-diagnose-fix
description: Diagnose why a dataset's trial run is missing expected metrics and fix the code. Reads run_spec, scenario, and registry files to identify the root cause, writes a diagnosis markdown file, then edits run_specs/scenarios to produce the correct metrics. Use when metrics-check-test reports missing metrics.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
user-invocable: true
---

# Metrics Diagnose & Fix

When a dataset's trial run is missing expected metrics, this skill diagnoses the root cause and fixes the code.

## Inputs

You will receive:
- **dataset**: the dataset name
- **missing**: list of metric names that are expected but not in stats.json
- **m1**: full list of expected metrics (from registry)
- **m2**: full list of actual metrics (from stats.json)
- **assignee**: the person's name (for writing diagnosis to their directory)

## Step 1: Diagnose (write .md file)

Read these files to understand the current state:

1. `data/registry/registry_metrics.yaml` — find the dataset entry. For each missing metric, note its `name`, `type`, `in_helm`, `helm_class`, and any judge/annotator config fields (`judge_model_name`, `judge_temperature`, `judge_max_new_tokens`, `judge_prompt`).
2. `run_specs/<dataset>_run_specs.py` — see what MetricSpecs and AnnotatorSpecs are currently defined.
3. Scenario file — only read this if the run_spec looks correct but the metric is still missing (i.e. diagnosing a **Scenario issue**). Check in order:
   - `scenarios/<dataset>_scenario.py` — primary location on the server (per CLAUDE.md)
   - `scenarios/<dataset>_scenario.py` — fallback (present in the git clone)
   - **Note:** The `scenarios.*` prefix in run_spec `class_name` fields refers to the `scenarios/` directory. On the actual evaluation server (`/home/public/vdeshpan/...`) this directory exists as `scenarios/`.

Then write a diagnosis file to:
```
debugging_scripts/metrics-check/<assignee>/<dataset>_diagnosis.md
```

The diagnosis must contain:

```markdown
# <dataset> — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| <name> | formula_based/model_based/llm_judge | true/false | <class or null> |

## Current Run Spec Metrics
- List what MetricSpecs and AnnotatorSpecs are currently in the run_spec file

## Actual Stats.json Metrics (m2)
- List what m2 contains

## Missing Metrics
For each missing metric, state WHY it is missing (see root cause categories below).

## Root Cause
One-paragraph explanation of why the metrics are missing.

## Proposed Fix
Describe exactly what changes to make in which files.
```

### Root Cause Categories

| Category | How to recognize it | Fix |
|----------|--------------------|----|
| **TODO fallback** | run_spec has `# TODO: no metrics in registry, using fallback` and only `BasicGenerationMetric` with `exact_match` | Replace with correct MetricSpec(s) |
| **Missing llm_judge wiring** | Registry says `type: llm_judge` but no `GenericLLMJudgeMetric` / `GenericLLMJudgeAnnotator` in run_spec | Add AnnotatorSpec + MetricSpec pair, write rubric |
| **Wrong class** | MetricSpec `class_name` doesn't match what the registry or pattern requires | Fix `class_name` |
| **Missing annotators field update** | AnnotatorSpec added but `annotators=None` still in RunSpec constructor | Change `annotators=None` to `annotators=[...]` |
| **Wrong constructor args** | MetricSpec exists with the right `class_name` but `args={}` for a class that requires args (e.g. `JSDMetric`, `CorrelationMetric`, `DistinctNMetric`, `VietnamesePoemMetric`) | Fix `args` to pass the required constructor parameter |
| **Unimplementable metric** | Registry has `in_helm: false`, `type: model_based`, `helm_class: null` | Skip — add `# TODO: model_based, helm_class=null` comment, note in diagnosis |
| **Scenario issue** | run_spec looks correct but scenario doesn't emit `Reference` objects the metric expects | Fix scenario file |

---

## Step 2: Fix the Code

Apply the correct pattern based on each missing metric's `type`, `in_helm`, and `helm_class` in the registry.

---

### Pattern A — `in_helm: true`, any type (standard HELM class)

**Use the `helm_class` value from the registry directly as `class_name`.** Do not substitute a different class even if the existing run_spec uses one — the registry is the source of truth.

Common `in_helm: true` classes and what metric names they produce:

| Registry `helm_class` | Metric names produced | `args` |
|---|---|---|
| `helm.benchmark.metrics.basic_metrics.BasicMetric` | `accuracy` | `{}` |
| `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` | `exact_match`, `bleu_4` | `{}` |
| `helm.benchmark.metrics.summarization_metrics.SummarizationMetric` | `sentence_bert_f1`, `sentence_bert_precision`, `sentence_bert_recall`, `rouge_*`, `bleu_*` — all at once from one MetricSpec | `{}` |
| `helm.benchmark.metrics.disinformation_metrics.DisinformationMetric` | `self_bleu` | `{}` |
| `helm.benchmark.metrics.image_generation.clip_score_metrics.CLIPScoreMetric` | `clip_score` | `{}` |

These are the **only five** `in_helm: true` `helm_class` values that appear in `registry_metrics.yaml`. The registry never uses `BasicGenerationMetric`, `ClassificationMetric`, or `MultipleChoiceClassificationMetric` as a `helm_class` — those appear in existing run_specs as wrong-class bugs, not as valid registry-directed fixes.

---

### Pattern B — `in_helm: false`, `type: formula_based`, `helm_class` is non-null

These use custom classes from the `metrics/` directory. Use the `helm_class` from the registry as `class_name`. **Several of these classes require constructor args** — passing `args={}` for them will crash at runtime with a `TypeError`.

**Constructor args rule:** derive the required arg directly from the metric `name` in the registry:

| Registry metric name | class_name | args to pass |
|---|---|---|
| `distinct_1` | `metrics.distinct_n_metric.DistinctNMetric` | `{"n": 1}` |
| `distinct_2` | `metrics.distinct_n_metric.DistinctNMetric` | `{"n": 2}` |
| `jensen_shannon_divergence_unigram` | `metrics.jsd_metric.JSDMetric` | `{"n": 1}` |
| `jensen_shannon_divergence_bigram` | `metrics.jsd_metric.JSDMetric` | `{"n": 2}` |
| `pearson_correlation` | `metrics.correlation_metric.CorrelationMetric` | `{"correlation_type": "pearson"}` |
| `spearman_correlation` | `metrics.correlation_metric.CorrelationMetric` | `{"correlation_type": "spearman"}` |
| `poem_score` / `length_score` / `tone_score` / `rhyme_score` | `metrics.vietnamese_poem_metric.VietnamesePoemMetric` | `{"metric": "<name>"}` e.g. `{"metric": "rhyme_score"}` |
| `creativity_score` | `metrics.creativity_score_metric.CreativityScoreMetric` | `{}` (no required args) |
| `type_token_ratio` | `metrics.type_token_ratio_metric.TypeTokenRatioMetric` | `{}` |
| `mean_absolute_error` | `metrics.mean_absolute_error_metric.MeanAbsoluteErrorMetric` | `{}` |
| `group_match_score` | `metrics.group_match_score_metric.GroupMatchScoreMetric` | `{}` |
| `json_validity` | `metrics.json_validity_metric.JsonValidityMetric` | `{}` |
| `xml_validity` | `metrics.xml_validity_metric.XmlValidityMetric` | `{}` |
| `validity` | `metrics.validity_metric.ValidityMetric` | `{}` |
| `array_dimensions` | `metrics.array_dimensions_metric.ArrayDimensionsMetric` | `{}` |
| `iou_score` | `metrics.iou_score_metric.IoUScoreMetric` | `{}` |
| `classification_accuracy` | `metrics.classification_accuracy_metric.ClassificationAccuracyMetric` | `{}` |
| `constraint_satisfaction` | `metrics.constraint_satisfaction_metric.ConstraintSatisfactionMetric` | `{}` |
| `layout_quality` | `metrics.layout_quality_metric.LayoutQualityMetric` | `{}` |
| `pass_at_1` | `metrics.pass_at_1_metric.PassAt1Metric` | `{}` |
| `percentile_rank` | `metrics.percentile_rank_metric.PercentileRankMetric` | `{}` |
| `validity_score` | `metrics.validity_score_metric.ValidityScoreMetric` | `{}` |

**When the same class appears multiple times** (for `distinct_1`/`distinct_2`, JSD unigram/bigram, `pearson`/`spearman`, Vietnamese poem sub-scores), add one `MetricSpec` per metric name, each with its own args.

**Also check existing MetricSpecs for this bug.** Some run_specs already have the right `class_name` but `args={}` for a class that requires args (e.g. `CorrelationMetric(args={})`, `JSDMetric(args={})`). If a MetricSpec exists but `m2` doesn't contain the expected metric name, check whether the args are correct — wrong args causes a runtime crash that silently produces no output.

To confirm the class file exists: `ls metrics/<module_name>.py`

---

### Pattern C — `in_helm: false`, `type: llm_judge`

**This is the most common and most complex case.** It requires:
1. One `AnnotatorSpec` per metric dimension using `GenericLLMJudgeAnnotator`
2. One `MetricSpec` per metric dimension using `GenericLLMJudgeMetric`
3. A rubric string constant (`_RUBRIC_<METRIC_NAME>`) defined at the top of the run_spec
4. The `annotators` field in `RunSpec(...)` must be a list, **not** `None`

**The correct class names for this repo are:**
```python
# MetricSpec class:
"llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric"

# AnnotatorSpec class:
"llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator"
```

**Full pattern:**

**Rubric writing rules** (the annotator hardcodes its own `Instruction:` / `Generated response:` / `Provide only the integer score:` framing — the rubric is prepended before all of that):
- Do **not** include `{input_text}`, `{generated_response}`, or any placeholders — the annotator fills those in itself
- Do **not** include "Instruction:", "Response:", or "Score:" labels — the annotator adds those
- **Do** include: one sentence describing what to evaluate, then score anchor points
- Use the score range from the registry `judge_prompt` if it contains one (e.g. 0–100, 1–5, 1–10); otherwise default to 1–5
- If `judge_prompt` is non-null, extract the scoring criteria from it but strip all placeholders and framing; rewrite as criteria-only

```python
from helm.benchmark.annotation.annotator import AnnotatorSpec

# 1. Define the rubric at module level (above the @run_spec_function)
# Criteria + score anchors only — NO {placeholders}, NO "Instruction:/Response:" labels
_RUBRIC_<METRIC_NAME_UPPER> = """\
Evaluate the <DIMENSION> of the generated response.

Score 1: <worst anchor description>
Score 3: <mid anchor description>
Score 5: <best anchor description>
"""

# 2. In metric_specs:
MetricSpec(
    class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
    args={"metric_name": "<metric_name>"}
)

# 3. In annotators list (one per metric):
AnnotatorSpec(
    class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
    args={
        "judge_model_name": "<from registry: judge_model_name>",
        "judge_temperature": <from registry: judge_temperature>,
        "judge_max_new_tokens": <from registry: judge_max_new_tokens>,
        "metric_name": "<metric_name>",   # must match MetricSpec above
        "rubric": _RUBRIC_<METRIC_NAME_UPPER>,
    },
)

# 4. In RunSpec(...):
return RunSpec(
    ...
    metric_specs=metric_specs,
    annotators=annotators,   # NOT None — must be the list
)
```

**Working reference:** `run_specs/alpaca_eval_2_run_specs.py` (1 metric), `run_specs/aaar_run_specs.py` (2 metrics, 2 subtasks), `run_specs/tinyfabulist_run_specs.py` (2 metrics, no reference answer needed).

**If the run_spec currently has `annotators=None`:** You must change it to `annotators=annotators` (or `annotators=[...]` inline). This is the most commonly missed step.

---

### Pattern D — `in_helm: false`, `type: model_based`, `helm_class: null`

These metrics require an external model (e.g., FastText, ESMFold, SentenceTransformers) that cannot be wired as a standard MetricSpec. **Do not attempt to implement these.**

Instead:
- Leave a comment in the run_spec at the metric position:
  ```python
  # TODO: model_based metric '<metric_name>' requires external model '<metric_model>' — not implementable as MetricSpec
  ```
- Note this in the diagnosis file under "Missing Metrics" and "Root Cause"
- This metric will remain missing — that is acceptable and expected

**If ALL missing metrics for the dataset are Pattern D (nothing is implementable):**
- Do **not** remove the existing `BasicGenerationMetric` fallback — keep it so the eval doesn't crash with zero MetricSpecs
- Add the `# TODO: model_based...` comment alongside it
- Write the diagnosis noting all metrics are unimplementable
- Do **not** make any further fix attempts — the orchestrator will exhaust its retries, which is the correct outcome for these datasets (`dat`, `noveltybench`, `sdat` are known examples)

---

## Verify imports are present

When adding Pattern C, make sure this import is at the top of the run_spec (add it if missing):

```python
from helm.benchmark.annotation.annotator import AnnotatorSpec
```

Pattern B custom classes don't need explicit imports — they are loaded by HELM's class registry via `class_name` string.

## Verify Python syntax after editing

```bash
python3 -m py_compile run_specs/<dataset>_run_specs.py && echo "Syntax OK"
```

Fix any syntax errors before returning. A broken file will crash the entire eval.

---

## Rules

- **Only modify** files in `run_specs/`, `scenarios/`, `scenarios/`, `metrics/`, or `eval_scripts/`
- **Do NOT modify** HELM's installed package files (anything under `site-packages/`)
- **Do NOT modify** `data/registry/registry_metrics.yaml` — it is the source of truth. If it looks wrong, note it in the diagnosis and ask Vijeta.
- **Be surgical** — only fix MetricSpecs/AnnotatorSpecs for the missing metrics. Do not touch working ones.
- Always write the diagnosis file **before** making code changes.
- Do NOT run the evaluation — return to the orchestrator after editing.

---

## Output Contract

When this skill is done:
1. `debugging_scripts/metrics-check/<assignee>/<dataset>_diagnosis.md` — written
2. `run_specs/<dataset>_run_specs.py` — edited with fixes applied (or commented TODO for unimplementable metrics)
3. `python3 -m py_compile run_specs/<dataset>_run_specs.py` passes

---

## Workflow Context

This skill is the "fix" step in the metrics-check debugging loop:
1. Test (`metrics-check-test`) — found missing metrics
2. **Diagnose + Fix** (this skill) — identify cause and edit code
3. Re-run eval (`init_eval.sh`)
4. Loop back to step 1
