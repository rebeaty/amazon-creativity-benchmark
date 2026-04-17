---
name: metrics-fix-loop
description: >
  Iteratively fix metrics mismatches for all pending datasets assigned to a person.
  For each dataset: checks metrics coverage, diagnoses root cause, edits run_spec/scenario/metrics,
  re-runs the eval, and repeats until all expected metrics pass or max attempts exhausted.
  Full repo context is embedded — no external skill calls needed.
  Trigger with: /metrics-fix-loop <assignee> [dataset]
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
user-invocable: true
---

# Metrics Fix Loop

Run the complete metrics-check → diagnose → fix → re-eval loop for an assignee's pending datasets.

## Trigger Format

```
/metrics-fix-loop <assignee>            # all pending datasets
/metrics-fix-loop <assignee> <dataset>  # single dataset
```

`assignee` must match a key in `debugging_scripts/metrics-check/debug_assignments_pending.json`
(case-sensitive: `rajkumar`, `vijeta`, `clin`, `swastik`).

---

## Repo Architecture (read this before starting)

```
amazon-creativity-benchmark/
├── scenarios/                        # HELM Scenario classes (~160 files)
├── run_specs/                        # HELM RunSpec functions (~160 files)
├── metrics/                          # Custom metric classes (formula-based)
├── llm_judge/                        # LLM-judge annotator + metric classes
├── eval_scripts/                     # Per-dataset bash eval wrappers
├── data/registry/
│   ├── registry_metrics.yaml         # Ground truth: expected metrics per dataset
│   ├── registry_inference.yaml       # Inference config per dataset
│   └── registry_master.yaml          # Dataset metadata (paper links, modality, etc.)
├── outputs/first-trial-run/trial/    # Phase 1 eval results (168 run dirs)
├── benchmark_output/runs/trial/      # Fresh eval output (written by eval_scripts)
└── debugging_scripts/metrics-check/
    ├── metrics_check.py              # Compares m1 vs m2, exits 0/1/2
    ├── _debug_helper.py              # Bookkeeping: next/done/status
    ├── init_eval.sh                  # Runs eval for one dataset (MODEL=google/gemini-2.5-flash-lite)
    ├── debug_assignments_pending.json
    └── debug_assignments_done.json
```

**Output path note:** `metrics_check.py` checks BOTH `outputs/first-trial-run/trial/` (Phase 1 archive)
and `benchmark_output/runs/trial/` (fresh runs). When you re-run eval after a fix, the new stats
land in `benchmark_output/runs/trial/` — metrics_check.py will find them there.

---

## Loop Pseudocode

```
MAX_ATTEMPTS = 10

for each dataset in pending[assignee]:
    for attempt in 1..MAX_ATTEMPTS:
        result = run metrics_check.py <dataset>

        if result.status == "pass":
            mark_done(assignee, dataset)
            break

        if result.status == "no_registry":
            print "SKIP — not in registry, tell Vijeta"
            break

        if result.status == "no_stats":
            run init_eval.sh <assignee> <dataset>
            continue   # re-check after eval

        # status == "fail": diagnose, fix, re-eval
        diagnose_and_fix(dataset, result.missing, result.m1, result.m2, assignee, attempt)
        run init_eval.sh <assignee> <dataset>

    if exhausted attempts:
        print "FAILED — log last error"
```

---

## Step 1: Get Pending Datasets

```bash
python3 debugging_scripts/metrics-check/metrics_check.py <dataset>   # JSON output
python3 debugging_scripts/metrics-check/_debug_helper.py status <assignee>
```

Read `debugging_scripts/metrics-check/debug_assignments_pending.json` to get the list.

---

## Step 2: Check Metrics

```bash
python3 debugging_scripts/metrics-check/metrics_check.py <dataset>
```

Exit codes: `0` = pass, `1` = metrics missing, `2` = no registry entry or no stats.json.
JSON output keys: `dataset`, `m1` (expected), `m2` (actual), `missing`, `stats_files`, `status`.

---

## Step 3: Diagnose & Fix

When `status == "fail"`, diagnose and fix in this order:

### 3.1 Read the registry entry

```bash
grep -A 40 "^  <dataset>:" data/registry/registry_metrics.yaml
```

For each missing metric note: `name`, `type`, `in_helm`, `helm_class`, and any judge fields
(`judge_model_name`, `judge_temperature`, `judge_max_new_tokens`, `judge_prompt`).

### 3.2 Read the run spec

```
run_specs/<dataset>_run_specs.py
```

Look for existing `MetricSpec` and `AnnotatorSpec` entries.

### 3.3 Read the scenario (only if run spec looks correct)

```
scenarios/<dataset>_scenario.py
```

Check if `Reference` objects are emitted correctly for metrics that require them.

### 3.4 Write diagnosis file

Path: `debugging_scripts/metrics-check/<assignee>/<dataset>_diagnosis.md`

```markdown
# <dataset> — Metrics Diagnosis (Attempt N)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
...

## Run Spec Currently Has
...

## Actual Output (m2)
...

## Missing
...

## Root Cause
...

## Proposed Fix
...
```

### 3.5 Apply the fix

Use the correct pattern based on the registry:

---

#### Pattern A — `in_helm: true` (standard HELM class)

Use `helm_class` from registry verbatim as `class_name`. Known classes:

| helm_class | Metrics produced |
|---|---|
| `helm.benchmark.metrics.basic_metrics.BasicMetric` | `accuracy` |
| `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` | `exact_match`, `bleu_4` |
| `helm.benchmark.metrics.summarization_metrics.SummarizationMetric` | `sentence_bert_f1/precision/recall`, `rouge_*`, `bleu_*` |
| `helm.benchmark.metrics.disinformation_metrics.DisinformationMetric` | `self_bleu` |
| `helm.benchmark.metrics.image_generation.clip_score_metrics.CLIPScoreMetric` | `clip_score` |

```python
MetricSpec(class_name="<helm_class_from_registry>", args={})
```

---

#### Pattern B — `in_helm: false`, `type: formula_based`, non-null `helm_class`

Custom classes in `metrics/`. Several require constructor args — use the table:

| metric name | class_name | args |
|---|---|---|
| `distinct_1` | `metrics.distinct_n_metric.DistinctNMetric` | `{"n": 1}` |
| `distinct_2` | `metrics.distinct_n_metric.DistinctNMetric` | `{"n": 2}` |
| `jensen_shannon_divergence_unigram` | `metrics.jsd_metric.JSDMetric` | `{"n": 1}` |
| `jensen_shannon_divergence_bigram` | `metrics.jsd_metric.JSDMetric` | `{"n": 2}` |
| `pearson_correlation` | `metrics.correlation_metric.CorrelationMetric` | `{"correlation_type": "pearson"}` |
| `spearman_correlation` | `metrics.correlation_metric.CorrelationMetric` | `{"correlation_type": "spearman"}` |
| `poem_score` / `length_score` / `tone_score` / `rhyme_score` | `metrics.vietnamese_poem_metric.VietnamesePoemMetric` | `{"metric": "<name>"}` |
| `creativity_score` | `metrics.creativity_score_metric.CreativityScoreMetric` | `{}` |
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

Verify the class file exists before referencing it:
```bash
ls metrics/<module_name>.py
```

When multiple metrics share a class (e.g. `distinct_1`/`distinct_2`), add one `MetricSpec` per metric with its own args.

Also check existing MetricSpecs with the right class but `args={}` — this causes a silent runtime crash.

---

#### Pattern C — `in_helm: false`, `type: llm_judge`

Requires one AnnotatorSpec + one MetricSpec per dimension, plus a rubric string constant.

```python
from helm.benchmark.annotation.annotator import AnnotatorSpec

_RUBRIC_<METRIC_UPPER> = """\
Evaluate the <dimension> of the response.

Score 1: <worst>
Score 3: <mid>
Score 5: <best>
"""

# In metric_specs:
MetricSpec(
    class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
    args={"metric_name": "<metric_name>"}
)

# In annotators list:
AnnotatorSpec(
    class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
    args={
        "judge_model_name": "<from registry>",
        "judge_temperature": <from registry>,
        "judge_max_new_tokens": <from registry>,
        "metric_name": "<metric_name>",
        "rubric": _RUBRIC_<METRIC_UPPER>,
    },
)

# In RunSpec(...): change annotators=None → annotators=annotators
```

Rubric rules: no `{placeholders}`, no "Instruction:/Response:" labels — the annotator adds those.
Use score range from registry `judge_prompt` if present; default to 1–5.

Reference working examples: `run_specs/alpaca_eval_2_run_specs.py`, `run_specs/aaar_run_specs.py`.

---

#### Pattern D — `in_helm: false`, `type: model_based`, `helm_class: null`

Cannot be implemented as a MetricSpec. Add a TODO comment and skip:

```python
# TODO: model_based metric '<metric_name>' requires external model — not implementable as MetricSpec
```

If ALL missing metrics are Pattern D, keep the existing `BasicGenerationMetric` fallback (prevents zero-MetricSpec crash), add the TODO, and let the loop exhaust its attempts naturally.

---

### 3.6 Verify syntax after every edit

```bash
python3 -m py_compile run_specs/<dataset>_run_specs.py && echo "Syntax OK"
```

Fix all syntax errors before proceeding. A broken file crashes the entire eval.

---

## Step 4: Re-run Eval

```bash
bash debugging_scripts/metrics-check/init_eval.sh <assignee> <dataset>
```

This runs `eval_scripts/<dataset>.sh` with `MODEL=google/gemini-2.5-flash-lite`, `SUITE=trial`,
`MAX_INSTANCES=10`. Output goes to `benchmark_output/runs/trial/`.

If `init_eval.sh` exits with code `2`, it's a data access error — skip the dataset.
If it exits with code `1`, the eval failed — check the log and attempt another fix.

---

## Step 5: Mark Done

When `metrics_check.py` exits 0:

```bash
python3 debugging_scripts/metrics-check/_debug_helper.py done <assignee> <dataset>
```

---

## Step 6: Write a Fixes Summary

After fixing, append to `debugging_scripts/metrics-check/<assignee>/<dataset>_fixes.md`:

```markdown
## <dataset> — Fix Applied (Attempt N, <date>)
- **Root cause**: <one line>
- **Files changed**: `run_specs/<dataset>_run_specs.py` [, others]
- **Change summary**: <what was added/corrected>
- **Result**: PASS / FAIL
```

---

## Rules

- **Only modify:** `run_specs/`, `scenarios/`, `metrics/`, `eval_scripts/`
- **Never modify:** `data/registry/registry_metrics.yaml` (source of truth — if wrong, note in diagnosis and tell Vijeta)
- **Never modify:** HELM's installed package files (`site-packages/`)
- **Be surgical:** only touch MetricSpecs/AnnotatorSpecs for the missing metrics
- Always write the diagnosis file **before** making any code change
- Verify Python syntax after every edit
- Do not remove working MetricSpecs while fixing broken ones

---

## Common Bugs Quick Reference

| Symptom | Fix |
|---|---|
| run_spec has `# TODO: no metrics in registry, using fallback` | Replace with correct MetricSpec(s) |
| `annotators=None` in RunSpec | Change to `annotators=annotators` |
| MetricSpec has right class_name but wrong `args={}` | Add correct args (see Pattern B table) |
| Missing `from helm.benchmark.annotation.annotator import AnnotatorSpec` | Add import |
| Scenario doesn't emit `Reference` objects | Fix scenario file |
| metric `win_rate` missing | Needs GenericLLMJudgeAnnotator + GenericLLMJudgeMetric (Pattern C) |

---

## Output Contract

After running this skill:
1. All processable datasets moved to `debug_assignments_done.json`
2. `debugging_scripts/metrics-check/<assignee>/<dataset>_diagnosis.md` written for each fixed dataset
3. `debugging_scripts/metrics-check/<assignee>/<dataset>_fixes.md` written for each fixed dataset
4. `debugging_scripts/metrics-check/<assignee>/eval_run_learnings.md` updated with session summary
5. All modified `run_specs/` files pass `py_compile`
6. Final status printed: N passed, M failed, K skipped
