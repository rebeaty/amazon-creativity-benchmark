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

1. `data/registry/registry_metrics.yaml` — find the dataset entry, note each metric's `type`, `helm_class`, `in_helm`, and any judge/model config
2. `run_specs/<dataset>_run_specs.py` — see what MetricSpecs are currently defined
3. `scenarios/<dataset>_scenario.py` or `scenarios_new/<dataset>_scenario.py` — check for metric-related logic

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
| ... | ... | ... | ... |

## Current Run Spec Metrics
- List what MetricSpecs are currently in the run_spec file

## Actual Stats.json Metrics
- List what m2 contains

## Missing Metrics
- List each missing metric and WHY it's missing

## Root Cause
One-paragraph explanation of why the metrics are missing.
Common causes:
- Run spec uses a TODO fallback (BasicGenerationMetric with exact_match) instead of the correct metric class
- Metric class is specified but wrong arguments are passed
- AnnotatorSpec needed for LLM-judge metrics but not configured
- Metric class doesn't exist or has wrong import path

## Proposed Fix
Describe exactly what changes to make in which files.
```

## Step 2: Fix the Code

Based on the diagnosis, edit the run_spec file (and scenario if needed):

### For `in_helm: true` + `type: formula_based` metrics:
These use standard HELM metric classes. Update the MetricSpec in the run_spec to use the correct `helm_class` from the registry.

### For `in_helm: true` + `type: model_based` metrics:
These need the correct HELM metric class plus potentially a `metric_model` argument.

### For `in_helm: false` + `type: llm_judge` metrics:
These require:
1. An `AnnotatorSpec` pointing to a judge annotator class
2. A corresponding `MetricSpec` that reads the annotator output
3. Check existing working examples (e.g., `alpaca_eval_2_run_specs.py`) for the pattern

### For `in_helm: false` + custom metrics:
Check if there's a custom metric class in `metrics/` directory. If not, note it as needing implementation.

## Rules

- ONLY modify files in `run_specs/`, `scenarios/`, `scenarios_new/`, `metrics/`, or `eval_scripts/`
- Do NOT modify HELM's installed package files
- Be surgical — only fix what's needed for the missing metrics
- If a metric requires a model/API that isn't available, note it in the diagnosis but don't add a broken MetricSpec
- Always verify the HELM class exists before referencing it:
  ```bash
  python3 -c "from <module_path> import <ClassName>"
  ```

## Key Patterns from Working Datasets

### BasicGenerationMetric (exact_match, f1_score, etc.)
```python
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "f1_score"]})
```

### SummarizationMetric (sentence_bert_*, rouge_*, bleu_*)
```python
MetricSpec(class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric", args={})
```

### LLM Judge via Annotator
```python
# In annotators list:
AnnotatorSpec(class_name="helm.benchmark.annotation.annotators...", args={...})
# In metric_specs:
MetricSpec(class_name="helm.benchmark.metrics.annotation_metrics.AnnotationMetric", args={"annotator_name": "..."})
```

### Classification Metrics
```python
MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.ClassificationMetric", args={})
```

## Workflow Context

This skill is the "fix" step in the metrics-check debugging loop:
1. Test (metrics-check-test) — found missing metrics
2. **Diagnose + Fix** (this skill) — identify cause and edit code
3. Re-run eval (`init_eval.sh`)
4. Loop back to step 1
