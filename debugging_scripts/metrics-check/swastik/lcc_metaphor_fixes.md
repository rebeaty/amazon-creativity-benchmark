# lcc_metaphor — Fixes Summary

## Attempt 3 Fix

### File Modified
`run_specs/lcc_metaphor_run_specs.py`

### Changes
1. Changed adapter method from `ADAPT_MULTIPLE_CHOICE_JOINT` → `ADAPT_GENERATION`
2. Updated import accordingly
3. Set `temperature=0.0` (binary classification task)
4. Set `max_tokens=16` (only needs "Yes" or "No")
5. Cleared `output_prefix` (was `"Answer: "`, redundant since prompt already ends with "Answer (Yes or No):")

### Why
The MetricSpecs were already correct after attempt 2 (`BasicMetric` + `compute_reference_metrics`), but `ADAPT_MULTIPLE_CHOICE_JOINT` was still set. Under MCQ adapter, `BasicMetric` emits `classification_macro_f1`/`classification_micro_f1` instead of `accuracy`. Switching to `ADAPT_GENERATION` allows `BasicMetric` to produce `accuracy` and `compute_reference_metrics` to produce `f1`. The scenario prompt is generative ("Answer (Yes or No):"), so `ADAPT_GENERATION` is semantically correct.

### Verification
`python3 -m py_compile run_specs/lcc_metaphor_run_specs.py` — Syntax OK

---

## History

### Attempt 2 Fix
Replaced two duplicate `MultipleChoiceClassificationMetric` MetricSpecs with correct classes from registry:
- `helm.benchmark.metrics.basic_metrics.BasicMetric` → produces `accuracy`
- `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` → produces `f1`

### Attempt 1 Fix
(Did not persist — run_spec reverted to single `MultipleChoiceClassificationMetric`)
