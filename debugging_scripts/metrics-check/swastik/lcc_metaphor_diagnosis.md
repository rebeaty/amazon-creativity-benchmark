# lcc_metaphor — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| accuracy | formula_based | true | helm.benchmark.metrics.basic_metrics.BasicMetric |
| f1 | formula_based | true | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics |

## Current Run Spec Metrics (attempt 3)
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})` ✓
- `MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={})` ✓
- Adapter method: `ADAPT_MULTIPLE_CHOICE_JOINT` ← still wrong

## Actual Stats.json Metrics (m2)
- `classification_macro_f1`
- `classification_micro_f1`

## Missing Metrics
- **accuracy**: The MetricSpec is now correct (`BasicMetric`), but `ADAPT_MULTIPLE_CHOICE_JOINT` adapter causes `BasicMetric` to emit `classification_macro_f1`/`classification_micro_f1` instead of `accuracy`. The adapter needs to be `ADAPT_GENERATION`.
- **f1**: Same root cause — `ADAPT_MULTIPLE_CHOICE_JOINT` prevents `compute_reference_metrics` from producing the expected `f1` stat.

## Root Cause
Despite fixing the MetricSpecs in attempt 2, the adapter is still `ADAPT_MULTIPLE_CHOICE_JOINT`. Under this adapter, HELM treats the task as MCQ and `BasicMetric` emits classification-specific metric names. The scenario prompts for a "Yes or No" answer (generative), not a multiple-choice selection. Switching to `ADAPT_GENERATION` allows `BasicMetric` to produce `accuracy` and `compute_reference_metrics` to produce `f1`.

## Proposed Fix
In `run_specs/lcc_metaphor_run_specs.py`:
1. Change import from `ADAPT_MULTIPLE_CHOICE_JOINT` to `ADAPT_GENERATION`
2. Change adapter `method=ADAPT_GENERATION`
3. Lower `temperature` to `0.0` (binary classification)
4. Reduce `max_tokens` to `16` (only needs "Yes" or "No")
