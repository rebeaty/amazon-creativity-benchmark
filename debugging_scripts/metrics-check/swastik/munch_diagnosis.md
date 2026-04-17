# munch — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| accuracy | formula_based | true | helm.benchmark.metrics.basic_metrics.BasicMetric |

## Current Run Spec Metrics (after attempt 7)
- `MetricSpec(class_name="metrics.accuracy_metric.AccuracyMetric", args={})`

## Actual Stats.json Metrics (m2) — pre-fix
- `classification_macro_f1`
- `classification_micro_f1`

## Missing Metrics
- **accuracy**

## Root Cause (confirmed at attempt 7)
The registry's `helm_class` of `helm.benchmark.metrics.basic_metrics.BasicMetric` does not exist in the installed HELM package — confirmed with:
```
ImportError: cannot import name 'BasicMetric' from 'helm.benchmark.metrics.basic_metrics'
```
When the class cannot be found, HELM silently skips the MetricSpec. The `classification_macro_f1`/`classification_micro_f1` stats come from HELM's default MCQ fallback behavior.

No HELM built-in metric produces a stat named `accuracy`. The existing custom `ClassificationAccuracyMetric` produces `classification_accuracy` (different name) and also has a bug where it checks `references[0]` instead of the CORRECT_TAG reference.

The adapter was already corrected to `ADAPT_GENERATION` in attempt 6. The remaining missing piece was a valid custom metric that produces `Stat(MetricName("accuracy"))`.

## Fix Applied (attempt 7)
1. Created `metrics/accuracy_metric.py` with `AccuracyMetric` class that:
   - Finds the reference tagged with `CORRECT_TAG`
   - Strips and case-insensitively compares the completion to the correct answer
   - Returns `Stat(MetricName("accuracy"))`
2. Updated `run_specs/munch_run_specs.py` to use `metrics.accuracy_metric.AccuracyMetric`
