# puntuguese — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| `accuracy` | formula_based | true | `helm.benchmark.metrics.basic_metrics.BasicMetric` |
| `f1` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |

## Current Run Spec Metrics
- Two identical `MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={})` (duplicated)
- No `AnnotatorSpec`s

## Actual Stats.json Metrics (m2)
- `classification_macro_f1`
- `classification_micro_f1`

## Missing Metrics
- **`accuracy`**: The run_spec uses `MultipleChoiceClassificationMetric`, which emits `classification_macro_f1` and `classification_micro_f1`, not `accuracy`. No `AccuracyMetric` is wired.
- **`f1`**: Same root cause — `MultipleChoiceClassificationMetric` does not emit a stat named `f1`. No `F1Metric` is wired.

## Root Cause
**Wrong class (and duplicate).** The run_spec uses `MultipleChoiceClassificationMetric` twice (duplicated, producing the same two classification F1 stats). This class emits `classification_macro_f1` and `classification_micro_f1`, which is wrong for this dataset. The registry expects `accuracy` and `f1`. Additionally, the registry's listed `helm_class` values (`BasicMetric`, `compute_reference_metrics`) do not exist in the installed HELM package under those names. The custom classes `metrics.accuracy_metric.AccuracyMetric` and `metrics.f1_metric.F1Metric` exist in the codebase and emit exactly the required stat names.

## Proposed Fix
Replace both `MultipleChoiceClassificationMetric` MetricSpecs in `run_specs/puntuguese_run_specs.py` with:
1. `MetricSpec(class_name="metrics.accuracy_metric.AccuracyMetric", args={})` → emits `accuracy`
2. `MetricSpec(class_name="metrics.f1_metric.F1Metric", args={})` → emits `f1`
