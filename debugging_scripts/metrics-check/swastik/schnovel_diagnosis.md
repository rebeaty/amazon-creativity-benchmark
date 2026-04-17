# schnovel — Metrics Diagnosis (updated attempt 9)

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| accuracy | formula_based | true | helm.benchmark.metrics.basic_metrics.BasicMetric |

## Current Run Spec Metrics (as of attempt 9)
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})`

## Actual Stats.json Metrics (m2)
- `classification_macro_f1`
- `classification_micro_f1`

## Missing Metrics
- **accuracy**: `helm.benchmark.metrics.basic_metrics.BasicMetric` does **not exist** in the installed HELM version. Confirmed with: `ImportError: cannot import name 'BasicMetric' from 'helm.benchmark.metrics.basic_metrics'`. The class silently fails to load, so no `accuracy` metric is emitted. The `classification_macro_f1`/`classification_micro_f1` in actual stats come from HELM's internal MCQ adapter machinery.

## Root Cause
**Wrong class** — the registry-specified class `helm.benchmark.metrics.basic_metrics.BasicMetric` does not exist in the installed HELM package. Available classes in `basic_metrics.py` are `BasicGenerationMetric`, `BasicReferenceMetric`, `compute_reference_metrics`, etc. — no `BasicMetric`. The correct replacement is the repo-local `metrics.accuracy_metric.AccuracyMetric`, which produces `accuracy` and is already used by structurally identical MCQ datasets (`puntuguese`, `munch`).

## Proposed Fix
In `run_specs/schnovel_run_specs.py`, replace:
```python
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})
```
with:
```python
MetricSpec(class_name="metrics.accuracy_metric.AccuracyMetric", args={})
```
