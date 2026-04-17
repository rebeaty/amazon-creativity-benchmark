## brainteaser — Fix Applied (Attempt 1, 2026-04-17)
- **Root cause**: Wrong MetricSpec class — run spec used `MultipleChoiceClassificationMetric`
  (produces `classification_macro_f1`/`classification_micro_f1`) instead of
  `compute_reference_metrics` (produces `exact_match`) as specified in the registry.
- **Files changed**: `run_specs/brainteaser_run_specs.py`
- **Change summary**: Replaced `class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric"` with `class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics"`
- **Result**: Code fixed; needs re-run on server (HELM request cache empty locally, API key required for `google/gemini-2.5-flash-lite`)
