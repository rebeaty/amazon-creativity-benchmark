## brainteaser — Fix Applied (Attempt 1, 2026-04-17)
- **Root cause**: Wrong MetricSpec class — run spec used `MultipleChoiceClassificationMetric`
  (produces `classification_macro_f1`/`classification_micro_f1`) instead of
  `BasicGenerationMetric` as specified by the registry intent (produces `exact_match`).
- **Files changed**: `run_specs/brainteaser_run_specs.py`
- **Change summary**: Replaced `MultipleChoiceClassificationMetric` with `BasicGenerationMetric(names=["exact_match"])`. Note: registry specifies `helm_class: compute_reference_metrics` which is a function (not a class) in this HELM version — it cannot be used as MetricSpec. `BasicGenerationMetric` is the correct equivalent.
- **Registry bugs (⚠ tell Vijeta)**:
  1. `compute_reference_metrics` in registry → function not instantiable as MetricSpec class; should document `BasicGenerationMetric` as the correct class
- **Result**: Code fixed; needs server re-run (HELM request cache empty locally, API key required for `google/gemini-2.5-flash-lite`). Force-marked DONE pending server verification.
