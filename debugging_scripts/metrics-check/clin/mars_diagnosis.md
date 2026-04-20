# Diagnosis: mars

## Expected vs Actual Metrics

- **Expected (m1)**: `["accuracy"]`
- **Actual (m2)**: `["classification_macro_f1", "classification_micro_f1"]`
- **Missing**: `["accuracy"]`

## Root Cause

The run_spec uses `MultipleChoiceClassificationMetric` with `ADAPT_MULTIPLE_CHOICE_JOINT`, but MARS is an **open-ended generation** task (analogical entity prediction), not a multiple-choice task.

- `MultipleChoiceClassificationMetric` produces `classification_macro_f1` and `classification_micro_f1` — not `accuracy`.
- The annotator notes explicitly state: "MARS uses open-ended generation with exact match evaluation."
- The scenario creates instances with free-form text references (entity ID + entity name), not labeled A/B/C/D choices.
- `BasicGenerationMetric(names=["exact_match"])` computes `exact_match` under the `accuracy` group, which is what the registry expects.

## Proposed Fix

1. Change adapter from `ADAPT_MULTIPLE_CHOICE_JOINT` → `ADAPT_GENERATION`
2. Replace `MultipleChoiceClassificationMetric` → `BasicGenerationMetric` with `names=["exact_match"]`
3. Adjust adapter settings for generation (lower temperature, no stop sequences on \n)
