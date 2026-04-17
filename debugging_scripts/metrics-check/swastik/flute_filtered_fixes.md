# flute_filtered — Fixes Summary

## Files Modified
- `scenarios/flute_filtered_scenario.py`

## Changes Made (attempt 4)

In `FLUTEFilteredScenario.get_instances()`, replaced the two-reference MCQ pattern with a single correct reference in classification mode.

**Before:**
```python
if self.mode == "classification":
    references = [
        Reference(Output(text="Entailment"), tags=[CORRECT_TAG] if item["label"] == "Entailment" else []),
        Reference(Output(text="Contradiction"), tags=[CORRECT_TAG] if item["label"] == "Contradiction" else []),
    ]
```

**After:**
```python
if self.mode == "classification":
    references = [
        Reference(Output(text=item["label"]), tags=[CORRECT_TAG]),
    ]
```

## Why

Attempts 1–3 fixed the adapter (`ADAPT_GENERATION`) and MetricSpec (`compute_reference_metrics`), but the scenario still emitted two references — one correct and one without tags. HELM's `compute_reference_metrics` treats any instance with multiple references (MCQ pattern) as a classification task, producing `classification_macro_f1`/`classification_micro_f1` instead of `exact_match`/`f1`.

Providing only the single correct reference (`item["label"]`) forces `compute_reference_metrics` into standard generation-eval mode, which produces the required `exact_match` and token-level `f1`.

## Verification
`python3 -m py_compile scenarios/flute_filtered_scenario.py` passes.
`python3 -m py_compile run_specs/flute_filtered_run_specs.py` passes.
