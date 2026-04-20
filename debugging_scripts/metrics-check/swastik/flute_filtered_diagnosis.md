# flute_filtered — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| `exact_match` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |
| `f1` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |

## Current Run Spec Metrics (attempt 4 — before fix)
- `MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={})` ← correct
- `adapter_spec.method = ADAPT_GENERATION` ← correct (fixed in attempt 3)

## Actual Stats.json Metrics (m2)
- `classification_macro_f1`
- `classification_micro_f1`

## Missing Metrics
- **`exact_match`** and **`f1`**: Even with `ADAPT_GENERATION` and `compute_reference_metrics`, the scenario still produces two references per instance (one with `CORRECT_TAG`, one without). HELM detects this multi-reference MCQ pattern and routes through classification metrics instead of generation metrics.

## Root Cause
In `classification` mode, `FLUTEFilteredScenario.get_instances()` creates:
```python
references = [
    Reference(Output("Entailment"),    tags=[CORRECT_TAG] if label == "Entailment" else []),
    Reference(Output("Contradiction"), tags=[CORRECT_TAG] if label == "Contradiction" else []),
]
```
This two-reference pattern (one tagged correct, one without tags) is how HELM represents MCQ tasks. Even under `ADAPT_GENERATION`, `compute_reference_metrics` interprets the presence of a non-correct reference as a classification task and emits `classification_macro_f1`/`classification_micro_f1` instead of `exact_match`/`f1`.

## Proposed Fix
In `scenarios/flute_filtered_scenario.py`, change the classification-mode reference block to provide only the single correct answer:

```python
if self.mode == "classification":
    references = [
        Reference(Output(text=item["label"]), tags=[CORRECT_TAG])
    ]
```

With a single correct reference, `compute_reference_metrics` treats it as a standard generation evaluation and produces `exact_match` (string equality) and `f1` (token-level F1).
