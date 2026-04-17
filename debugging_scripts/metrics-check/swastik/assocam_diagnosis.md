# assocam — Metrics Diagnosis (Attempt 4)

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| `exact_match` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |
| `rouge_l` | formula_based | true | `helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics` |

## Current Run Spec Metrics (after Attempt 2 fix)
All three run spec functions (`assocam_4T1`, `assocam_7T1`, `assocam_10T1`) now use:
```python
MetricSpec(
    class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics",
    args={},
)
```
Adapter method: `ADAPT_MULTIPLE_CHOICE_JOINT`

## Actual Stats.json Metrics (m2)
- `classification_macro_f1`
- `classification_micro_f1`
- `exact_match`
- `num_instances`

## Missing Metrics
- **`rouge_l`**: Still missing after Attempt 2. The MetricSpec class is now correct (`compute_reference_metrics`), but HELM's `compute_reference_metrics` function only runs the ROUGE-L computation branch when the adapter is `ADAPT_GENERATION`. When the adapter is `ADAPT_MULTIPLE_CHOICE_JOINT`, HELM routes through the classification metric branch, producing `classification_macro_f1`, `classification_micro_f1`, and `exact_match` (option-selection accuracy), but skipping ROUGE-L entirely.

## Root Cause
**Adapter mismatch**: `ADAPT_MULTIPLE_CHOICE_JOINT` with `compute_reference_metrics` skips the ROUGE-L computation branch. The registry expects `rouge_l`, which `compute_reference_metrics` only produces under `ADAPT_GENERATION` mode. The scenario creates references as single letters (A–J) tagged with `CORRECT_TAG`, which is compatible with generation mode. Switching the adapter to `ADAPT_GENERATION` will cause `compute_reference_metrics` to compare the model's generated text against the correct reference letter using both `exact_match` and `rouge_l`.

## Proposed Fix
In `run_specs/assocam_run_specs.py`:
1. Change the import from `ADAPT_MULTIPLE_CHOICE_JOINT` to `ADAPT_GENERATION`
2. Update all three run_spec functions to use `method=ADAPT_GENERATION` in `AdapterSpec`
3. The MetricSpec (`compute_reference_metrics`) stays as-is — it's correct
4. Under generation mode: `exact_match` checks if the generated text matches the correct letter, and `rouge_l` computes ROUGE-L against it (effectively 1.0 for match, 0.0 for mismatch on single-letter references)
