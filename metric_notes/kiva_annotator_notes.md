# Annotator Notes: KiVA

## No Custom Annotators Required

**KiVA uses standard exact-match evaluation**, not LLM-as-judge or human annotation.

**Evaluation type**: `exact_match`

**Metric**: Accuracy (% of correct answers)

**Why no annotators?**
- Task has objective correct answers (A, B, or C)
- Ground truth is provided in the dataset
- Simple string matching determines correctness
- No subjective judgment needed

## Evaluation Details

For complete evaluation documentation, see: **[evaluation_notes.md](./evaluation_notes.md)**

## Quick Summary

**What is evaluated**:
- Model must choose which transformation (A/B/C) matches the training example
- 3-way multiple choice with single correct answer

**How it's evaluated**:
- Extract model's answer: `(A)`, `(B)`, or `(C)`
- Compare to correct answer in metadata
- Count matches to compute accuracy

**Metric computation**:
```python
from helm.benchmark.metrics.metric_specs import get_exact_match_metric_specs

# Standard HELM exact match evaluation
metric_specs = get_exact_match_metric_specs()
```

**No additional annotation needed** - this is a fully automated evaluation.

## If You Need Annotators (You Don't)

If you were to add human annotation or LLM-based judging (not recommended for this task), you could evaluate:

- **Answer quality**: Is the model's reasoning correct even if final answer is wrong?
- **Explanation coherence**: How well does the model explain its choice?
- **Visual understanding**: Does the model correctly identify transformation features?

However, **these are NOT part of the standard KiVA benchmark**. The official evaluation is simply accuracy on the final answer choice.

See [evaluation_notes.md](./evaluation_notes.md) for the complete evaluation protocol.
