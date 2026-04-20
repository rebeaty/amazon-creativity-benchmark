# Hummus Diagnosis

## Expected vs Actual Metrics

| | Metrics |
|---|---|
| **Expected (m1)** | `accuracy`, `llm_judge_quality` |
| **Actual (m2)** | `bleu_1`, `bleu_4`, `exact_match`, `f1_score`, `llm_judge_quality`, `quasi_exact_match`, `rouge_l` |
| **Missing** | `accuracy` |

## Root Cause

The hummus scenario defaults to the `"classification"` subset (Task 1), which creates MCQ-style instances:
- Two references per instance: `"Yes"` and `"No"`, one tagged `CORRECT_TAG`
- This structure requires `ADAPT_MULTIPLE_CHOICE_JOINT` + `MultipleChoiceClassificationMetric` to produce `accuracy`

The run_spec was instead using:
- `ADAPT_GENERATION` adapter — causes free-text generation rather than MCQ ranking
- `BasicGenerationMetric` — produces text-similarity metrics (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) but NOT `accuracy`

Reference: `d_humor_run_specs.py` uses the identical pattern (binary Yes/No classification) with `ADAPT_MULTIPLE_CHOICE_JOINT` + `MultipleChoiceClassificationMetric`, which correctly emits `accuracy`.

## Proposed Fix

1. Change adapter `method` from `ADAPT_GENERATION` to `ADAPT_MULTIPLE_CHOICE_JOINT`
2. Add `output_prefix="Answer: "` (standard MCQ convention)
3. Replace `BasicGenerationMetric` with `MultipleChoiceClassificationMetric`
4. Keep the LLM judge `MetricSpec` and `AnnotatorSpec` unchanged (already producing `llm_judge_quality`)
