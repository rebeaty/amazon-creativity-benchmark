# munch — Fixes Summary

## Attempt 6 Fix

### File Modified
`run_specs/munch_run_specs.py`

### Changes
1. Changed adapter import: `ADAPT_MULTIPLE_CHOICE_JOINT` → `ADAPT_GENERATION`
2. Changed `method=ADAPT_MULTIPLE_CHOICE_JOINT` → `method=ADAPT_GENERATION`
3. Cleared `output_prefix` (`"Answer: "` → `""`) — prompt already ends with "Correct answer: Option"
4. Set `temperature=0.7` → `temperature=0.0` (MCQ, deterministic)
5. Set `max_tokens=512` → `max_tokens=16` (only a single letter needed)

### Why
`ADAPT_MULTIPLE_CHOICE_JOINT` causes `BasicMetric` to compute `classification_macro_f1`/`classification_micro_f1` instead of `accuracy`. The scenario already builds the full MCQ prompt (with all four options) in `Input.text`, so the MCQ joint adapter is not needed. Switching to `ADAPT_GENERATION` allows `BasicMetric` to produce `accuracy` by comparing the model's generated letter against the correct reference. This is the same root cause and fix that resolved `lcc_metaphor` at attempt 3.

### Verification
`python3 -m py_compile run_specs/munch_run_specs.py` — Syntax OK

---

## History

### Attempt 5 Fix
Replaced `MultipleChoiceClassificationMetric` with `BasicMetric` (necessary but not sufficient — adapter was not changed).

### Attempts 1–4
Earlier attempts also failed to address the adapter issue.
