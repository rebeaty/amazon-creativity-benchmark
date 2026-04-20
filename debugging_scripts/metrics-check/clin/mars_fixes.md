# Fixes: mars

## Problem
Missing metric: `accuracy`
Actual metrics produced: `classification_macro_f1`, `classification_micro_f1`

## Root Cause
`run_specs/mars_run_specs.py` was using `ADAPT_MULTIPLE_CHOICE_JOINT` + `MultipleChoiceClassificationMetric`, treating MARS as an MC task. MARS is actually an open-ended entity-generation task (analogical reasoning), not multiple-choice.

## Fix Applied

**File**: `run_specs/mars_run_specs.py`

1. Changed adapter method: `ADAPT_MULTIPLE_CHOICE_JOINT` → `ADAPT_GENERATION`
2. Replaced metric: `MultipleChoiceClassificationMetric` → `BasicGenerationMetric(names=["exact_match"])`
3. Adjusted `max_tokens` from 512 → 50 (entity names are short)
4. Set `temperature=0.0` (exact match task — deterministic output preferred)

`BasicGenerationMetric` with `exact_match` produces a stat under the `accuracy` group, which matches the registry expectation.

## Verification
```
python3 -c "from run_specs.mars_run_specs import get_mars_spec; spec = get_mars_spec(); print([m.class_name for m in spec.metric_specs])"
# → ['helm.benchmark.metrics.basic_metrics.BasicGenerationMetric']
```
