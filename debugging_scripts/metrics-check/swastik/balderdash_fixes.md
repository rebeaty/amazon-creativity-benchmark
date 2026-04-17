# balderdash — Fixes Applied

## Attempt 1
### `scenarios/balderdash_scenario.py`
- Added `CORRECT_TAG`, `Output`, `Reference` to imports
- Changed `references=[]` to `references=[Reference(Output(text=real_definition), tags=[CORRECT_TAG])]`

**Why:** `BasicMetric` needs a `Reference` with `CORRECT_TAG` to compute `accuracy`.

## Attempt 2
### `run_specs/balderdash_run_specs.py`
**Change:** Replaced TODO-fallback MetricSpec with registry-mandated `BasicMetric`.

```python
# Before
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
           args={"names": ["exact_match"]})

# After
MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})
```

**Why:** Registry specifies `helm_class: helm.benchmark.metrics.basic_metrics.BasicMetric` for `accuracy`. `BasicGenerationMetric` with `exact_match` produces `exact_match`, not `accuracy`.

## Verification
`python3 -m py_compile run_specs/balderdash_run_specs.py` → Syntax OK
