# balderdash — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| accuracy | formula_based | true | `helm.benchmark.metrics.basic_metrics.BasicMetric` |

## Current Run Spec Metrics (attempt 2 state)
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match"]})`
- `annotators=None`

## Actual Stats.json Metrics (m2)
- `exact_match`

## Missing Metrics
- **accuracy**: The run_spec uses `BasicGenerationMetric` with `args={"names": ["exact_match"]}` (TODO fallback), which produces `exact_match` not `accuracy`. The scenario was already fixed in attempt 1 to emit `Reference` objects with `CORRECT_TAG` — only the run_spec MetricSpec still needs updating.

## Root Cause
Attempt 1 correctly fixed the scenario to include `Reference(Output(text=real_definition), tags=[CORRECT_TAG])` objects, but the run_spec was either reverted or not fully saved. The MetricSpec still uses `BasicGenerationMetric` + `exact_match` instead of `BasicMetric` + `{}`.

## Fix Applied (attempt 2)
**run_specs/balderdash_run_specs.py**: Replaced `BasicGenerationMetric(names=["exact_match"])` with `BasicMetric(args={})` — the class that produces the `accuracy` stat. Scenario already correct from attempt 1.
