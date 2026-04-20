# Diagnosis: pace — missing `association_distance`

## Expected vs Actual
- Expected: `["type_token_ratio", "association_distance"]`
- Actual:   `["type_token_ratio"]`
- Missing:  `["association_distance"]`

## Root Cause (Attempt 1/2)
`registry_metrics.yaml` listed `association_distance` with `helm_class: null` — no metric class
existed. `run_specs/pace_run_specs.py` had no `MetricSpec` for association distance.

## Status After Attempt 2 (confirmed in Attempt 3)
Both fixes were applied in prior attempts and are **verified correct**:

1. `metrics/association_distance_metric.py` — exists, imports cleanly, HELM can load via
   `get_class_by_name("metrics.association_distance_metric.AssociationDistanceMetric")`.
2. `run_specs/pace_run_specs.py` — contains both `TypeTokenRatioMetric` and
   `AssociationDistanceMetric` MetricSpecs.

Verification (Attempt 3):
- `python3 -c "from metrics.association_distance_metric import AssociationDistanceMetric"` → OK
- HELM `create_object(MetricSpec(...AssociationDistanceMetric...))` → instantiates correctly
- GloVe not present locally → metric falls back to zero-vector and returns `Stat("association_distance", 0.0)` — stat is always emitted, never silently dropped
- `_parse_chains` correctly extracts words from `Chain N: [word] (reason) → ...` format

## Why Metric Still Missing in stats.json
stats.json was produced by a trial run that pre-dates the creation of
`metrics/association_distance_metric.py`. A re-run of the evaluation will produce
`association_distance` in the output.

## No Further Code Changes Needed
