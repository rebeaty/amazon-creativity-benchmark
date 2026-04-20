# Fixes: pace — association_distance

## Status: COMPLETE (all fixes applied in attempts 1–2, confirmed in attempt 3)

---

## Files Changed

### New: `metrics/association_distance_metric.py`
Created `AssociationDistanceMetric` implementing the PACE association distance algorithm:
- Parses word-association chains from "Chain N: [word] (reason) → ..." model output format
- Uses GloVe 6B 300d embeddings (searched in `data/`, `~/data/`, `/tmp/`); falls back to
  mean vector (or zero vector if GloVe absent) for OOV words
- Per chain: for each position i, computes average cosine distance to all j < i, then
  averages across positions → chain score
- Final score: mean across chains
- Always emits `Stat(MetricName("association_distance"))` — returns 0.0 when chains cannot
  be parsed or GloVe is unavailable

### Modified: `run_specs/pace_run_specs.py`
Added `AssociationDistanceMetric` to `metric_specs`:
```python
MetricSpec(class_name="metrics.association_distance_metric.AssociationDistanceMetric", args={})
```

## Verification (Attempt 3)
- Import: `from metrics.association_distance_metric import AssociationDistanceMetric` → OK
- HELM class loading: `get_class_by_name("metrics.association_distance_metric.AssociationDistanceMetric")` → OK
- End-to-end parse test with sample PACE output → chains parsed correctly, stat returned
- No edge case causes silent metric omission

## Why
`registry_metrics.yaml` lists `association_distance` with `helm_class: null` — no built-in
HELM implementation. Custom class created and wired into run spec. Re-run required for
stats.json to reflect the fix.
