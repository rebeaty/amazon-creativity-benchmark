# Diagnosis: protein_bench Missing Metrics

## Expected vs Actual

| Metric | Expected | Present |
|--------|----------|---------|
| validity | ✓ | ✓ |
| plddt_score | ✓ | ✗ |
| sctm_score | ✓ | ✗ |
| novelty_tmscore | ✓ | ✗ |

## Root Cause

`run_specs/protein_bench_run_specs.py` only registers one MetricSpec:
```python
metric_specs = [
    MetricSpec(class_name="metrics.validity_metric.ValidityMetric", args={}),
]
```

The three missing metrics (`plddt_score`, `sctm_score`, `novelty_tmscore`) have `helm_class: null` in `registry_metrics.yaml` — no metric classes existed for them. They were never added to the run spec.

## Metric Details (from registry_metrics.yaml)

| Metric | Type | Model Required |
|--------|------|----------------|
| plddt_score | model_based | ESMFold |
| sctm_score | model_based | ESMFold |
| novelty_tmscore | model_based | ESMFold + PDB lookup |

From `protein_bench_eval_metrics_notes.md`:
- **pLDDT**: Mean per-residue confidence from ESMFold (0–100 → normalized to 0–1)
- **scTM**: Self-consistency TM-score; ESMFold's `ptm` output is the practical proxy
- **Novelty**: 1 − max(TM-score vs PDB); approximated as `1 − pTM` when PDB lookup is unavailable

## Proposed Fix

1. Create `metrics/protein_bench_metrics.py` with:
   - `PlddtScoreMetric` — folds sequence with ESMFold, returns mean pLDDT / 100
   - `SctmScoreMetric` — returns ESMFold `ptm` as scTM proxy
   - `NoveltyTmScoreMetric` — returns `1 − ptm` as novelty proxy

2. Update `run_specs/protein_bench_run_specs.py` to add MetricSpecs for all three.

All three classes share a single lazy-loaded ESMFold singleton (thread-safe). They return 0.0 gracefully when ESMFold is not installed, so stats are always emitted.
