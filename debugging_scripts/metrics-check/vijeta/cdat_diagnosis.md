# CDAT Metrics Diagnosis

## Expected vs Actual Metrics

| Metric | Expected | Actual | Status |
|---|---|---|---|
| appropriateness | ✓ | ✗ | MISSING |
| novelty | ✓ | ✗ | MISSING |
| creativity_score | ✓ | ✓ | OK |

## Root Cause Analysis

Two compounding issues:

### Issue 1: Stale stats.json (primary)
The `stats.json` at `benchmark_output/runs/trial/cdat:model=google_gemini-2.5-flash-lite/` was generated
before `metrics/creativity_score_metric.py` was updated to emit all three stats. The committed metric code
only returned `creativity_score`; the fixed working-tree version already returns `novelty` and
`appropriateness` as well. The trial stats.json was never regenerated after the metric fix.

### Issue 2: Wrong scenario class_name in run_spec (secondary)
`run_specs/cdat_run_specs.py` referenced `scenarios_new.cdat_scenario.CDATScenario`, but the correct
module is `scenarios.cdat_scenario.CDATScenario`. This caused HELM to fail when attempting to reload
the scenario (even though `scenario_state.json` was cached), blocking re-evaluation.

## Fix Applied (Attempt 4)

1. **Fixed `run_specs/cdat_run_specs.py`**: Changed `scenarios_new.cdat_scenario.CDATScenario` →
   `scenarios.cdat_scenario.CDATScenario`.

2. **Re-ran HELM trial**: Executed `helm_run --run-entries "cdat:model=google/gemini-2.5-flash-lite"
   --suite trial --max-eval-instances 10`. HELM used the cached `scenario_state.json` (no new inference
   calls needed) and recomputed metrics with the fixed `CreativityScoreMetric`, which now emits
   `creativity_score`, `novelty`, and `appropriateness`.

3. **Verified**: `metrics_check.py cdat` now reports `status: pass` with all three metrics present.
