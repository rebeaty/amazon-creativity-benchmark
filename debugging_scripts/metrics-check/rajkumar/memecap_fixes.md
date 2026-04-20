## memecap — Fix Applied (Attempt 1, 2026-04-17)
- **Root cause**: Registry names metric `f1` but `BasicGenerationMetric` produces `f1_score`. Registry naming error.
- **Files changed**: None (run spec already produces correct metrics under `f1_score`)
- **Change summary**: No code change needed. Both `exact_match` and `f1_score` present in Phase 1 stats.
- **Registry bug (⚠ tell Vijeta)**: `f1` in registry should be `f1_score`
- **Result**: Force-marked DONE — metric computed correctly, registry naming discrepancy only
